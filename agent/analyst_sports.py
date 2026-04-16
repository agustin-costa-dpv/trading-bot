"""
agent/analyst_sports.py
Analista deportivo del bot. Evalúa value en mercados 1X2 de fútbol en Azuro.

Flujo:
1. Trae mercados de fútbol desde bot/azuro.py
2. Filtra los 1X2 (condition con outcomes 29=Home, 30=Draw, 31=Away)
3. Calcula probabilidad implícita desde las odds
4. Enriquece con datos xG (bot/xg_data.py)
5. Pasa contexto a Claude (equipos, liga, odds, xG) para que estime probabilidad real
6. Si la probabilidad real > probabilidad implícita + margen → señal de value

Devuelve SenalDeportiva lista para main.py → risk.py → ejecución.
"""

import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional

from anthropic import Anthropic

from bot.azuro import listar_mercados
from bot.xg_data import enriquecer_mercado_con_xg
from agent.sessions import get_sesion_actual


# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────

CLIENTE_CLAUDE = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODELO_HAIKU = "claude-haiku-4-5-20251001"
MODELO_SONNET = "claude-sonnet-4-6"

# Outcome IDs del mercado 1X2 en Azuro (Full Time Result en fútbol)
OUTCOME_ID_HOME = "29"
OUTCOME_ID_DRAW = "30"
OUTCOME_ID_AWAY = "31"
OUTCOME_IDS_1X2 = {OUTCOME_ID_HOME, OUTCOME_ID_DRAW, OUTCOME_ID_AWAY}

# Umbrales de value
MARGEN_MINIMO_VALUE = 0.05        # probabilidad real debe superar la implícita por 5%
PROBABILIDAD_MINIMA_FINAL = 0.55  # prob real mínima para considerar la apuesta
ODDS_MINIMAS = 1.5                # odds muy bajas = poco upside
ODDS_MAXIMAS = 6.0                # odds muy altas = especulativo

# Ligas con mayor ineficiencia en Azuro (oráculos calibran peor → más edge)
LIGAS_PRIORITARIAS = {
    "eredivisie", "liga portugal", "championship",
    "superliga argentina", "bundesliga 2", "2. bundesliga",
}


# ─────────────────────────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────────────────────────

class SeleccionDeportiva(str, Enum):
    HOME = "HOME"
    DRAW = "DRAW"
    AWAY = "AWAY"


@dataclass
class Mercado1X2:
    """Representación limpia de un mercado 1X2."""
    game_id: str
    titulo: str
    liga: str
    pais: str
    condition_id: str
    margen: float
    odds_home: float
    odds_draw: float
    odds_away: float
    outcome_id_home: str
    outcome_id_draw: str
    outcome_id_away: str
    es_liga_prioritaria: bool = False

    def prob_implicita_home(self) -> float:
        return 1.0 / self.odds_home if self.odds_home else 0.0

    def prob_implicita_draw(self) -> float:
        return 1.0 / self.odds_draw if self.odds_draw else 0.0

    def prob_implicita_away(self) -> float:
        return 1.0 / self.odds_away if self.odds_away else 0.0


@dataclass
class SenalDeportiva:
    """Señal final que sale del analyst_sports y va a risk.py."""
    game_id: str
    titulo: str
    liga: str
    seleccion: SeleccionDeportiva
    odds: float
    probabilidad_implicita: float
    probabilidad_real: float
    value: float
    confianza: str
    razon: str
    condition_id: str
    outcome_id: str
    plataforma: str = "azuro"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seleccion"] = self.seleccion.value
        return d


# ─────────────────────────────────────────────────────────────
# Filtrado y priorización de mercados
# ─────────────────────────────────────────────────────────────

def _es_mercado_1x2(mercado: dict) -> bool:
    outcomes = mercado.get("outcomes", [])
    if len(outcomes) != 3:
        return False
    ids = {o["id"] for o in outcomes}
    return ids == OUTCOME_IDS_1X2


def _es_liga_prioritaria(liga: str) -> bool:
    return any(lp in liga.lower() for lp in LIGAS_PRIORITARIAS)


def _parsear_mercado_1x2(mercado: dict) -> Optional[Mercado1X2]:
    if not _es_mercado_1x2(mercado):
        return None

    odds_por_id = {o["id"]: o["odds"] for o in mercado["outcomes"]}
    liga = mercado.get("liga", "")

    return Mercado1X2(
        game_id=mercado["game_id"],
        titulo=mercado["titulo"],
        liga=liga,
        pais=mercado.get("pais", ""),
        condition_id=mercado["condition_id"],
        margen=mercado.get("margen", 0.0),
        odds_home=odds_por_id[OUTCOME_ID_HOME],
        odds_draw=odds_por_id[OUTCOME_ID_DRAW],
        odds_away=odds_por_id[OUTCOME_ID_AWAY],
        outcome_id_home=OUTCOME_ID_HOME,
        outcome_id_draw=OUTCOME_ID_DRAW,
        outcome_id_away=OUTCOME_ID_AWAY,
        es_liga_prioritaria=_es_liga_prioritaria(liga),
    )


def obtener_mercados_1x2_futbol() -> list[Mercado1X2]:
    """
    Devuelve mercados 1X2 de fútbol activos en Azuro.
    Las ligas prioritarias van primero.
    """
    crudos = listar_mercados()
    futbol = [
        m for m in crudos
        if m.get("deporte", "").lower() in ("football", "soccer")
    ]

    mercados_1x2 = []
    for m in futbol:
        parsed = _parsear_mercado_1x2(m)
        if parsed:
            mercados_1x2.append(parsed)

    # Ligas prioritarias primero (más ineficiencias explotables)
    mercados_1x2.sort(key=lambda m: (0 if m.es_liga_prioritaria else 1))
    return mercados_1x2


# ─────────────────────────────────────────────────────────────
# Análisis con Claude
# ─────────────────────────────────────────────────────────────

def _llamar_claude(modelo: str, prompt: str, max_tokens: int = 500) -> dict:
    response = CLIENTE_CLAUDE.messages.create(
        model=modelo,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    texto = response.content[0].text.strip()
    if texto.startswith("```"):
        texto = texto.split("```")[1]
        if texto.startswith("json"):
            texto = texto[4:]
        texto = texto.strip()
    return json.loads(texto)


def filtro_haiku_partido(mercado: Mercado1X2) -> dict:
    """
    Haiku hace un primer filtrado rápido.
    Devuelve {conozco_equipos, vale_analisis_profundo, razon}
    """
    liga_tag = "⭐ LIGA PRIORITARIA (mayor ineficiencia de odds)" if mercado.es_liga_prioritaria else ""

    prompt = f"""Sos un analista de fútbol. Evaluá este partido rápidamente.

PARTIDO: {mercado.titulo}
LIGA: {mercado.liga} ({mercado.pais}) {liga_tag}
ODDS:
- Gana local: {mercado.odds_home:.2f} (prob implícita: {mercado.prob_implicita_home():.1%})
- Empate:    {mercado.odds_draw:.2f} (prob implícita: {mercado.prob_implicita_draw():.1%})
- Gana visitante: {mercado.odds_away:.2f} (prob implícita: {mercado.prob_implicita_away():.1%})
- Margen casa: {mercado.margen:.1%}

Respondé SOLO con JSON (sin markdown):
{{
  "conozco_equipos": true/false,
  "vale_analisis_profundo": true/false,
  "razon": "1 frase corta"
}}

Si NO conocés a los equipos o la liga → vale_analisis_profundo = false.
Si las odds parecen muy obvias sin value → false.
Si es liga prioritaria o hay algo interesante en las odds → true.
"""
    return _llamar_claude(MODELO_HAIKU, prompt, max_tokens=300)


def analisis_sonnet_partido(mercado: Mercado1X2, xg_data: dict) -> dict:
    """
    Sonnet hace el análisis profundo con contexto xG incluido.
    Devuelve {prob_real_home, prob_real_draw, prob_real_away, seleccion_con_value, confianza, razon}
    """
    xg_resumen = xg_data.get("resumen", "Datos xG no disponibles para esta liga.")
    liga_tag = "⭐ LIGA PRIORITARIA — los oráculos de Azuro suelen tener más ineficiencias aquí." if mercado.es_liga_prioritaria else ""

    prompt = f"""Sos un trader profesional de apuestas deportivas analizando value en un partido de fútbol.

PARTIDO: {mercado.titulo}
LIGA: {mercado.liga} ({mercado.pais})
{liga_tag}

ODDS OFRECIDAS:
- Local: {mercado.odds_home:.2f} (prob implícita: {mercado.prob_implicita_home():.1%})
- Empate: {mercado.odds_draw:.2f} (prob implícita: {mercado.prob_implicita_draw():.1%})
- Visitante: {mercado.odds_away:.2f} (prob implícita: {mercado.prob_implicita_away():.1%})
- Margen casa: {mercado.margen:.1%}

{xg_resumen}

Tu tarea:
1. Estimá probabilidad REAL de cada resultado basándote en:
   - Los datos xG provistos (priorizalos sobre intuición general)
   - Si hay flag de REGRESIÓN A LA MEDIA, ponderalo fuerte en tu estimación
   - Forma reciente, posición en tabla, head-to-head, localía
   - Lesiones o ausencias conocidas
2. Compará prob real vs prob implícita → identificá VALUE (real > implícita)
3. Si ninguna opción tiene value claro (>5%), respondé seleccion=NINGUNA

Respondé SOLO con JSON (sin markdown):
{{
  "prob_real_home": 0.XX,
  "prob_real_draw": 0.XX,
  "prob_real_away": 0.XX,
  "seleccion_con_value": "HOME" | "DRAW" | "AWAY" | "NINGUNA",
  "confianza": "ALTA" | "MEDIA" | "BAJA",
  "razon": "1-2 frases explicando el análisis y la selección"
}}

Las 3 probabilidades deben sumar 1.0 (±0.02 de error aceptable).
Si no tenés info concreta de los equipos, sé conservador: seleccion_con_value = "NINGUNA".
"""
    return _llamar_claude(MODELO_SONNET, prompt, max_tokens=600)


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────

def analizar_partido(mercado: Mercado1X2) -> Optional[SenalDeportiva]:
    """
    Pipeline para un partido:
    1. Filtro odds extremas
    2. Haiku decide si vale el análisis profundo
    3. Enriquecer con datos xG
    4. Sonnet estima probabilidades reales (con contexto xG)
    5. Aplica bonus de regresión a la media si corresponde
    6. Construye SenalDeportiva si hay value
    """
    # 1. Filtro por odds razonables
    odds_validas = [mercado.odds_home, mercado.odds_draw, mercado.odds_away]
    if any(o < ODDS_MINIMAS or o > ODDS_MAXIMAS for o in odds_validas):
        return None

    # 2. Filtro Haiku
    try:
        filtro = filtro_haiku_partido(mercado)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"⚠️  Error Haiku en {mercado.titulo}: {e}")
        return None

    if not filtro.get("vale_analisis_profundo", False):
        return None

    # 3. Enriquecer con xG (falla silenciosamente si liga no soportada)
    try:
        xg_data = enriquecer_mercado_con_xg(mercado.titulo, mercado.liga)
    except Exception as e:
        print(f"⚠️  Error xG en {mercado.titulo}: {e}")
        xg_data = {"disponible": False, "score_bonus": 1.0, "resumen": "Error al obtener xG."}

    # 4. Análisis Sonnet (con contexto xG)
    try:
        analisis = analisis_sonnet_partido(mercado, xg_data)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"⚠️  Error Sonnet en {mercado.titulo}: {e}")
        return None

    seleccion_str = analisis.get("seleccion_con_value", "NINGUNA")
    if seleccion_str == "NINGUNA":
        return None

    # Mapear selección
    if seleccion_str == "HOME":
        seleccion = SeleccionDeportiva.HOME
        odds = mercado.odds_home
        prob_implicita = mercado.prob_implicita_home()
        prob_real = analisis["prob_real_home"]
        outcome_id = mercado.outcome_id_home
    elif seleccion_str == "DRAW":
        seleccion = SeleccionDeportiva.DRAW
        odds = mercado.odds_draw
        prob_implicita = mercado.prob_implicita_draw()
        prob_real = analisis["prob_real_draw"]
        outcome_id = mercado.outcome_id_draw
    else:  # AWAY
        seleccion = SeleccionDeportiva.AWAY
        odds = mercado.odds_away
        prob_implicita = mercado.prob_implicita_away()
        prob_real = analisis["prob_real_away"]
        outcome_id = mercado.outcome_id_away

    value = prob_real - prob_implicita

    # 5. Bonus por regresión a la media (xG dice que el equipo va a corregir)
    score_bonus = xg_data.get("score_bonus", 1.0)
    if score_bonus > 1.0:
        value = value * score_bonus
        print(f"  📈 Bonus regresión aplicado ({score_bonus}x): value={value:.3f}")

    # 6. Validaciones finales
    if value < MARGEN_MINIMO_VALUE:
        return None
    if prob_real < PROBABILIDAD_MINIMA_FINAL:
        return None

    return SenalDeportiva(
        game_id=mercado.game_id,
        titulo=mercado.titulo,
        liga=f"{mercado.liga} ({mercado.pais})",
        seleccion=seleccion,
        odds=odds,
        probabilidad_implicita=prob_implicita,
        probabilidad_real=prob_real,
        value=value,
        confianza=analisis.get("confianza", "MEDIA"),
        razon=analisis.get("razon", ""),
        condition_id=mercado.condition_id,
        outcome_id=outcome_id,
    )


def analizar_todos_los_partidos() -> list[SenalDeportiva]:
    """
    Analiza todos los partidos de fútbol 1X2 activos.
    Las ligas prioritarias se analizan primero.
    Devuelve solo los que generaron señal con value.
    """
    sesion = get_sesion_actual()
    if not sesion.puede_operar:
        print(f"⏸️  Sesión {sesion.nombre} — análisis deportivo pausado")
        return []

    mercados = obtener_mercados_1x2_futbol()
    prioritarios = sum(1 for m in mercados if m.es_liga_prioritaria)
    print(f"🔎 Analizando {len(mercados)} partidos 1X2 ({prioritarios} en ligas prioritarias primero)...")

    senales = []
    for m in mercados:
        senal = analizar_partido(m)
        if senal:
            senales.append(senal)
            print(f"  ✅ Value detectado: {senal.titulo} → {senal.seleccion.value} @ {senal.odds:.2f} (value={senal.value:.1%})")

    return senales


# ─────────────────────────────────────────────────────────────
# Testing — python -m agent.analyst_sports
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("TEST: Analyst deportivo (Azuro 1X2 fútbol + xG)")
    print("=" * 60)

    mercados = obtener_mercados_1x2_futbol()
    print(f"\n📊 Mercados 1X2 encontrados: {len(mercados)}")
    prioritarios = [m for m in mercados if m.es_liga_prioritaria]
    print(f"⭐ Ligas prioritarias: {len(prioritarios)}")

    if mercados:
        print(f"\n🔎 Primeros 3 mercados:")
        for m in mercados[:3]:
            tag = "⭐" if m.es_liga_prioritaria else "  "
            print(f"  {tag} {m.titulo} ({m.liga})")
            print(f"     H: {m.odds_home:.2f} | D: {m.odds_draw:.2f} | A: {m.odds_away:.2f}")

    if mercados:
        print(f"\n🤖 Analizando primer partido con xG + Claude...")
        senal = analizar_partido(mercados[0])
        if senal:
            print(f"\n✅ SEÑAL DEPORTIVA:")
            print(f"   Partido:    {senal.titulo}")
            print(f"   Selección:  {senal.seleccion.value}")
            print(f"   Odds:       {senal.odds:.2f}")
            print(f"   Prob impl:  {senal.probabilidad_implicita:.1%}")
            print(f"   Prob real:  {senal.probabilidad_real:.1%}")
            print(f"   Value:      {senal.value:+.1%}")
            print(f"   Confianza:  {senal.confianza}")
            print(f"   Razón:      {senal.razon}")
        else:
            print("   ❌ Sin value detectado")

    print("\n✅ Test terminado")
