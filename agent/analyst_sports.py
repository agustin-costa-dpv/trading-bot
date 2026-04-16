"""
agent/analyst_sports.py
Analista deportivo del bot. Evalúa value en mercados 1X2 de fútbol en Azuro.

Flujo:
1. Trae mercados de fútbol desde bot/azuro.py
2. Filtra los 1X2 (condition con outcomes 29=Home, 30=Draw, 31=Away)
3. Guarda odds actuales en Supabase (para tracking de movimiento)
4. Calcula probabilidad implícita desde las odds
5. Enriquece con datos xG (bot/xg_data.py)
6. Enriquece con movimiento de odds (bot/odds_movement.py)
7. Pasa contexto completo a Claude para que estime probabilidad real
8. Si prob real > prob implícita + margen → señal de value

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
from bot.odds_movement import guardar_odds, get_movimiento_odds
from agent.sessions import get_sesion_actual


# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────

CLIENTE_CLAUDE = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODELO_HAIKU = "claude-haiku-4-5-20251001"
MODELO_SONNET = "claude-sonnet-4-6"

OUTCOME_ID_HOME = "29"
OUTCOME_ID_DRAW = "30"
OUTCOME_ID_AWAY = "31"
OUTCOME_IDS_1X2 = {OUTCOME_ID_HOME, OUTCOME_ID_DRAW, OUTCOME_ID_AWAY}

MARGEN_MINIMO_VALUE = 0.05
PROBABILIDAD_MINIMA_FINAL = 0.55
ODDS_MINIMAS = 1.5
ODDS_MAXIMAS = 6.0

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
# Filtrado y priorización
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
    """Devuelve mercados 1X2 de fútbol, ligas prioritarias primero."""
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


def analisis_sonnet_partido(mercado: Mercado1X2, xg_data: dict, odds_mov_resumen: str) -> dict:
    """
    Sonnet hace el análisis profundo con contexto xG + movimiento de odds.
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

{odds_mov_resumen}

Tu tarea:
1. Estimá probabilidad REAL de cada resultado basándote en:
   - Los datos xG provistos (priorizalos sobre intuición general)
   - Si hay flag de REGRESIÓN A LA MEDIA, ponderalo fuerte
   - El movimiento de odds: si las odds subieron pero el xG favorece al equipo,
     es divergencia → señal fuerte de value
   - Forma reciente, posición en tabla, head-to-head, localía
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
    3. Enriquecer con xG
    4. Enriquecer con movimiento de odds
    5. Sonnet estima probabilidades (con xG + movimiento)
    6. Aplica bonus de regresión y divergencia
    7. Construye SenalDeportiva si hay value
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

    # 3. Enriquecer con xG
    try:
        xg_data = enriquecer_mercado_con_xg(mercado.titulo, mercado.liga)
    except Exception as e:
        print(f"⚠️  Error xG en {mercado.titulo}: {e}")
        xg_data = {"disponible": False, "score_bonus": 1.0, "resumen": "Error al obtener xG."}

    # 4. Movimiento de odds
    odds_mov = None
    odds_mov_resumen = "Movimiento de odds: sin historial suficiente aún (primeros ciclos)."
    try:
        odds_mov = get_movimiento_odds(
            mercado.game_id,
            mercado.odds_home,
            mercado.odds_draw,
            mercado.odds_away,
        )
        if odds_mov:
            odds_mov_resumen = odds_mov.resumen()
    except Exception as e:
        print(f"⚠️  Error odds_movement en {mercado.titulo}: {e}")

    # 5. Análisis Sonnet (con xG + movimiento de odds)
    try:
        analisis = analisis_sonnet_partido(mercado, xg_data, odds_mov_resumen)
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

    # 6a. Bonus por regresión a la media (xG)
    score_bonus = xg_data.get("score_bonus", 1.0)
    if score_bonus > 1.0:
        value = value * score_bonus
        print(f"  📈 Bonus regresión xG ({score_bonus}x): value={value:.3f}")

    # 6b. Bonus por divergencia odds (mercado se alejó pero xG favorece)
    if odds_mov:
        xg_favorece = xg_data.get("disponible", False) and (
            (seleccion_str == "HOME" and xg_data.get("home") and
             xg_data["home"].xg_promedio > xg_data["home"].xga_promedio) or
            (seleccion_str == "AWAY" and xg_data.get("away") and
             xg_data["away"].xg_promedio > xg_data["away"].xga_promedio)
        )
        if odds_mov.hay_divergencia(seleccion_str, xg_favorece):
            value = value * 1.2
            print(f"  📈 Bonus divergencia odds (1.2x): value={value:.3f}")

    # 7. Validaciones finales
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
    Guarda odds al inicio del ciclo para tracking de movimiento.
    Ligas prioritarias se analizan primero.
    """
    sesion = get_sesion_actual()
    if not sesion.puede_operar:
        print(f"⏸️  Sesión {sesion.nombre} — análisis deportivo pausado")
        return []

    mercados = obtener_mercados_1x2_futbol()
    prioritarios = sum(1 for m in mercados if m.es_liga_prioritaria)
    print(f"🔎 Analizando {len(mercados)} partidos 1X2 ({prioritarios} en ligas prioritarias primero)...")

    # Guardar odds al inicio del ciclo (para movimiento futuro)
    try:
        guardar_odds(mercados)
    except Exception as e:
        print(f"⚠️  Error guardando odds históricas: {e}")

    senales = []
    for m in mercados:
        senal = analizar_partido(m)
        if senal:
            senales.append(senal)
            print(f"  ✅ Value: {senal.titulo} → {senal.seleccion.value} @ {senal.odds:.2f} (value={senal.value:.1%})")

    return senales


# ─────────────────────────────────────────────────────────────
# Testing — python -m agent.analyst_sports
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("TEST: Analyst deportivo (Azuro 1X2 + xG + odds movement)")
    print("=" * 60)

    mercados = obtener_mercados_1x2_futbol()
    print(f"\n📊 Mercados 1X2: {len(mercados)}")
    print(f"⭐ Ligas prioritarias: {sum(1 for m in mercados if m.es_liga_prioritaria)}")

    if mercados:
        print(f"\n🤖 Analizando primer partido...")
        senal = analizar_partido(mercados[0])
        if senal:
            print(f"\n✅ SEÑAL:")
            print(f"   {senal.titulo} → {senal.seleccion.value} @ {senal.odds:.2f}")
            print(f"   Value: {senal.value:+.1%} | Confianza: {senal.confianza}")
            print(f"   {senal.razon}")
        else:
            print("   ❌ Sin value")

    print("\n✅ Test terminado")
