"""
agent/analyst_sports.py
Analista deportivo del bot. Evalúa value en mercados 1X2 de fútbol en Azuro.

Flujo:
1. Trae mercados de fútbol desde bot/azuro.py
2. Filtra los 1X2 (condition con outcomes 29=Home, 30=Draw, 31=Away)
3. Calcula probabilidad implícita desde las odds
4. Pasa contexto a Claude (equipos, liga, odds) para que estime probabilidad real
5. Si la probabilidad real > probabilidad implícita + margen → señal de value

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
MARGEN_MINIMO_VALUE = 0.05       # probabilidad real debe superar la implícita por 5%
PROBABILIDAD_MINIMA_FINAL = 0.55  # prob real mínima para considerar la apuesta
ODDS_MINIMAS = 1.5                # odds muy bajas = poco upside, no vale la pena
ODDS_MAXIMAS = 6.0                # odds muy altas = tirada, apuesta especulativa


# ─────────────────────────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────────────────────────

class SeleccionDeportiva(str, Enum):
    HOME = "HOME"   # gana local
    DRAW = "DRAW"   # empate
    AWAY = "AWAY"   # gana visitante


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
    probabilidad_implicita: float     # 1/odds
    probabilidad_real: float          # estimación de Claude
    value: float                       # prob_real - prob_implicita
    confianza: str                    # ALTA / MEDIA / BAJA
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
# Filtrado de mercados 1X2
# ─────────────────────────────────────────────────────────────

def _es_mercado_1x2(mercado: dict) -> bool:
    """
    Un mercado es 1X2 si tiene exactamente 3 outcomes con IDs 29, 30, 31.
    """
    outcomes = mercado.get("outcomes", [])
    if len(outcomes) != 3:
        return False
    ids = {o["id"] for o in outcomes}
    return ids == OUTCOME_IDS_1X2


def _parsear_mercado_1x2(mercado: dict) -> Optional[Mercado1X2]:
    """Convierte un mercado crudo en Mercado1X2 tipado."""
    if not _es_mercado_1x2(mercado):
        return None

    # Mapear outcomes por ID
    odds_por_id = {o["id"]: o["odds"] for o in mercado["outcomes"]}

    return Mercado1X2(
        game_id=mercado["game_id"],
        titulo=mercado["titulo"],
        liga=mercado.get("liga", ""),
        pais=mercado.get("pais", ""),
        condition_id=mercado["condition_id"],
        margen=mercado.get("margen", 0.0),
        odds_home=odds_por_id[OUTCOME_ID_HOME],
        odds_draw=odds_por_id[OUTCOME_ID_DRAW],
        odds_away=odds_por_id[OUTCOME_ID_AWAY],
        outcome_id_home=OUTCOME_ID_HOME,
        outcome_id_draw=OUTCOME_ID_DRAW,
        outcome_id_away=OUTCOME_ID_AWAY,
    )


def obtener_mercados_1x2_futbol() -> list[Mercado1X2]:
    """
    Devuelve todos los mercados 1X2 de fútbol activos en Azuro.
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
    return mercados_1x2


# ─────────────────────────────────────────────────────────────
# Análisis con Claude
# ─────────────────────────────────────────────────────────────

def _llamar_claude(modelo: str, prompt: str, max_tokens: int = 500) -> dict:
    """Llama a Claude y parsea JSON."""
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
    Haiku hace un primer filtrado rápido:
    - ¿Conozco a estos equipos? Si no, probabilidad razonable.
    - ¿Las odds parecen alineadas con el contexto obvio?
    - Filtro pre-Sonnet para no gastar en partidos que no valen
    """
    prompt = f"""Sos un analista de fútbol. Evaluá este partido rápidamente.

PARTIDO: {mercado.titulo}
LIGA: {mercado.liga} ({mercado.pais})
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
Si las odds parecen muy obvias sin value (ej: favorito claro con odds esperables) → false.
Si hay algo interesante (sorpresa en odds, partido de liga conocida) → true.
"""
    return _llamar_claude(MODELO_HAIKU, prompt, max_tokens=300)


def analisis_sonnet_partido(mercado: Mercado1X2) -> dict:
    """
    Sonnet hace el análisis profundo:
    - Estima probabilidad real de cada outcome
    - Identifica si hay value en alguno
    - Asigna confianza
    """
    prompt = f"""Sos un trader profesional de apuestas deportivas analizando value en un partido de fútbol.

PARTIDO: {mercado.titulo}
LIGA: {mercado.liga} ({mercado.pais})

ODDS OFRECIDAS:
- Local: {mercado.odds_home:.2f} (prob implícita: {mercado.prob_implicita_home():.1%})
- Empate: {mercado.odds_draw:.2f} (prob implícita: {mercado.prob_implicita_draw():.1%})
- Visitante: {mercado.odds_away:.2f} (prob implícita: {mercado.prob_implicita_away():.1%})
- Margen casa: {mercado.margen:.1%}

Tu tarea:
1. Estimá probabilidad REAL de cada resultado (home/draw/away) basándote en:
   - Forma reciente conocida de los equipos
   - Posición en la tabla de la liga
   - Historial head-to-head
   - Localía
   - Lesiones o ausencias conocidas
2. Compará prob real vs prob implícita → identificá si hay VALUE (real > implícita)
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
    1. Filtro odds extremas (muy bajas o muy altas)
    2. Haiku decide si vale el análisis profundo
    3. Sonnet estima probabilidades reales
    4. Construye SenalDeportiva si hay value
    """
    # 1. Filtro por odds razonables
    odds_validas = [
        mercado.odds_home,
        mercado.odds_draw,
        mercado.odds_away,
    ]
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

    # 3. Análisis Sonnet
    try:
        analisis = analisis_sonnet_partido(mercado)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"⚠️  Error Sonnet en {mercado.titulo}: {e}")
        return None

    seleccion_str = analisis.get("seleccion_con_value", "NINGUNA")
    if seleccion_str == "NINGUNA":
        return None

    # Mapear selección a datos concretos
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

    # 4. Validaciones finales
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
    Devuelve solo los que generaron señal con value.
    Respeta sesión horaria — si estamos en EVITAR, no consulta Claude.
    """
    sesion = get_sesion_actual()
    if not sesion.puede_operar:
        print(f"⏸️  Sesión {sesion.nombre} — análisis deportivo pausado")
        return []

    mercados = obtener_mercados_1x2_futbol()
    print(f"🔎 Analizando {len(mercados)} partidos 1X2 de fútbol...")

    senales = []
    for m in mercados:
        senal = analizar_partido(m)
        if senal:
            senales.append(senal)
            print(f"  ✅ Value detectado: {senal.titulo} → {senal.seleccion.value}")
        else:
            pass  # sin señal, seguimos

    return senales


# ─────────────────────────────────────────────────────────────
# Testing — python -m agent.analyst_sports
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("TEST: Analyst deportivo (Azuro 1X2 fútbol)")
    print("=" * 60)

    # Test 1: obtener mercados 1X2
    mercados = obtener_mercados_1x2_futbol()
    print(f"\n📊 Mercados 1X2 de fútbol encontrados: {len(mercados)}")

    if mercados:
        print(f"\n🔎 Primeros 3 mercados:")
        for m in mercados[:3]:
            print(f"  • {m.titulo} ({m.liga}, {m.pais})")
            print(f"    H: {m.odds_home:.2f} ({m.prob_implicita_home():.1%}) | "
                  f"D: {m.odds_draw:.2f} ({m.prob_implicita_draw():.1%}) | "
                  f"A: {m.odds_away:.2f} ({m.prob_implicita_away():.1%})")

    # Test 2: analizar solo 1 partido con Claude (para no gastar muchos créditos)
    if mercados:
        print(f"\n🤖 Analizando primer partido con Claude (Haiku + Sonnet si aplica)...")
        primer_mercado = mercados[0]
        senal = analizar_partido(primer_mercado)

        if senal:
            print(f"\n✅ SEÑAL DEPORTIVA:")
            print(f"   Partido: {senal.titulo}")
            print(f"   Selección: {senal.seleccion.value}")
            print(f"   Odds: {senal.odds:.2f}")
            print(f"   Prob implícita: {senal.probabilidad_implicita:.1%}")
            print(f"   Prob real (Claude): {senal.probabilidad_real:.1%}")
            print(f"   Value: {senal.value:+.1%}")
            print(f"   Confianza: {senal.confianza}")
            print(f"   Razón: {senal.razon}")
        else:
            print(f"   ❌ Sin value detectado en este partido (filtrado por Haiku o Sonnet)")

    print("\n✅ Test terminado")