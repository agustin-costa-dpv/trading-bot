"""
agent/analyst.py
Cerebro del bot. Combina:
1. Indicadores técnicos calculados en Python (RSI, EMAs, Bollinger)
2. Tres estrategias de trading evaluadas localmente
3. Filtrado con Claude Haiku (barato) para descartar ruido
4. Decisión final con Claude Sonnet (preciso) para señales fuertes

Devuelve un objeto Senal con probabilidad, dirección y confianza
que main.py va a pasar a risk.py para decidir si apostar.
"""

import os
import json
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
from anthropic import Anthropic

from bot.binance import get_snapshot_completo
from agent.sessions import get_sesion_actual, Prioridad


# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────

CLIENTE_CLAUDE = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODELO_HAIKU = "claude-haiku-4-5-20251001"      # Filtro rápido y barato
MODELO_SONNET = "claude-sonnet-4-6"             # Decisión final precisa

# Umbrales para disparar análisis con Claude
PROBABILIDAD_MINIMA_HAIKU = 0.55     # Por debajo de esto, descarta sin llamar a Sonnet
PROBABILIDAD_MINIMA_FINAL = 0.60     # Por debajo de esto, no se genera señal


# ─────────────────────────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────────────────────────

class Direccion(str, Enum):
    SUBE = "SUBE"
    BAJA = "BAJA"
    NEUTRAL = "NEUTRAL"


class Estrategia(str, Enum):
    MEAN_REVERSION = "MEAN_REVERSION"
    TREND_FOLLOWING = "TREND_FOLLOWING"
    ARBITRAJE_LATENCIA = "ARBITRAJE_LATENCIA"
    NINGUNA = "NINGUNA"


@dataclass
class Indicadores:
    """Indicadores técnicos calculados sobre las velas."""
    rsi: float                   # 0-100
    ema_9: float
    ema_21: float
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    precio_actual: float
    volumen_actual: float
    volumen_promedio: float      # promedio últimas 20 velas
    cambio_pct_5min: float       # movimiento últimos 5 min (para arbitraje)


@dataclass
class SenalEstrategia:
    """Señal generada por una estrategia individual (antes de Claude)."""
    estrategia: Estrategia
    direccion: Direccion
    fuerza: float                # 0-1, qué tan clara es la señal
    razon: str                   # explicación breve


@dataclass
class Senal:
    """Señal final que sale del analyst y va a risk.py."""
    activo: str
    estrategia: Estrategia
    direccion: Direccion
    probabilidad: float          # 0-1, estimación de Claude de que la apuesta gana
    confianza: str               # ALTA / MEDIA / BAJA
    razon: str                   # explicación
    indicadores: dict
    sesion: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────
# Cálculo de indicadores
# ─────────────────────────────────────────────────────────────

def _velas_a_df(snapshot: dict, key: str = "velas_15m") -> pd.DataFrame:
    """Convierte la lista de velas del snapshot en DataFrame de pandas."""
    df = pd.DataFrame(snapshot[key])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df


def calcular_indicadores(snapshot: dict) -> Indicadores:
    """
    Calcula RSI, EMAs, Bollinger Bands sobre las velas del snapshot.
    Usa velas de 15m para indicadores principales y 5m para detección de movimientos rápidos.
    """
    df_15m = _velas_a_df(snapshot, "velas_15m")
    df_5m = _velas_a_df(snapshot, "velas_5m")

    # RSI 14 períodos (estándar)
    rsi_serie = ta.rsi(df_15m["close"], length=14)
    rsi = float(rsi_serie.iloc[-1]) if not rsi_serie.empty else 50.0

    # EMAs 9 y 21
    ema_9 = float(ta.ema(df_15m["close"], length=9).iloc[-1])
    ema_21 = float(ta.ema(df_15m["close"], length=21).iloc[-1])

    # Bollinger Bands (20 períodos, 2 desv std)
    # pandas-ta devuelve columnas en orden: Lower, Middle, Upper, Bandwidth, Percent
    # Leemos por posición para ser inmunes a cambios de naming entre versiones
    bb = ta.bbands(df_15m["close"], length=20, std=2)
    bb_lower = float(bb.iloc[-1, 0])    # BBL
    bb_middle = float(bb.iloc[-1, 1])   # BBM
    bb_upper = float(bb.iloc[-1, 2])    # BBU

    # Volumen
    volumen_actual = float(df_15m["volume"].iloc[-1])
    volumen_promedio = float(df_15m["volume"].tail(20).mean())

    # Cambio % últimos 5 min (1 vela de 5m)
    if len(df_5m) >= 2:
        precio_5min_atras = float(df_5m["close"].iloc[-2])
        precio_actual = float(df_5m["close"].iloc[-1])
        cambio_pct_5min = ((precio_actual - precio_5min_atras) / precio_5min_atras) * 100
    else:
        cambio_pct_5min = 0.0

    return Indicadores(
        rsi=rsi,
        ema_9=ema_9,
        ema_21=ema_21,
        bollinger_upper=bb_upper,
        bollinger_middle=bb_middle,
        bollinger_lower=bb_lower,
        precio_actual=snapshot["precio"],
        volumen_actual=volumen_actual,
        volumen_promedio=volumen_promedio,
        cambio_pct_5min=cambio_pct_5min,
    )


# ─────────────────────────────────────────────────────────────
# Las 3 estrategias (lógica determinística en Python)
# ─────────────────────────────────────────────────────────────

def estrategia_mean_reversion(ind: Indicadores) -> SenalEstrategia:
    """
    Mean Reversion: opera reversiones en mercados laterales.
    Entrada LARGO: RSI < 30 + precio toca o cruza banda inferior Bollinger
    Entrada CORTO: RSI > 70 + precio toca o cruza banda superior Bollinger
    Win rate esperado: 68-71%
    """
    # Sobrevendido → SUBE
    if ind.rsi < 30 and ind.precio_actual <= ind.bollinger_lower * 1.001:
        fuerza = min(1.0, (30 - ind.rsi) / 30 + 0.5)
        return SenalEstrategia(
            estrategia=Estrategia.MEAN_REVERSION,
            direccion=Direccion.SUBE,
            fuerza=fuerza,
            razon=f"RSI sobrevendido ({ind.rsi:.1f}) + precio en banda inferior Bollinger",
        )

    # Sobrecomprado → BAJA
    if ind.rsi > 70 and ind.precio_actual >= ind.bollinger_upper * 0.999:
        fuerza = min(1.0, (ind.rsi - 70) / 30 + 0.5)
        return SenalEstrategia(
            estrategia=Estrategia.MEAN_REVERSION,
            direccion=Direccion.BAJA,
            fuerza=fuerza,
            razon=f"RSI sobrecomprado ({ind.rsi:.1f}) + precio en banda superior Bollinger",
        )

    return SenalEstrategia(
        estrategia=Estrategia.MEAN_REVERSION,
        direccion=Direccion.NEUTRAL,
        fuerza=0.0,
        razon="Sin condiciones de reversión",
    )


def estrategia_trend_following(ind: Indicadores) -> SenalEstrategia:
    """
    Trend Following: opera con la tendencia confirmada.
    Entrada LARGO: EMA 9 cruzó arriba de EMA 21 + volumen > promedio
    Entrada CORTO: EMA 9 cruzó abajo de EMA 21 + volumen > promedio
    Win rate esperado: 58-62%
    """
    volumen_alto = ind.volumen_actual > ind.volumen_promedio * 1.2
    distancia_emas_pct = abs(ind.ema_9 - ind.ema_21) / ind.ema_21 * 100

    # EMA 9 arriba de 21 con separación clara → tendencia alcista
    if ind.ema_9 > ind.ema_21 and distancia_emas_pct > 0.1 and volumen_alto:
        fuerza = min(1.0, distancia_emas_pct / 1.0)  # 1% de separación = fuerza máxima
        return SenalEstrategia(
            estrategia=Estrategia.TREND_FOLLOWING,
            direccion=Direccion.SUBE,
            fuerza=fuerza,
            razon=f"EMA9 ({ind.ema_9:.2f}) > EMA21 ({ind.ema_21:.2f}) + volumen alto",
        )

    # EMA 9 abajo de 21 con separación clara → tendencia bajista
    if ind.ema_9 < ind.ema_21 and distancia_emas_pct > 0.1 and volumen_alto:
        fuerza = min(1.0, distancia_emas_pct / 1.0)
        return SenalEstrategia(
            estrategia=Estrategia.TREND_FOLLOWING,
            direccion=Direccion.BAJA,
            fuerza=fuerza,
            razon=f"EMA9 ({ind.ema_9:.2f}) < EMA21 ({ind.ema_21:.2f}) + volumen alto",
        )

    return SenalEstrategia(
        estrategia=Estrategia.TREND_FOLLOWING,
        direccion=Direccion.NEUTRAL,
        fuerza=0.0,
        razon="Sin cruce claro de EMAs o volumen insuficiente",
    )


def estrategia_arbitraje_latencia(ind: Indicadores) -> SenalEstrategia:
    """
    Arbitraje de latencia: aprovecha que Azuro actualiza odds más lento que Binance.
    Si Binance se mueve fuerte (>1.5% en 5 min), Azuro tarda en reflejarlo.
    Win rate esperado: 75-85%
    """
    UMBRAL = 1.5  # % en 5 minutos

    if ind.cambio_pct_5min >= UMBRAL:
        fuerza = min(1.0, ind.cambio_pct_5min / 3.0)  # 3% = fuerza máxima
        return SenalEstrategia(
            estrategia=Estrategia.ARBITRAJE_LATENCIA,
            direccion=Direccion.SUBE,
            fuerza=fuerza,
            razon=f"Movimiento +{ind.cambio_pct_5min:.2f}% en 5min — Azuro lag",
        )

    if ind.cambio_pct_5min <= -UMBRAL:
        fuerza = min(1.0, abs(ind.cambio_pct_5min) / 3.0)
        return SenalEstrategia(
            estrategia=Estrategia.ARBITRAJE_LATENCIA,
            direccion=Direccion.BAJA,
            fuerza=fuerza,
            razon=f"Movimiento {ind.cambio_pct_5min:.2f}% en 5min — Azuro lag",
        )

    return SenalEstrategia(
        estrategia=Estrategia.ARBITRAJE_LATENCIA,
        direccion=Direccion.NEUTRAL,
        fuerza=0.0,
        razon=f"Movimiento solo {ind.cambio_pct_5min:+.2f}% (umbral {UMBRAL}%)",
    )


def evaluar_todas_estrategias(ind: Indicadores) -> SenalEstrategia:
    """
    Corre las 3 estrategias y devuelve la señal MÁS FUERTE (si hay alguna).
    Si ninguna dispara, devuelve NINGUNA con fuerza 0.
    """
    senales = [
        estrategia_arbitraje_latencia(ind),  # Prioridad 1: mejor win rate
        estrategia_mean_reversion(ind),       # Prioridad 2
        estrategia_trend_following(ind),      # Prioridad 3
    ]

    # Filtrar las que no son neutrales
    activas = [s for s in senales if s.direccion != Direccion.NEUTRAL]

    if not activas:
        return SenalEstrategia(
            estrategia=Estrategia.NINGUNA,
            direccion=Direccion.NEUTRAL,
            fuerza=0.0,
            razon="Ninguna estrategia disparó señal",
        )

    # Devolver la de mayor fuerza
    return max(activas, key=lambda s: s.fuerza)


# ─────────────────────────────────────────────────────────────
# Filtrado con Claude
# ─────────────────────────────────────────────────────────────

def _llamar_claude(modelo: str, prompt: str, max_tokens: int = 500) -> dict:
    """Llama a Claude y parsea la respuesta JSON."""
    response = CLIENTE_CLAUDE.messages.create(
        model=modelo,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    texto = response.content[0].text.strip()
    # Limpiar markdown si Claude lo agrega
    if texto.startswith("```"):
        texto = texto.split("```")[1]
        if texto.startswith("json"):
            texto = texto[4:]
        texto = texto.strip()
    return json.loads(texto)


def filtro_haiku(activo: str, senal: SenalEstrategia, ind: Indicadores) -> dict:
    """
    Haiku evalúa rápido si la señal es genuina o ruido.
    Devuelve {aprobada: bool, probabilidad: float, razon: str}
    """
    prompt = f"""Sos un analista de trading cripto. Evaluá si esta señal es genuina o ruido.

ACTIVO: {activo}
ESTRATEGIA: {senal.estrategia.value}
DIRECCIÓN: {senal.direccion.value}
RAZÓN TÉCNICA: {senal.razon}
FUERZA DETECTADA: {senal.fuerza:.2f}

INDICADORES:
- RSI: {ind.rsi:.1f}
- EMA 9: {ind.ema_9:.2f}
- EMA 21: {ind.ema_21:.2f}
- Bollinger: {ind.bollinger_lower:.2f} / {ind.bollinger_middle:.2f} / {ind.bollinger_upper:.2f}
- Precio actual: {ind.precio_actual:.2f}
- Volumen actual vs promedio: {ind.volumen_actual:.2f} vs {ind.volumen_promedio:.2f}
- Cambio últimos 5 min: {ind.cambio_pct_5min:+.2f}%

Respondé SOLO con un JSON válido (sin markdown, sin explicación previa):
{{"aprobada": true/false, "probabilidad": 0.0-1.0, "razon": "explicación breve en una frase"}}
"""
    return _llamar_claude(MODELO_HAIKU, prompt, max_tokens=300)


def decision_sonnet(activo: str, senal: SenalEstrategia, ind: Indicadores,
                     filtro: dict, sesion_nombre: str) -> dict:
    """
    Sonnet hace el análisis profundo y da la decisión final.
    Devuelve {probabilidad: float, confianza: str, razon: str}
    """
    prompt = f"""Sos un trader cripto profesional analizando una operación en Azuro Protocol.

CONTEXTO:
- Activo: {activo}
- Sesión actual: {sesion_nombre}
- Estrategia detectada: {senal.estrategia.value}
- Dirección propuesta: {senal.direccion.value}
- Razón técnica: {senal.razon}

INDICADORES TÉCNICOS:
- RSI(14): {ind.rsi:.1f}
- EMA 9: {ind.ema_9:.2f} | EMA 21: {ind.ema_21:.2f}
- Bollinger Bands: lower={ind.bollinger_lower:.2f}, mid={ind.bollinger_middle:.2f}, upper={ind.bollinger_upper:.2f}
- Precio actual: {ind.precio_actual:.2f}
- Volumen actual: {ind.volumen_actual:.2f} (promedio 20: {ind.volumen_promedio:.2f})
- Cambio 5min: {ind.cambio_pct_5min:+.2f}%

ANÁLISIS PRELIMINAR (Haiku):
- Probabilidad: {filtro.get('probabilidad', 0):.2f}
- Razón: {filtro.get('razon', 'N/A')}

Tu tarea: dar la decisión final. Considerá:
1. ¿Los indicadores realmente confirman la dirección propuesta?
2. ¿La sesión horaria es favorable para esta estrategia?
3. ¿Hay riesgo de reversión inminente?

Respondé SOLO con un JSON válido (sin markdown):
{{"probabilidad": 0.0-1.0, "confianza": "ALTA"|"MEDIA"|"BAJA", "razon": "explicación de 1-2 frases"}}
"""
    return _llamar_claude(MODELO_SONNET, prompt, max_tokens=400)


# ─────────────────────────────────────────────────────────────
# Función principal — orquestación
# ─────────────────────────────────────────────────────────────

def analizar(activo: str) -> Optional[Senal]:
    """
    Pipeline completo de análisis para un activo.
    Devuelve Senal si hay oportunidad clara, None si no.

    Flujo:
    1. Snapshot de Binance
    2. Cálculo de indicadores
    3. Evaluación de las 3 estrategias
    4. Si hay señal → Haiku filtra
    5. Si Haiku aprueba → Sonnet decide
    6. Si probabilidad final >= umbral → devolver Senal
    """
    sesion = get_sesion_actual()

    # Si la sesión bloquea operación, ni siquiera consultamos
    if not sesion.puede_operar:
        return None

    # 1. Datos de mercado
    snapshot = get_snapshot_completo(activo)

    # 2. Indicadores
    ind = calcular_indicadores(snapshot)

    # 3. Estrategias
    senal_estrategia = evaluar_todas_estrategias(ind)
    if senal_estrategia.estrategia == Estrategia.NINGUNA:
        return None

    # 4. Filtro Haiku
    try:
        filtro = filtro_haiku(activo, senal_estrategia, ind)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"⚠️  Error en filtro Haiku para {activo}: {e}")
        return None

    if not filtro.get("aprobada", False) or filtro.get("probabilidad", 0) < PROBABILIDAD_MINIMA_HAIKU:
        return None

    # 5. Decisión Sonnet
    try:
        decision = decision_sonnet(activo, senal_estrategia, ind, filtro, sesion.nombre)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"⚠️  Error en decisión Sonnet para {activo}: {e}")
        return None

    probabilidad_final = decision.get("probabilidad", 0)
    if probabilidad_final < PROBABILIDAD_MINIMA_FINAL:
        return None

    # 6. Construir señal final
    return Senal(
        activo=activo,
        estrategia=senal_estrategia.estrategia,
        direccion=senal_estrategia.direccion,
        probabilidad=probabilidad_final,
        confianza=decision.get("confianza", "MEDIA"),
        razon=decision.get("razon", senal_estrategia.razon),
        indicadores={
            "rsi": ind.rsi,
            "ema_9": ind.ema_9,
            "ema_21": ind.ema_21,
            "bb_upper": ind.bollinger_upper,
            "bb_lower": ind.bollinger_lower,
            "precio": ind.precio_actual,
            "cambio_5min": ind.cambio_pct_5min,
        },
        sesion=sesion.nombre,
        timestamp=datetime.now().isoformat(),
    )


# ─────────────────────────────────────────────────────────────
# Testing — correr con: python -m agent.analyst
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Pipeline completo de análisis")
    print("=" * 60)

    for activo in ["BTC", "ETH", "SOL"]:
        print(f"\n🔍 Analizando {activo}...")
        try:
            snapshot = get_snapshot_completo(activo)
            ind = calcular_indicadores(snapshot)
            print(f"  RSI: {ind.rsi:.1f} | EMA9/21: {ind.ema_9:.2f}/{ind.ema_21:.2f}")
            print(f"  Bollinger: {ind.bollinger_lower:.2f} | {ind.bollinger_middle:.2f} | {ind.bollinger_upper:.2f}")
            print(f"  Cambio 5min: {ind.cambio_pct_5min:+.2f}%")

            senal_est = evaluar_todas_estrategias(ind)
            print(f"  Estrategia: {senal_est.estrategia.value} → {senal_est.direccion.value} (fuerza {senal_est.fuerza:.2f})")
            print(f"  Razón: {senal_est.razon}")

            # Análisis completo (con Claude) solo si hay señal
            if senal_est.estrategia != Estrategia.NINGUNA:
                print(f"  🤖 Consultando Claude...")
                senal = analizar(activo)
                if senal:
                    print(f"  ✅ SEÑAL FINAL: {senal.direccion.value} | prob {senal.probabilidad:.2f} | {senal.confianza}")
                    print(f"     Razón: {senal.razon}")
                else:
                    print(f"  ❌ Claude descartó la señal")
            else:
                print(f"  ⏭️  Sin señal técnica, no se consulta Claude (ahorro de costo)")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n✅ Test terminado")