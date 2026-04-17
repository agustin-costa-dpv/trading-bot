"""
agent/analyst.py — v3 (validado por backtest de 6 meses)

Configuración validada:
- BTC: solo TREND_FOLLOWING
- ETH: TREND_FOLLOWING + ARBITRAJE
- SOL: desactivado (pierde consistentemente)
- MEAN_REVERSION: desactivada globalmente (PF 0.84 en 6 meses)

Métricas validadas (backtest 6m):
- Sharpe 1.03 | PF 1.19 | Max DD -13.8% | PnL +76.6%

Parámetros por estrategia (TP/SL/horizonte específicos):
- TREND_FOLLOWING: TP 2.0% / SL 1.0% / 6 velas 15m
- ARBITRAJE:       TP 1.5% / SL 0.7% / 2 velas 15m
"""

import os
import json
import logging
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
from anthropic import Anthropic

from bot.binance import get_snapshot_completo
from agent.sessions import get_sesion_actual


# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────

CLIENTE_CLAUDE = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

logger = logging.getLogger("agent.analyst")

MODELO_HAIKU = "claude-haiku-4-5-20251001"
MODELO_SONNET = "claude-sonnet-4-6"

PROBABILIDAD_MINIMA_FINAL = float(os.getenv("PROBABILIDAD_MIN_EJECUCION", 0.58))

# Régimen (validado por backtest)
ADX_TENDENCIA = 28
ADX_LATERAL = 18
ARBITRAJE_ATR_MULT = 0.8

# Funding (contrarian)
FUNDING_EXTREMO = 0.01

# Matriz de estrategias habilitadas por activo (decisión basada en datos)
ESTRATEGIAS_HABILITADAS = {
    "BTC": {"TREND_FOLLOWING"},
    "ETH": {"TREND_FOLLOWING", "ARBITRAJE"},
    # SOL: ninguna (descartado por backtest)
}

# Parámetros TP/SL/horizonte por estrategia (en %, velas de 15m)
PARAMS_ESTRATEGIA = {
    "TREND_FOLLOWING": {"tp": 2.0, "sl": 1.0, "horizonte_min": 90},   # 6 velas = 90min
    "ARBITRAJE":       {"tp": 1.5, "sl": 0.7, "horizonte_min": 30},   # 2 velas = 30min
}


# ─────────────────────────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────────────────────────

class Direccion(str, Enum):
    SUBE = "SUBE"
    BAJA = "BAJA"
    NEUTRAL = "NEUTRAL"


class Regimen(str, Enum):
    LATERAL = "LATERAL"
    TENDENCIA = "TENDENCIA"
    MOVIMIENTO_BRUSCO = "MOVIMIENTO_BRUSCO"
    INDEFINIDO = "INDEFINIDO"


class Estrategia(str, Enum):
    TREND_FOLLOWING = "TREND_FOLLOWING"
    ARBITRAJE_LATENCIA = "ARBITRAJE_LATENCIA"
    NINGUNA = "NINGUNA"


@dataclass
class Indicadores:
    rsi: float
    ema_9: float
    ema_21: float
    ema_50: float
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    bb_width_pct: float
    adx: float
    atr_pct: float
    precio_actual: float
    volumen_actual: float
    volumen_promedio: float
    cambio_pct_5min: float
    cambio_pct_15min: float
    funding_rate: float
    long_short_ratio: float


@dataclass
class SenalEstrategia:
    estrategia: Estrategia
    direccion: Direccion
    fuerza: float
    razon: str


@dataclass
class Senal:
    activo: str
    estrategia: Estrategia
    direccion: Direccion
    probabilidad: float
    confianza: str
    razon: str
    regimen: str
    tp_pct: float
    sl_pct: float
    horizonte_min: int
    indicadores: dict
    sesion: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────
# Indicadores
# ─────────────────────────────────────────────────────────────

def _velas_a_df(snapshot: dict, key: str = "velas_15m") -> pd.DataFrame:
    df = pd.DataFrame(snapshot[key])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")


def calcular_indicadores(snapshot: dict) -> Indicadores:
    df_15m = _velas_a_df(snapshot, "velas_15m")
    df_5m = _velas_a_df(snapshot, "velas_5m")

    rsi = float(ta.rsi(df_15m["close"], length=14).iloc[-1])
    ema_9 = float(ta.ema(df_15m["close"], length=9).iloc[-1])
    ema_21 = float(ta.ema(df_15m["close"], length=21).iloc[-1])
    ema_50 = float(ta.ema(df_15m["close"], length=50).iloc[-1]) if len(df_15m) >= 50 else ema_21

    bb = ta.bbands(df_15m["close"], length=20, std=2)
    bb_lower = float(bb.iloc[-1, 0])
    bb_middle = float(bb.iloc[-1, 1])
    bb_upper = float(bb.iloc[-1, 2])
    bb_width_pct = ((bb_upper - bb_lower) / bb_middle) * 100

    adx_df = ta.adx(df_15m["high"], df_15m["low"], df_15m["close"], length=14)
    adx = float(adx_df.iloc[-1, 0]) if adx_df is not None and not adx_df.empty else 20.0

    atr = float(ta.atr(df_15m["high"], df_15m["low"], df_15m["close"], length=14).iloc[-1])
    precio_actual = snapshot["precio"]
    atr_pct = (atr / precio_actual) * 100

    volumen_actual = float(df_15m["volume"].iloc[-1])
    volumen_promedio = float(df_15m["volume"].tail(20).mean())

    cambio_5min = 0.0
    if len(df_5m) >= 2:
        cambio_5min = ((df_5m["close"].iloc[-1] - df_5m["close"].iloc[-2]) / df_5m["close"].iloc[-2]) * 100
    cambio_15min = 0.0
    if len(df_15m) >= 2:
        cambio_15min = ((df_15m["close"].iloc[-1] - df_15m["close"].iloc[-2]) / df_15m["close"].iloc[-2]) * 100

    funding_rate = float(snapshot.get("funding_rate", 0.0))
    long_short_ratio = float(snapshot.get("long_short_ratio", 1.0))

    return Indicadores(
        rsi=rsi, ema_9=ema_9, ema_21=ema_21, ema_50=ema_50,
        bollinger_upper=bb_upper, bollinger_middle=bb_middle, bollinger_lower=bb_lower,
        bb_width_pct=bb_width_pct, adx=adx, atr_pct=atr_pct,
        precio_actual=precio_actual,
        volumen_actual=volumen_actual, volumen_promedio=volumen_promedio,
        cambio_pct_5min=cambio_5min, cambio_pct_15min=cambio_15min,
        funding_rate=funding_rate, long_short_ratio=long_short_ratio,
    )


# ─────────────────────────────────────────────────────────────
# Detector de régimen
# ─────────────────────────────────────────────────────────────

def detectar_regimen(ind: Indicadores) -> Regimen:
    if abs(ind.cambio_pct_5min) >= max(0.7, ind.atr_pct * ARBITRAJE_ATR_MULT):
        return Regimen.MOVIMIENTO_BRUSCO
    if ind.adx >= ADX_TENDENCIA:
        return Regimen.TENDENCIA
    if ind.adx <= ADX_LATERAL:
        return Regimen.LATERAL
    return Regimen.INDEFINIDO


# ─────────────────────────────────────────────────────────────
# Estrategias
# ─────────────────────────────────────────────────────────────

def estrategia_trend_following(ind: Indicadores) -> SenalEstrategia:
    dist_emas_pct = abs(ind.ema_9 - ind.ema_21) / ind.ema_21 * 100
    volumen_ok = ind.volumen_actual >= ind.volumen_promedio * 0.9

    if ind.ema_9 > ind.ema_21 > ind.ema_50 and dist_emas_pct > 0.05 and volumen_ok:
        fuerza = min(1.0, dist_emas_pct / 0.8 + ind.adx / 100)
        return SenalEstrategia(
            Estrategia.TREND_FOLLOWING, Direccion.SUBE, fuerza,
            f"EMAs alineadas alcistas + ADX {ind.adx:.1f}",
        )

    if ind.ema_9 < ind.ema_21 < ind.ema_50 and dist_emas_pct > 0.05 and volumen_ok:
        fuerza = min(1.0, dist_emas_pct / 0.8 + ind.adx / 100)
        return SenalEstrategia(
            Estrategia.TREND_FOLLOWING, Direccion.BAJA, fuerza,
            f"EMAs alineadas bajistas + ADX {ind.adx:.1f}",
        )

    return SenalEstrategia(Estrategia.TREND_FOLLOWING, Direccion.NEUTRAL, 0.0, "Sin tendencia alineada")


def estrategia_arbitraje_latencia(ind: Indicadores) -> SenalEstrategia:
    umbral = max(0.7, ind.atr_pct * ARBITRAJE_ATR_MULT)

    if ind.cambio_pct_5min >= umbral:
        fuerza = min(1.0, ind.cambio_pct_5min / (umbral * 2))
        return SenalEstrategia(
            Estrategia.ARBITRAJE_LATENCIA, Direccion.SUBE, fuerza,
            f"+{ind.cambio_pct_5min:.2f}% en 5min (umbral {umbral:.2f}%) — Azuro lag",
        )

    if ind.cambio_pct_5min <= -umbral:
        fuerza = min(1.0, abs(ind.cambio_pct_5min) / (umbral * 2))
        return SenalEstrategia(
            Estrategia.ARBITRAJE_LATENCIA, Direccion.BAJA, fuerza,
            f"{ind.cambio_pct_5min:.2f}% en 5min (umbral {umbral:.2f}%) — Azuro lag",
        )

    return SenalEstrategia(Estrategia.ARBITRAJE_LATENCIA, Direccion.NEUTRAL, 0.0, f"Mov {ind.cambio_pct_5min:+.2f}%")


def evaluar_por_regimen_y_activo(regimen: Regimen, ind: Indicadores, activo: str) -> SenalEstrategia:
    """
    Solo activa estrategia si:
    1. El activo tiene al menos una estrategia habilitada
    2. La estrategia del régimen está habilitada para ese activo
    """
    estrategias_ok = ESTRATEGIAS_HABILITADAS.get(activo.upper(), set())
    if not estrategias_ok:
        return SenalEstrategia(Estrategia.NINGUNA, Direccion.NEUTRAL, 0.0,
                                f"{activo} desactivado por backtest")

    if regimen == Regimen.MOVIMIENTO_BRUSCO:
        if "ARBITRAJE" not in estrategias_ok:
            return SenalEstrategia(Estrategia.NINGUNA, Direccion.NEUTRAL, 0.0,
                                    f"ARBITRAJE no habilitado en {activo}")
        return estrategia_arbitraje_latencia(ind)

    if regimen == Regimen.TENDENCIA:
        if "TREND_FOLLOWING" not in estrategias_ok:
            return SenalEstrategia(Estrategia.NINGUNA, Direccion.NEUTRAL, 0.0,
                                    f"TREND_FOLLOWING no habilitado en {activo}")
        return estrategia_trend_following(ind)

    # LATERAL o INDEFINIDO: MEAN_REVERSION desactivada globalmente
    return SenalEstrategia(Estrategia.NINGUNA, Direccion.NEUTRAL, 0.0,
                            f"Régimen {regimen.value} sin estrategia")


# ─────────────────────────────────────────────────────────────
# Claude
# ─────────────────────────────────────────────────────────────

def _llamar_claude(modelo: str, prompt: str, max_tokens: int = 400) -> dict:
    response = CLIENTE_CLAUDE.messages.create(
        model=modelo, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    texto = response.content[0].text.strip()
    if texto.startswith("```"):
        texto = texto.split("```")[1]
        if texto.startswith("json"):
            texto = texto[4:]
        texto = texto.strip()
    return json.loads(texto)


def validar_con_claude(activo: str, senal: SenalEstrategia, ind: Indicadores,
                       regimen: Regimen, sesion_nombre: str) -> dict:
    modelo = MODELO_SONNET if senal.fuerza > 0.8 else MODELO_HAIKU

    prompt = f"""Trader cripto profesional. Validar operación en Azuro.

ACTIVO: {activo} | SESIÓN: {sesion_nombre} | RÉGIMEN: {regimen.value}
ESTRATEGIA: {senal.estrategia.value} → {senal.direccion.value} (fuerza {senal.fuerza:.2f})
RAZÓN: {senal.razon}

INDICADORES:
RSI {ind.rsi:.1f} | ADX {ind.adx:.1f} | ATR% {ind.atr_pct:.2f}
EMA9/21/50: {ind.ema_9:.2f}/{ind.ema_21:.2f}/{ind.ema_50:.2f}
BB: {ind.bollinger_lower:.2f}/{ind.bollinger_middle:.2f}/{ind.bollinger_upper:.2f}
Precio: {ind.precio_actual:.2f} | Vol: {ind.volumen_actual:.0f} vs {ind.volumen_promedio:.0f}
Cambio 5m/15m: {ind.cambio_pct_5min:+.2f}% / {ind.cambio_pct_15min:+.2f}%
Funding: {ind.funding_rate:+.4f} | L/S: {ind.long_short_ratio:.2f}

¿Confirmás la dirección? Considerá coherencia técnica, sentiment, sesión, riesgo de reversión.

SOLO JSON:
{{"probabilidad": 0.0-1.0, "confianza": "ALTA"|"MEDIA"|"BAJA", "razon": "1 frase"}}
"""
    return _llamar_claude(modelo, prompt, max_tokens=300)


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────

def analizar(activo: str) -> Optional[Senal]:
    # Early exit: si el activo no tiene estrategias habilitadas, cero consumo
    if not ESTRATEGIAS_HABILITADAS.get(activo.upper()):
        return None

    sesion = get_sesion_actual()
    if not sesion.puede_operar:
        logger.info(f"{activo}: sesión {sesion.nombre} no opera")
        return None

    snapshot = get_snapshot_completo(activo)
    ind = calcular_indicadores(snapshot)

    regimen = detectar_regimen(ind)
    logger.info(
        f"{activo}: ADX={ind.adx:.1f} | RSI={ind.rsi:.1f} | ATR%={ind.atr_pct:.2f} | "
        f"cambio_5m={ind.cambio_pct_5min:+.2f}% | régimen={regimen.value}"
    )
    if regimen in (Regimen.INDEFINIDO, Regimen.LATERAL):
        logger.info(f"{activo}: descartado por régimen {regimen.value}")
        return None

    senal_est = evaluar_por_regimen_y_activo(regimen, ind, activo)
    if senal_est.estrategia == Estrategia.NINGUNA or senal_est.direccion == Direccion.NEUTRAL:
        logger.info(f"{activo}: sin señal técnica — {senal_est.razon}")
        return None

    try:
        decision = validar_con_claude(activo, senal_est, ind, regimen, sesion.nombre)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning(f"{activo}: error Claude — {e}")
        return None

    prob = decision.get("probabilidad", 0)
    logger.info(f"{activo}: Claude prob={prob:.2f} (umbral={PROBABILIDAD_MINIMA_FINAL})")
    if prob < PROBABILIDAD_MINIMA_FINAL:
        return None

    # Parámetros TP/SL/horizonte según estrategia
    estrategia_key = "TREND_FOLLOWING" if senal_est.estrategia == Estrategia.TREND_FOLLOWING else "ARBITRAJE"
    params = PARAMS_ESTRATEGIA[estrategia_key]

    return Senal(
        activo=activo,
        estrategia=senal_est.estrategia,
        direccion=senal_est.direccion,
        probabilidad=prob,
        confianza=decision.get("confianza", "MEDIA"),
        razon=decision.get("razon", senal_est.razon),
        regimen=regimen.value,
        tp_pct=params["tp"],
        sl_pct=params["sl"],
        horizonte_min=params["horizonte_min"],
        indicadores={
            "rsi": ind.rsi, "adx": ind.adx, "atr_pct": ind.atr_pct,
            "ema_9": ind.ema_9, "ema_21": ind.ema_21, "ema_50": ind.ema_50,
            "bb_upper": ind.bollinger_upper, "bb_lower": ind.bollinger_lower,
            "precio": ind.precio_actual, "cambio_5min": ind.cambio_pct_5min,
            "funding": ind.funding_rate, "ls_ratio": ind.long_short_ratio,
        },
        sesion=sesion.nombre,
        timestamp=datetime.now().isoformat(),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("TEST analyst.py v3 (configuración validada por backtest)")
    print("=" * 60)
    print(f"\nEstrategias habilitadas:")
    for a, e in ESTRATEGIAS_HABILITADAS.items():
        print(f"  {a}: {', '.join(e)}")
    print(f"  SOL: desactivado\n")

    for activo in ["BTC", "ETH", "SOL"]:
        print(f"🔍 {activo}")
        try:
            if not ESTRATEGIAS_HABILITADAS.get(activo):
                print(f"  ⏭️  Desactivado (sin consumo)\n")
                continue
            snap = get_snapshot_completo(activo)
            ind = calcular_indicadores(snap)
            reg = detectar_regimen(ind)
            print(f"  RSI {ind.rsi:.1f} | ADX {ind.adx:.1f} | ATR% {ind.atr_pct:.2f}")
            print(f"  Régimen: {reg.value}")
            senal_est = evaluar_por_regimen_y_activo(reg, ind, activo)
            print(f"  Estrategia: {senal_est.estrategia.value} → {senal_est.direccion.value} ({senal_est.fuerza:.2f})")
            if senal_est.direccion != Direccion.NEUTRAL:
                print(f"  🤖 Claude...")
                s = analizar(activo)
                if s:
                    print(f"  ✅ {s.direccion.value} prob={s.probabilidad:.2f} [{s.confianza}]")
                    print(f"     TP {s.tp_pct}% / SL {s.sl_pct}% / {s.horizonte_min}min")
                else:
                    print(f"  ❌ Claude descartó o prob < {PROBABILIDAD_MINIMA_FINAL}")
            else:
                print(f"  ⏭️  Sin señal (ahorro)")
            print()
        except Exception as e:
            print(f"  ❌ {e}\n")