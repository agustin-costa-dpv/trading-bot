"""
agent/analyst.py — v2 (régimen adaptativo)

Cambios clave vs v1:
1. Detector de régimen con ADX: activa solo la estrategia apropiada
2. Umbrales relajados con base estadística (percentiles + ATR)
3. Funding rate como filtro contrarian (edge institucional)
4. Un solo umbral Claude (no doble embudo)
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
from agent.sessions import get_sesion_actual


# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────

CLIENTE_CLAUDE = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODELO_HAIKU = "claude-haiku-4-5-20251001"
MODELO_SONNET = "claude-sonnet-4-6"

# Un solo umbral final — más simple y genera más datos
PROBABILIDAD_MINIMA_FINAL = 0.58

# Régimen
ADX_TENDENCIA = 22       # > 22 = tendencia clara
ADX_LATERAL = 18         # < 18 = lateral claro
# Entre 18-22 = zona gris, no opera

# Funding rate (contrarian signal)
FUNDING_EXTREMO = 0.01   # >1% anualizado = sentiment muy cargado


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
    MEAN_REVERSION = "MEAN_REVERSION"
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
    bb_width_pct: float          # ancho de bandas normalizado
    adx: float                   # fuerza de tendencia
    atr_pct: float               # volatilidad normalizada
    precio_actual: float
    volumen_actual: float
    volumen_promedio: float
    cambio_pct_5min: float
    cambio_pct_15min: float
    funding_rate: float          # de Binance Futures
    long_short_ratio: float      # sentiment institucional


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
    indicadores: dict
    sesion: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────
# Cálculo de indicadores
# ─────────────────────────────────────────────────────────────

def _velas_a_df(snapshot: dict, key: str = "velas_15m") -> pd.DataFrame:
    df = pd.DataFrame(snapshot[key])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")


def calcular_indicadores(snapshot: dict) -> Indicadores:
    df_15m = _velas_a_df(snapshot, "velas_15m")
    df_5m = _velas_a_df(snapshot, "velas_5m")

    # RSI
    rsi = float(ta.rsi(df_15m["close"], length=14).iloc[-1])

    # EMAs
    ema_9 = float(ta.ema(df_15m["close"], length=9).iloc[-1])
    ema_21 = float(ta.ema(df_15m["close"], length=21).iloc[-1])
    ema_50 = float(ta.ema(df_15m["close"], length=50).iloc[-1]) if len(df_15m) >= 50 else ema_21

    # Bollinger
    bb = ta.bbands(df_15m["close"], length=20, std=2)
    bb_lower = float(bb.iloc[-1, 0])
    bb_middle = float(bb.iloc[-1, 1])
    bb_upper = float(bb.iloc[-1, 2])
    bb_width_pct = ((bb_upper - bb_lower) / bb_middle) * 100

    # ADX (fuerza de tendencia) — CLAVE para detectar régimen
    adx_df = ta.adx(df_15m["high"], df_15m["low"], df_15m["close"], length=14)
    adx = float(adx_df.iloc[-1, 0]) if adx_df is not None and not adx_df.empty else 20.0

    # ATR % (volatilidad normalizada)
    atr = float(ta.atr(df_15m["high"], df_15m["low"], df_15m["close"], length=14).iloc[-1])
    precio_actual = snapshot["precio"]
    atr_pct = (atr / precio_actual) * 100

    # Volumen
    volumen_actual = float(df_15m["volume"].iloc[-1])
    volumen_promedio = float(df_15m["volume"].tail(20).mean())

    # Cambios de precio
    cambio_5min = 0.0
    if len(df_5m) >= 2:
        cambio_5min = ((df_5m["close"].iloc[-1] - df_5m["close"].iloc[-2]) / df_5m["close"].iloc[-2]) * 100
    cambio_15min = 0.0
    if len(df_15m) >= 2:
        cambio_15min = ((df_15m["close"].iloc[-1] - df_15m["close"].iloc[-2]) / df_15m["close"].iloc[-2]) * 100

    # Datos de derivados (si el snapshot los trae, si no, defaults neutros)
    funding_rate = float(snapshot.get("funding_rate", 0.0))
    long_short_ratio = float(snapshot.get("long_short_ratio", 1.0))

    return Indicadores(
        rsi=rsi,
        ema_9=ema_9,
        ema_21=ema_21,
        ema_50=ema_50,
        bollinger_upper=bb_upper,
        bollinger_middle=bb_middle,
        bollinger_lower=bb_lower,
        bb_width_pct=bb_width_pct,
        adx=adx,
        atr_pct=atr_pct,
        precio_actual=precio_actual,
        volumen_actual=volumen_actual,
        volumen_promedio=volumen_promedio,
        cambio_pct_5min=cambio_5min,
        cambio_pct_15min=cambio_15min,
        funding_rate=funding_rate,
        long_short_ratio=long_short_ratio,
    )


# ─────────────────────────────────────────────────────────────
# Detector de régimen
# ─────────────────────────────────────────────────────────────

def detectar_regimen(ind: Indicadores) -> Regimen:
    """
    Detecta el régimen de mercado actual para activar la estrategia correcta.
    - Movimiento brusco (>1.2 ATR en 5min) → arbitraje
    - ADX > 22 → tendencia
    - ADX < 18 → lateral
    - Entre medias → indefinido, no opera
    """
    # Movimiento brusco tiene prioridad (ventana de arbitraje)
    if abs(ind.cambio_pct_5min) >= max(1.2, ind.atr_pct * 1.2):
        return Regimen.MOVIMIENTO_BRUSCO

    if ind.adx >= ADX_TENDENCIA:
        return Regimen.TENDENCIA

    if ind.adx <= ADX_LATERAL:
        return Regimen.LATERAL

    return Regimen.INDEFINIDO


# ─────────────────────────────────────────────────────────────
# Estrategias (una por régimen, umbrales relajados)
# ─────────────────────────────────────────────────────────────

def estrategia_mean_reversion(ind: Indicadores) -> SenalEstrategia:
    """
    Activa SOLO en régimen lateral.
    Relajado: RSI<35 + distancia a banda <0.5%, o RSI>65 + distancia <0.5%.
    Funding rate refuerza (contrarian).
    """
    dist_lower_pct = ((ind.precio_actual - ind.bollinger_lower) / ind.precio_actual) * 100
    dist_upper_pct = ((ind.bollinger_upper - ind.precio_actual) / ind.precio_actual) * 100

    # Sobrevendido → SUBE
    if ind.rsi < 35 and dist_lower_pct < 0.5:
        fuerza = (35 - ind.rsi) / 35 * 0.6 + (0.5 - dist_lower_pct) * 0.4
        # Funding negativo extremo = todos short = refuerzo alcista contrarian
        if ind.funding_rate < -FUNDING_EXTREMO:
            fuerza = min(1.0, fuerza + 0.2)
        return SenalEstrategia(
            Estrategia.MEAN_REVERSION, Direccion.SUBE, min(1.0, fuerza),
            f"RSI {ind.rsi:.1f} + banda inf ({dist_lower_pct:.2f}%) | funding {ind.funding_rate:+.4f}",
        )

    # Sobrecomprado → BAJA
    if ind.rsi > 65 and dist_upper_pct < 0.5:
        fuerza = (ind.rsi - 65) / 35 * 0.6 + (0.5 - dist_upper_pct) * 0.4
        if ind.funding_rate > FUNDING_EXTREMO:
            fuerza = min(1.0, fuerza + 0.2)
        return SenalEstrategia(
            Estrategia.MEAN_REVERSION, Direccion.BAJA, min(1.0, fuerza),
            f"RSI {ind.rsi:.1f} + banda sup ({dist_upper_pct:.2f}%) | funding {ind.funding_rate:+.4f}",
        )

    return SenalEstrategia(Estrategia.MEAN_REVERSION, Direccion.NEUTRAL, 0.0, "Sin reversión")


def estrategia_trend_following(ind: Indicadores) -> SenalEstrategia:
    """
    Activa SOLO en régimen de tendencia (ADX>22).
    Relajado: volumen normal (no exige 1.2x), confirmación con EMA50.
    """
    dist_emas_pct = abs(ind.ema_9 - ind.ema_21) / ind.ema_21 * 100
    volumen_ok = ind.volumen_actual >= ind.volumen_promedio * 0.9  # relajado

    # Alcista: EMA9>21>50 (alineación clásica)
    if ind.ema_9 > ind.ema_21 > ind.ema_50 and dist_emas_pct > 0.05 and volumen_ok:
        fuerza = min(1.0, dist_emas_pct / 0.8 + ind.adx / 100)
        return SenalEstrategia(
            Estrategia.TREND_FOLLOWING, Direccion.SUBE, fuerza,
            f"EMAs alineadas alcistas + ADX {ind.adx:.1f}",
        )

    # Bajista: EMA9<21<50
    if ind.ema_9 < ind.ema_21 < ind.ema_50 and dist_emas_pct > 0.05 and volumen_ok:
        fuerza = min(1.0, dist_emas_pct / 0.8 + ind.adx / 100)
        return SenalEstrategia(
            Estrategia.TREND_FOLLOWING, Direccion.BAJA, fuerza,
            f"EMAs alineadas bajistas + ADX {ind.adx:.1f}",
        )

    return SenalEstrategia(Estrategia.TREND_FOLLOWING, Direccion.NEUTRAL, 0.0, "Sin tendencia alineada")


def estrategia_arbitraje_latencia(ind: Indicadores) -> SenalEstrategia:
    """
    Activa cuando hay movimiento brusco en 5min.
    Umbral adaptativo según ATR (no fijo en 1.5%).
    """
    umbral = max(1.0, ind.atr_pct * 1.2)

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


def evaluar_por_regimen(regimen: Regimen, ind: Indicadores) -> SenalEstrategia:
    """
    Activa SOLO la estrategia apropiada al régimen (no todas juntas).
    Esto es el cambio institucional clave: régimen → estrategia → señal.
    """
    if regimen == Regimen.MOVIMIENTO_BRUSCO:
        return estrategia_arbitraje_latencia(ind)
    if regimen == Regimen.LATERAL:
        return estrategia_mean_reversion(ind)
    if regimen == Regimen.TENDENCIA:
        return estrategia_trend_following(ind)
    return SenalEstrategia(Estrategia.NINGUNA, Direccion.NEUTRAL, 0.0, f"Régimen {regimen.value}")


# ─────────────────────────────────────────────────────────────
# Claude (un solo call, no dos)
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
    """
    Un solo call a Haiku (barato). Sonnet solo se usa si fuerza > 0.8 (alta convicción).
    """
    modelo = MODELO_SONNET if senal.fuerza > 0.8 else MODELO_HAIKU

    prompt = f"""Trader cripto profesional. Validar operación en Azuro.

ACTIVO: {activo} | SESIÓN: {sesion_nombre} | RÉGIMEN: {regimen.value}
ESTRATEGIA: {senal.estrategia.value} → {senal.direccion.value} (fuerza {senal.fuerza:.2f})
RAZÓN: {senal.razon}

INDICADORES:
RSI {ind.rsi:.1f} | ADX {ind.adx:.1f} | ATR% {ind.atr_pct:.2f}
EMA9/21/50: {ind.ema_9:.2f}/{ind.ema_21:.2f}/{ind.ema_50:.2f}
BB: {ind.bollinger_lower:.2f}/{ind.bollinger_middle:.2f}/{ind.bollinger_upper:.2f} (ancho {ind.bb_width_pct:.2f}%)
Precio: {ind.precio_actual:.2f} | Vol: {ind.volumen_actual:.0f} vs {ind.volumen_promedio:.0f}
Cambio 5m/15m: {ind.cambio_pct_5min:+.2f}% / {ind.cambio_pct_15min:+.2f}%
Funding: {ind.funding_rate:+.4f} | L/S ratio: {ind.long_short_ratio:.2f}

¿Confirmás la dirección? Considerá: coherencia técnica, sentiment (funding/ratio), sesión, riesgo de reversión.

SOLO JSON:
{{"probabilidad": 0.0-1.0, "confianza": "ALTA"|"MEDIA"|"BAJA", "razon": "1 frase"}}
"""
    return _llamar_claude(modelo, prompt, max_tokens=300)


# ─────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────

def analizar(activo: str) -> Optional[Senal]:
    sesion = get_sesion_actual()
    if not sesion.puede_operar:
        return None

    snapshot = get_snapshot_completo(activo)
    ind = calcular_indicadores(snapshot)

    # NUEVO: detectar régimen primero
    regimen = detectar_regimen(ind)
    if regimen == Regimen.INDEFINIDO:
        return None

    # Activar solo la estrategia del régimen
    senal_est = evaluar_por_regimen(regimen, ind)
    if senal_est.estrategia == Estrategia.NINGUNA or senal_est.direccion == Direccion.NEUTRAL:
        return None

    # Un solo call a Claude
    try:
        decision = validar_con_claude(activo, senal_est, ind, regimen, sesion.nombre)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"⚠️  Error Claude en {activo}: {e}")
        return None

    prob = decision.get("probabilidad", 0)
    if prob < PROBABILIDAD_MINIMA_FINAL:
        return None

    return Senal(
        activo=activo,
        estrategia=senal_est.estrategia,
        direccion=senal_est.direccion,
        probabilidad=prob,
        confianza=decision.get("confianza", "MEDIA"),
        razon=decision.get("razon", senal_est.razon),
        regimen=regimen.value,
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
    print("TEST analyst.py v2 (régimen adaptativo)")
    print("=" * 60)
    for activo in ["BTC", "ETH", "SOL"]:
        print(f"\n🔍 {activo}")
        try:
            snap = get_snapshot_completo(activo)
            ind = calcular_indicadores(snap)
            reg = detectar_regimen(ind)
            print(f"  RSI {ind.rsi:.1f} | ADX {ind.adx:.1f} | ATR% {ind.atr_pct:.2f}")
            print(f"  Régimen: {reg.value}")
            senal_est = evaluar_por_regimen(reg, ind)
            print(f"  Estrategia: {senal_est.estrategia.value} → {senal_est.direccion.value} ({senal_est.fuerza:.2f})")
            if senal_est.direccion != Direccion.NEUTRAL:
                print(f"  🤖 Claude...")
                s = analizar(activo)
                if s:
                    print(f"  ✅ {s.direccion.value} prob={s.probabilidad:.2f} [{s.confianza}]")
                else:
                    print(f"  ❌ Claude descartó")
            else:
                print(f"  ⏭️  Sin señal (ahorro)")
        except Exception as e:
            print(f"  ❌ {e}")