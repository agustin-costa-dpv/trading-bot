"""
scripts/backtest_v2.py
Backtest optimizado: parámetros granulares por estrategia.

Cambios vs backtest.py:
- ADX_TENDENCIA 22 → 28 (filtrar ruido direccional)
- Umbral ARBITRAJE 1.2×ATR → 0.8×ATR (disparar más)
- TP/SL/Horizonte específicos por estrategia

Uso:
    python -m scripts.backtest_v2 --meses 6
"""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pandas_ta as ta
import requests


BINANCE_BASE = "https://data-api.binance.vision"
SIMBOLOS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

# ────────────────────────────────────────────
# PARÁMETROS OPTIMIZADOS
# ────────────────────────────────────────────

# Régimen (más estrictos)
ADX_TENDENCIA = 28         # antes 22
ADX_LATERAL = 18           # igual

# Arbitraje más sensible (antes 1.2)
ARBITRAJE_ATR_MULT = 0.8

# Parámetros por estrategia: (tp%, sl%, horizonte_velas_15m)
PARAMS_ESTRATEGIA = {
    "MEAN_REVERSION":  {"tp": 1.0, "sl": 0.8, "horizonte": 3},  # reversiones chicas y rápidas
    "TREND_FOLLOWING": {"tp": 2.0, "sl": 1.0, "horizonte": 6},  # dejar correr tendencias
    "ARBITRAJE":       {"tp": 1.5, "sl": 0.7, "horizonte": 2},  # rápido, alta convicción
}


# ────────────────────────────────────────────
# Descarga
# ────────────────────────────────────────────

def descargar_velas_historicas(simbolo: str, meses: int = 6, intervalo: str = "15m") -> pd.DataFrame:
    ahora_ms = int(datetime.now().timestamp() * 1000)
    inicio_ms = int((datetime.now() - timedelta(days=meses * 30)).timestamp() * 1000)
    todas = []
    cursor = inicio_ms
    print(f"  Descargando {meses}m {simbolo} ({intervalo})...")
    while cursor < ahora_ms:
        r = requests.get(
            f"{BINANCE_BASE}/api/v3/klines",
            params={"symbol": simbolo, "interval": intervalo,
                    "startTime": cursor, "limit": 1000},
            timeout=15,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        todas.extend(batch)
        cursor = batch[-1][0] + 1
        time.sleep(0.15)
    df = pd.DataFrame(todas, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
    print(f"  ✓ {len(df):,} velas")
    return df


# ────────────────────────────────────────────
# Indicadores
# ────────────────────────────────────────────

def calcular_indicadores_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["ema_50"] = ta.ema(df["close"], length=50)
    bb = ta.bbands(df["close"], length=20, std=2)
    df["bb_lower"] = bb.iloc[:, 0]
    df["bb_middle"] = bb.iloc[:, 1]
    df["bb_upper"] = bb.iloc[:, 2]
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx"] = adx_df.iloc[:, 0]
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr_pct"] = (atr / df["close"]) * 100
    df["volumen_promedio"] = df["volume"].rolling(20).mean()
    return df


def calcular_cambio_5min(df_15m: pd.DataFrame, df_5m: pd.DataFrame) -> pd.Series:
    df_5m = df_5m.copy()
    df_5m["cambio_5m"] = df_5m["close"].pct_change() * 100
    resampled = df_5m["cambio_5m"].resample("15min").last()
    return resampled.reindex(df_15m.index, method="ffill")


# ────────────────────────────────────────────
# Régimen y estrategias
# ────────────────────────────────────────────

def detectar_regimen(row) -> str:
    cambio_5m = row.get("cambio_5min", 0) or 0
    atr_pct = row.get("atr_pct", 0) or 0
    adx = row.get("adx", 0) or 0

    # Arbitraje con umbral más bajo
    if abs(cambio_5m) >= max(0.7, atr_pct * ARBITRAJE_ATR_MULT):
        return "MOVIMIENTO_BRUSCO"
    if adx >= ADX_TENDENCIA:
        return "TENDENCIA"
    if adx <= ADX_LATERAL:
        return "LATERAL"
    return "INDEFINIDO"


def senal_mean_reversion(row) -> Optional[tuple[str, float]]:
    rsi, precio = row["rsi"], row["close"]
    bb_lower, bb_upper = row["bb_lower"], row["bb_upper"]
    if pd.isna(rsi) or pd.isna(bb_lower):
        return None
    dist_lower = ((precio - bb_lower) / precio) * 100
    dist_upper = ((bb_upper - precio) / precio) * 100
    if rsi < 35 and dist_lower < 0.5:
        fuerza = (35 - rsi) / 35 * 0.6 + (0.5 - dist_lower) * 0.4
        return ("SUBE", min(1.0, fuerza))
    if rsi > 65 and dist_upper < 0.5:
        fuerza = (rsi - 65) / 35 * 0.6 + (0.5 - dist_upper) * 0.4
        return ("BAJA", min(1.0, fuerza))
    return None


def senal_trend_following(row) -> Optional[tuple[str, float]]:
    ema_9, ema_21, ema_50 = row["ema_9"], row["ema_21"], row["ema_50"]
    volumen, vol_prom, adx = row["volume"], row["volumen_promedio"], row["adx"]
    if any(pd.isna(x) for x in [ema_9, ema_21, ema_50, vol_prom, adx]):
        return None
    dist_pct = abs(ema_9 - ema_21) / ema_21 * 100
    volumen_ok = volumen >= vol_prom * 0.9
    if ema_9 > ema_21 > ema_50 and dist_pct > 0.05 and volumen_ok:
        return ("SUBE", min(1.0, dist_pct / 0.8 + adx / 100))
    if ema_9 < ema_21 < ema_50 and dist_pct > 0.05 and volumen_ok:
        return ("BAJA", min(1.0, dist_pct / 0.8 + adx / 100))
    return None


def senal_arbitraje(row) -> Optional[tuple[str, float]]:
    cambio_5m = row.get("cambio_5min", 0) or 0
    atr_pct = row.get("atr_pct", 0) or 0
    umbral = max(0.7, atr_pct * ARBITRAJE_ATR_MULT)
    if cambio_5m >= umbral:
        return ("SUBE", min(1.0, cambio_5m / (umbral * 2)))
    if cambio_5m <= -umbral:
        return ("BAJA", min(1.0, abs(cambio_5m) / (umbral * 2)))
    return None


def evaluar(row) -> Optional[dict]:
    regimen = detectar_regimen(row)
    if regimen == "INDEFINIDO":
        return None
    if regimen == "MOVIMIENTO_BRUSCO":
        s, estrategia = senal_arbitraje(row), "ARBITRAJE"
    elif regimen == "LATERAL":
        s, estrategia = senal_mean_reversion(row), "MEAN_REVERSION"
    else:
        s, estrategia = senal_trend_following(row), "TREND_FOLLOWING"
    if s is None:
        return None
    direccion, fuerza = s
    return {"regimen": regimen, "estrategia": estrategia,
            "direccion": direccion, "fuerza": fuerza}


# ────────────────────────────────────────────
# Simulación con params por estrategia
# ────────────────────────────────────────────

@dataclass
class Trade:
    activo: str
    entrada_timestamp: pd.Timestamp
    salida_timestamp: pd.Timestamp
    direccion: str
    estrategia: str
    regimen: str
    precio_entrada: float
    precio_salida: float
    pnl_pct: float
    resultado: str
    fuerza: float


def simular_trade(df: pd.DataFrame, idx: int, senal: dict, activo: str) -> Optional[Trade]:
    params = PARAMS_ESTRATEGIA[senal["estrategia"]]
    tp_pct, sl_pct, horizonte = params["tp"], params["sl"], params["horizonte"]

    if idx + horizonte >= len(df):
        return None

    precio_entrada = df.iloc[idx]["close"]
    direccion = senal["direccion"]

    if direccion == "SUBE":
        tp = precio_entrada * (1 + tp_pct / 100)
        sl = precio_entrada * (1 - sl_pct / 100)
    else:
        tp = precio_entrada * (1 - tp_pct / 100)
        sl = precio_entrada * (1 + sl_pct / 100)

    for i in range(1, horizonte + 1):
        vela = df.iloc[idx + i]
        high, low = vela["high"], vela["low"]

        if direccion == "SUBE":
            if low <= sl:
                return _crear_trade(df, idx, idx + i, senal, activo,
                                     precio_entrada, sl, -sl_pct, "SL")
            if high >= tp:
                return _crear_trade(df, idx, idx + i, senal, activo,
                                     precio_entrada, tp, tp_pct, "TP")
        else:
            if high >= sl:
                return _crear_trade(df, idx, idx + i, senal, activo,
                                     precio_entrada, sl, -sl_pct, "SL")
            if low <= tp:
                return _crear_trade(df, idx, idx + i, senal, activo,
                                     precio_entrada, tp, tp_pct, "TP")

    precio_salida = df.iloc[idx + horizonte]["close"]
    if direccion == "SUBE":
        pnl = (precio_salida - precio_entrada) / precio_entrada * 100
    else:
        pnl = (precio_entrada - precio_salida) / precio_entrada * 100
    return _crear_trade(df, idx, idx + horizonte, senal, activo,
                         precio_entrada, precio_salida, pnl, "TIMEOUT")


def _crear_trade(df, idx_in, idx_out, senal, activo, p_in, p_out, pnl, resultado):
    return Trade(
        activo=activo,
        entrada_timestamp=df.index[idx_in],
        salida_timestamp=df.index[idx_out],
        direccion=senal["direccion"],
        estrategia=senal["estrategia"],
        regimen=senal["regimen"],
        precio_entrada=p_in,
        precio_salida=p_out,
        pnl_pct=pnl,
        resultado=resultado,
        fuerza=senal["fuerza"],
    )


def backtest_activo(activo: str, meses: int) -> tuple[list[Trade], dict]:
    simbolo = SIMBOLOS[activo]
    df_15m = descargar_velas_historicas(simbolo, meses, "15m")
    df_5m = descargar_velas_historicas(simbolo, meses, "5m")

    print(f"  Calculando indicadores...")
    df = calcular_indicadores_df(df_15m)
    df["cambio_5min"] = calcular_cambio_5min(df, df_5m)

    print(f"  Simulando trades...")
    trades = []
    regimen_counts = {"LATERAL": 0, "TENDENCIA": 0, "MOVIMIENTO_BRUSCO": 0, "INDEFINIDO": 0}
    cooldown_hasta = -1

    for idx in range(50, len(df)):
        row = df.iloc[idx]
        regimen = detectar_regimen(row)
        regimen_counts[regimen] += 1

        if idx < cooldown_hasta:
            continue

        senal = evaluar(row)
        if senal is None:
            continue

        trade = simular_trade(df, idx, senal, activo)
        if trade:
            trades.append(trade)
            # Cooldown = horizonte de esa estrategia
            cooldown_hasta = idx + PARAMS_ESTRATEGIA[senal["estrategia"]]["horizonte"]

    print(f"  ✓ {len(trades)} trades")
    return trades, regimen_counts


# ────────────────────────────────────────────
# Métricas
# ────────────────────────────────────────────

def calcular_metricas(trades: list[Trade]) -> dict:
    if not trades:
        return {"total": 0}
    pnls = [t.pnl_pct for t in trades]
    ganadores = [p for p in pnls if p > 0]
    perdedores = [p for p in pnls if p < 0]
    total = len(trades)
    win_rate = len(ganadores) / total * 100
    avg_g = sum(ganadores) / len(ganadores) if ganadores else 0
    avg_p = sum(perdedores) / len(perdedores) if perdedores else 0
    g_total = sum(ganadores)
    p_total = abs(sum(perdedores))
    pf = g_total / p_total if p_total > 0 else float("inf")
    s = pd.Series(pnls)
    sharpe = (s.mean() / s.std()) * (252 ** 0.5) if s.std() > 0 else 0
    equity = s.cumsum()
    dd = (equity - equity.cummax()).min()
    return {
        "total": total, "win_rate": win_rate,
        "avg_ganador": avg_g, "avg_perdedor": avg_p,
        "profit_factor": pf, "sharpe": sharpe,
        "max_drawdown": dd, "pnl_neto_pct": sum(pnls),
        "mejor_trade": max(pnls), "peor_trade": min(pnls),
    }


def metricas_por_estrategia(trades: list[Trade]) -> dict:
    return {est: calcular_metricas([t for t in trades if t.estrategia == est])
            for est in ["MEAN_REVERSION", "TREND_FOLLOWING", "ARBITRAJE"]}


def imprimir_reporte(activo: str, trades: list[Trade], regimen_counts: dict):
    print(f"\n{'=' * 60}")
    print(f"  RESULTADOS: {activo}")
    print(f"{'=' * 60}")

    total_v = sum(regimen_counts.values())
    print(f"\n📊 RÉGIMEN ({total_v:,} velas):")
    for reg, c in regimen_counts.items():
        pct = c / total_v * 100 if total_v else 0
        print(f"  {reg:20s}  {c:6,}  ({pct:5.1f}%)")

    m = calcular_metricas(trades)
    if m["total"] == 0:
        print(f"\n⚠️  Sin trades")
        return

    print(f"\n💹 GLOBAL:")
    print(f"  Trades:         {m['total']}")
    print(f"  Win rate:       {m['win_rate']:.1f}%")
    print(f"  PnL neto:       {m['pnl_neto_pct']:+.2f}%")
    print(f"  Profit factor:  {m['profit_factor']:.2f}")
    print(f"  Sharpe:         {m['sharpe']:.2f}")
    print(f"  Max drawdown:   {m['max_drawdown']:.2f}%")
    print(f"  Avg G/P:        +{m['avg_ganador']:.2f}% / {m['avg_perdedor']:.2f}%")

    print(f"\n📈 POR ESTRATEGIA:")
    for est, me in metricas_por_estrategia(trades).items():
        if me["total"] == 0:
            print(f"  {est:18s}  0 trades")
            continue
        params = PARAMS_ESTRATEGIA[est]
        print(f"  {est:18s}  n={me['total']:4d}  WR={me['win_rate']:5.1f}%  "
              f"PnL={me['pnl_neto_pct']:+7.2f}%  PF={me['profit_factor']:.2f}  "
              f"[TP{params['tp']}/SL{params['sl']}/H{params['horizonte']}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meses", type=int, default=6)
    parser.add_argument("--activos", nargs="+", default=["BTC", "ETH", "SOL"])
    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print(f"#  BACKTEST v2 OPTIMIZADO — {args.meses} meses")
    print(f"#  ADX_TEND=28 | ARB_MULT=0.8×ATR | Params por estrategia")
    print(f"{'#' * 60}")

    todos_trades = []
    todos_reg = {"LATERAL": 0, "TENDENCIA": 0, "MOVIMIENTO_BRUSCO": 0, "INDEFINIDO": 0}

    for activo in args.activos:
        print(f"\n🔍 {activo}...")
        try:
            trades, reg = backtest_activo(activo, args.meses)
            imprimir_reporte(activo, trades, reg)
            todos_trades.extend(trades)
            for k, v in reg.items():
                todos_reg[k] += v
        except Exception as e:
            print(f"  ❌ {e}")

    if len(args.activos) > 1:
        print(f"\n{'#' * 60}")
        print(f"#  CONSOLIDADO")
        print(f"{'#' * 60}")
        imprimir_reporte("TODOS", todos_trades, todos_reg)

    if todos_trades:
        pd.DataFrame([{
            "activo": t.activo, "entrada": t.entrada_timestamp,
            "salida": t.salida_timestamp, "direccion": t.direccion,
            "estrategia": t.estrategia, "regimen": t.regimen,
            "precio_entrada": t.precio_entrada, "precio_salida": t.precio_salida,
            "pnl_pct": t.pnl_pct, "resultado": t.resultado, "fuerza": t.fuerza,
        } for t in todos_trades]).to_csv("backtest_v2_trades.csv", index=False)
        print(f"\n💾 backtest_v2_trades.csv guardado")


if __name__ == "__main__":
    main()