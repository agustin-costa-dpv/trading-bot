"""
scripts/backtest.py
Backtest determinístico de la estrategia v2 (régimen adaptativo).

NO usa Claude — solo lógica técnica sobre datos históricos de Binance.
Gratis, rápido, reproducible.

Uso:
    python -m scripts.backtest
    python -m scripts.backtest --meses 6 --activos BTC ETH SOL
    python -m scripts.backtest --tp 1.5 --sl 1.0 --horizonte 4

Métricas calculadas:
- Total de señales por estrategia
- Win rate por estrategia
- Profit factor (ganancias totales / pérdidas totales)
- Sharpe ratio anualizado
- Drawdown máximo
- Distribución de regímenes
- Mejor y peor trade
"""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pandas_ta as ta
import requests


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

BINANCE_BASE = "https://data-api.binance.vision"

SIMBOLOS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

# Mismos parámetros que analyst.py v2
ADX_TENDENCIA = 22
ADX_LATERAL = 18

# Parámetros del trade (configurables por CLI)
TP_DEFAULT_PCT = 1.5     # take profit en %
SL_DEFAULT_PCT = 1.0     # stop loss en %
HORIZONTE_DEFAULT = 4    # velas de 15m = 1 hora


# ─────────────────────────────────────────────────────────────
# Carga de datos históricos
# ─────────────────────────────────────────────────────────────

def descargar_velas_historicas(simbolo: str, meses: int = 6, intervalo: str = "15m") -> pd.DataFrame:
    """
    Descarga velas de Binance paginando (máximo 1000 por call).
    Para 6 meses de 15m necesitamos ~17,500 velas = 18 calls.
    """
    ahora_ms = int(datetime.now().timestamp() * 1000)
    inicio_ms = int((datetime.now() - timedelta(days=meses * 30)).timestamp() * 1000)

    todas_las_velas = []
    cursor = inicio_ms

    print(f"  Descargando {meses} meses de {simbolo} ({intervalo})...")
    while cursor < ahora_ms:
        r = requests.get(
            f"{BINANCE_BASE}/api/v3/klines",
            params={
                "symbol": simbolo,
                "interval": intervalo,
                "startTime": cursor,
                "limit": 1000,
            },
            timeout=15,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break

        todas_las_velas.extend(batch)
        cursor = batch[-1][0] + 1
        time.sleep(0.15)  # respetar rate limit

    df = pd.DataFrame(todas_las_velas, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
    print(f"  ✓ {len(df):,} velas descargadas")
    return df


# ─────────────────────────────────────────────────────────────
# Indicadores (mismos que analyst.py)
# ─────────────────────────────────────────────────────────────

def calcular_indicadores_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula TODOS los indicadores de una vez sobre el DataFrame completo.
    Es 1000x más rápido que recalcular en cada vela.
    """
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
    df["cambio_pct_15min"] = df["close"].pct_change() * 100

    return df


def calcular_cambio_5min(df_15m: pd.DataFrame, df_5m: pd.DataFrame) -> pd.Series:
    """
    Para cada vela de 15m, calcula el cambio % de la última vela de 5m.
    Alineación temporal: para vela 15m que abre a T, usamos vela 5m que cierra cerca de T+15m.
    """
    df_5m = df_5m.copy()
    df_5m["cambio_5m"] = df_5m["close"].pct_change() * 100
    # Reindexar 5m a 15m usando último valor antes del siguiente bucket
    resampled = df_5m["cambio_5m"].resample("15min").last()
    return resampled.reindex(df_15m.index, method="ffill")


# ─────────────────────────────────────────────────────────────
# Detector de régimen y estrategias (copia de analyst.py)
# ─────────────────────────────────────────────────────────────

def detectar_regimen(row) -> str:
    cambio_5m = row.get("cambio_5min", 0) or 0
    atr_pct = row.get("atr_pct", 0) or 0
    adx = row.get("adx", 0) or 0

    if abs(cambio_5m) >= max(1.2, atr_pct * 1.2):
        return "MOVIMIENTO_BRUSCO"
    if adx >= ADX_TENDENCIA:
        return "TENDENCIA"
    if adx <= ADX_LATERAL:
        return "LATERAL"
    return "INDEFINIDO"


def senal_mean_reversion(row) -> Optional[tuple[str, float]]:
    """Devuelve (direccion, fuerza) o None."""
    rsi = row["rsi"]
    precio = row["close"]
    bb_lower = row["bb_lower"]
    bb_upper = row["bb_upper"]

    if pd.isna(rsi) or pd.isna(bb_lower):
        return None

    dist_lower_pct = ((precio - bb_lower) / precio) * 100
    dist_upper_pct = ((bb_upper - precio) / precio) * 100

    if rsi < 35 and dist_lower_pct < 0.5:
        fuerza = (35 - rsi) / 35 * 0.6 + (0.5 - dist_lower_pct) * 0.4
        return ("SUBE", min(1.0, fuerza))

    if rsi > 65 and dist_upper_pct < 0.5:
        fuerza = (rsi - 65) / 35 * 0.6 + (0.5 - dist_upper_pct) * 0.4
        return ("BAJA", min(1.0, fuerza))

    return None


def senal_trend_following(row) -> Optional[tuple[str, float]]:
    ema_9 = row["ema_9"]
    ema_21 = row["ema_21"]
    ema_50 = row["ema_50"]
    volumen = row["volume"]
    vol_prom = row["volumen_promedio"]
    adx = row["adx"]

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

    umbral = max(1.0, atr_pct * 1.2)

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
        s = senal_arbitraje(row)
        estrategia = "ARBITRAJE"
    elif regimen == "LATERAL":
        s = senal_mean_reversion(row)
        estrategia = "MEAN_REVERSION"
    else:
        s = senal_trend_following(row)
        estrategia = "TREND_FOLLOWING"

    if s is None:
        return None

    direccion, fuerza = s
    return {"regimen": regimen, "estrategia": estrategia, "direccion": direccion, "fuerza": fuerza}


# ─────────────────────────────────────────────────────────────
# Simulación de trades
# ─────────────────────────────────────────────────────────────

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
    resultado: str  # TP, SL, TIMEOUT
    fuerza: float


def simular_trade(df: pd.DataFrame, idx: int, senal: dict, activo: str,
                  tp_pct: float, sl_pct: float, horizonte: int) -> Optional[Trade]:
    """
    Dado un índice de entrada, simula el trade:
    - Sale en TP (take profit) si el precio llega antes
    - Sale en SL (stop loss) si el precio llega antes
    - Sale al final del horizonte si nada disparó
    """
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
        high = vela["high"]
        low = vela["low"]

        # Chequear TP y SL (asumimos que en una vela se ejecuta el peor escenario primero)
        if direccion == "SUBE":
            # Si la vela toca SL primero (conservador): asumimos SL si el low cruza SL
            if low <= sl:
                return Trade(
                    activo=activo,
                    entrada_timestamp=df.index[idx],
                    salida_timestamp=df.index[idx + i],
                    direccion=direccion,
                    estrategia=senal["estrategia"],
                    regimen=senal["regimen"],
                    precio_entrada=precio_entrada,
                    precio_salida=sl,
                    pnl_pct=-sl_pct,
                    resultado="SL",
                    fuerza=senal["fuerza"],
                )
            if high >= tp:
                return Trade(
                    activo=activo,
                    entrada_timestamp=df.index[idx],
                    salida_timestamp=df.index[idx + i],
                    direccion=direccion,
                    estrategia=senal["estrategia"],
                    regimen=senal["regimen"],
                    precio_entrada=precio_entrada,
                    precio_salida=tp,
                    pnl_pct=tp_pct,
                    resultado="TP",
                    fuerza=senal["fuerza"],
                )
        else:  # BAJA
            if high >= sl:
                return Trade(
                    activo=activo,
                    entrada_timestamp=df.index[idx],
                    salida_timestamp=df.index[idx + i],
                    direccion=direccion,
                    estrategia=senal["estrategia"],
                    regimen=senal["regimen"],
                    precio_entrada=precio_entrada,
                    precio_salida=sl,
                    pnl_pct=-sl_pct,
                    resultado="SL",
                    fuerza=senal["fuerza"],
                )
            if low <= tp:
                return Trade(
                    activo=activo,
                    entrada_timestamp=df.index[idx],
                    salida_timestamp=df.index[idx + i],
                    direccion=direccion,
                    estrategia=senal["estrategia"],
                    regimen=senal["regimen"],
                    precio_entrada=precio_entrada,
                    precio_salida=tp,
                    pnl_pct=tp_pct,
                    resultado="TP",
                    fuerza=senal["fuerza"],
                )

    # Timeout: salida al último precio
    precio_salida = df.iloc[idx + horizonte]["close"]
    if direccion == "SUBE":
        pnl = (precio_salida - precio_entrada) / precio_entrada * 100
    else:
        pnl = (precio_entrada - precio_salida) / precio_entrada * 100

    return Trade(
        activo=activo,
        entrada_timestamp=df.index[idx],
        salida_timestamp=df.index[idx + horizonte],
        direccion=direccion,
        estrategia=senal["estrategia"],
        regimen=senal["regimen"],
        precio_entrada=precio_entrada,
        precio_salida=precio_salida,
        pnl_pct=pnl,
        resultado="TIMEOUT",
        fuerza=senal["fuerza"],
    )


def backtest_activo(activo: str, meses: int, tp_pct: float, sl_pct: float,
                     horizonte: int) -> tuple[list[Trade], dict]:
    """Corre el backtest sobre un activo y devuelve trades + stats de régimen."""
    simbolo = SIMBOLOS[activo]

    df_15m = descargar_velas_historicas(simbolo, meses, "15m")
    df_5m = descargar_velas_historicas(simbolo, meses, "5m")

    print(f"  Calculando indicadores...")
    df = calcular_indicadores_df(df_15m)
    df["cambio_pct_5min"] = calcular_cambio_5min(df, df_5m)
    df = df.rename(columns={"cambio_pct_5min": "cambio_5min"})

    print(f"  Simulando trades...")
    trades = []
    regimen_counts = {"LATERAL": 0, "TENDENCIA": 0, "MOVIMIENTO_BRUSCO": 0, "INDEFINIDO": 0}

    # Cooldown: después de un trade no volver a entrar por `horizonte` velas
    cooldown_hasta = -1

    for idx in range(50, len(df)):  # empezar en 50 para que EMA50 esté calculada
        row = df.iloc[idx]
        regimen = detectar_regimen(row)
        regimen_counts[regimen] += 1

        if idx < cooldown_hasta:
            continue

        senal = evaluar(row)
        if senal is None:
            continue

        trade = simular_trade(df, idx, senal, activo, tp_pct, sl_pct, horizonte)
        if trade:
            trades.append(trade)
            cooldown_hasta = idx + horizonte

    print(f"  ✓ {len(trades)} trades generados")
    return trades, regimen_counts


# ─────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────

def calcular_metricas(trades: list[Trade]) -> dict:
    if not trades:
        return {"total": 0}

    pnls = [t.pnl_pct for t in trades]
    ganadores = [p for p in pnls if p > 0]
    perdedores = [p for p in pnls if p < 0]

    total = len(trades)
    win_rate = len(ganadores) / total * 100
    avg_ganador = sum(ganadores) / len(ganadores) if ganadores else 0
    avg_perdedor = sum(perdedores) / len(perdedores) if perdedores else 0

    ganancia_total = sum(ganadores)
    perdida_total = abs(sum(perdedores))
    profit_factor = ganancia_total / perdida_total if perdida_total > 0 else float("inf")

    # Sharpe simplificado (asume trades ~uniformes en el tiempo)
    df_pnl = pd.Series(pnls)
    if df_pnl.std() > 0:
        sharpe = (df_pnl.mean() / df_pnl.std()) * (252 ** 0.5)
    else:
        sharpe = 0

    # Drawdown máximo (sobre equity curve)
    equity = pd.Series(pnls).cumsum()
    running_max = equity.cummax()
    drawdown = (equity - running_max)
    max_dd = drawdown.min()

    pnl_neto = sum(pnls)

    return {
        "total": total,
        "win_rate": win_rate,
        "avg_ganador": avg_ganador,
        "avg_perdedor": avg_perdedor,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "pnl_neto_pct": pnl_neto,
        "mejor_trade": max(pnls),
        "peor_trade": min(pnls),
    }


def metricas_por_estrategia(trades: list[Trade]) -> dict:
    por_est = {}
    for est in ["MEAN_REVERSION", "TREND_FOLLOWING", "ARBITRAJE"]:
        subset = [t for t in trades if t.estrategia == est]
        por_est[est] = calcular_metricas(subset)
    return por_est


def imprimir_reporte(activo: str, trades: list[Trade], regimen_counts: dict, params: dict):
    print(f"\n{'=' * 60}")
    print(f"  RESULTADOS: {activo}")
    print(f"{'=' * 60}")

    total_velas = sum(regimen_counts.values())
    print(f"\n📊 DISTRIBUCIÓN DE RÉGIMEN ({total_velas:,} velas):")
    for regimen, count in regimen_counts.items():
        pct = count / total_velas * 100 if total_velas else 0
        print(f"  {regimen:20s}  {count:6,}  ({pct:5.1f}%)")

    metricas = calcular_metricas(trades)
    if metricas["total"] == 0:
        print(f"\n⚠️  Sin trades generados")
        return

    print(f"\n💹 MÉTRICAS GLOBALES:")
    print(f"  Total trades:       {metricas['total']}")
    print(f"  Win rate:           {metricas['win_rate']:.1f}%")
    print(f"  PnL neto:           {metricas['pnl_neto_pct']:+.2f}%")
    print(f"  Profit factor:      {metricas['profit_factor']:.2f}")
    print(f"  Sharpe (anual):     {metricas['sharpe']:.2f}")
    print(f"  Max drawdown:       {metricas['max_drawdown']:.2f}%")
    print(f"  Avg ganador:        +{metricas['avg_ganador']:.2f}%")
    print(f"  Avg perdedor:       {metricas['avg_perdedor']:.2f}%")
    print(f"  Mejor trade:        +{metricas['mejor_trade']:.2f}%")
    print(f"  Peor trade:         {metricas['peor_trade']:.2f}%")

    print(f"\n📈 POR ESTRATEGIA:")
    por_est = metricas_por_estrategia(trades)
    for est, m in por_est.items():
        if m["total"] == 0:
            print(f"  {est:18s}  0 trades")
            continue
        print(f"  {est:18s}  n={m['total']:3d}  WR={m['win_rate']:5.1f}%  "
              f"PnL={m['pnl_neto_pct']:+6.2f}%  PF={m['profit_factor']:.2f}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meses", type=int, default=6)
    parser.add_argument("--activos", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--tp", type=float, default=TP_DEFAULT_PCT)
    parser.add_argument("--sl", type=float, default=SL_DEFAULT_PCT)
    parser.add_argument("--horizonte", type=int, default=HORIZONTE_DEFAULT)
    args = parser.parse_args()

    params = {"tp_pct": args.tp, "sl_pct": args.sl, "horizonte": args.horizonte}

    print(f"\n{'#' * 60}")
    print(f"#  BACKTEST v2 — {args.meses} meses")
    print(f"#  TP={args.tp}% | SL={args.sl}% | Horizonte={args.horizonte} velas (15m)")
    print(f"#  Risk/Reward: 1:{args.tp/args.sl:.2f}")
    print(f"{'#' * 60}")

    todos_trades = []
    todos_regimenes = {"LATERAL": 0, "TENDENCIA": 0, "MOVIMIENTO_BRUSCO": 0, "INDEFINIDO": 0}

    for activo in args.activos:
        print(f"\n🔍 Procesando {activo}...")
        try:
            trades, regimenes = backtest_activo(
                activo, args.meses, args.tp, args.sl, args.horizonte,
            )
            imprimir_reporte(activo, trades, regimenes, params)
            todos_trades.extend(trades)
            for k, v in regimenes.items():
                todos_regimenes[k] += v
        except Exception as e:
            print(f"  ❌ Error: {e}")

    if len(args.activos) > 1:
        print(f"\n{'#' * 60}")
        print(f"#  CONSOLIDADO")
        print(f"{'#' * 60}")
        imprimir_reporte("TODOS", todos_trades, todos_regimenes, params)

    # Guardar trades en CSV para análisis posterior
    if todos_trades:
        df_trades = pd.DataFrame([{
            "activo": t.activo,
            "entrada": t.entrada_timestamp,
            "salida": t.salida_timestamp,
            "direccion": t.direccion,
            "estrategia": t.estrategia,
            "regimen": t.regimen,
            "precio_entrada": t.precio_entrada,
            "precio_salida": t.precio_salida,
            "pnl_pct": t.pnl_pct,
            "resultado": t.resultado,
            "fuerza": t.fuerza,
        } for t in todos_trades])
        df_trades.to_csv("backtest_trades.csv", index=False)
        print(f"\n💾 Trades guardados en backtest_trades.csv")


if __name__ == "__main__":
    main()
