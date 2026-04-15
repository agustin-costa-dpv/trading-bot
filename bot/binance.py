"""
bot/binance.py
Cliente para Binance API pública (sin auth, solo lectura).
Provee precios, velas (klines), volumen y order book para BTC/ETH/SOL.

Endpoints usados (todos públicos, sin API key):
- /api/v3/ticker/price        → precio actual
- /api/v3/ticker/24hr         → stats 24h (volumen, cambio %)
- /api/v3/klines              → velas históricas (para RSI, EMA, Bollinger)
- /api/v3/depth               → order book (para Nivel 3)

Rate limit: 1200 requests/min por IP. Estamos MUY por debajo.
"""

import requests
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


BINANCE_BASE = "https://data-api.binance.vision"

# Símbolos que vamos a operar (par contra USDT)
SIMBOLOS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}

# Intervalos de velas soportados por Binance
Intervalo = Literal["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]


# ─────────────────────────────────────────────────────────────
# Modelos de datos
# ─────────────────────────────────────────────────────────────

@dataclass
class Vela:
    """Una vela (kline) de Binance."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class Stats24h:
    """Estadísticas de las últimas 24 horas."""
    simbolo: str
    precio_actual: float
    cambio_pct_24h: float        # % de cambio en 24h
    volumen_24h: float            # en moneda base (ej: BTC)
    volumen_quote_24h: float      # en USDT
    high_24h: float
    low_24h: float


@dataclass
class OrderBook:
    """Snapshot del order book."""
    simbolo: str
    bids: list[tuple[float, float]] = field(default_factory=list)  # [(precio, cantidad)]
    asks: list[tuple[float, float]] = field(default_factory=list)

    @property
    def mejor_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def mejor_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def spread(self) -> float:
        return self.mejor_ask - self.mejor_bid

    @property
    def spread_pct(self) -> float:
        if self.mejor_bid == 0:
            return 0.0
        return (self.spread / self.mejor_bid) * 100


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _resolver_simbolo(activo: str) -> str:
    """Convierte 'BTC' → 'BTCUSDT'. Acepta también el símbolo completo."""
    activo = activo.upper()
    if activo in SIMBOLOS:
        return SIMBOLOS[activo]
    if activo.endswith("USDT"):
        return activo
    raise ValueError(f"Activo no soportado: {activo}. Usá BTC, ETH o SOL.")


def _request(endpoint: str, params: dict | None = None) -> dict | list:
    """Wrapper de requests con manejo de errores."""
    url = f"{BINANCE_BASE}{endpoint}"
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Timeout consultando Binance: {endpoint}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Error HTTP {r.status_code} en Binance: {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error de red consultando Binance: {e}")


# ─────────────────────────────────────────────────────────────
# Funciones públicas
# ─────────────────────────────────────────────────────────────

def get_precio(activo: str) -> float:
    """
    Precio actual del activo en USDT.

    Args:
        activo: 'BTC', 'ETH', 'SOL' (o símbolo completo tipo 'BTCUSDT')

    Returns:
        Precio en USDT como float.
    """
    simbolo = _resolver_simbolo(activo)
    data = _request("/api/v3/ticker/price", {"symbol": simbolo})
    return float(data["price"])


def get_precios_todos() -> dict[str, float]:
    """Precio actual de BTC, ETH y SOL en una sola llamada."""
    return {nombre: get_precio(nombre) for nombre in SIMBOLOS.keys()}


def get_stats_24h(activo: str) -> Stats24h:
    """
    Stats de las últimas 24 horas: precio, cambio %, volumen, high, low.
    Útil para detectar movimientos fuertes (Estrategia 3 - Arbitraje).
    """
    simbolo = _resolver_simbolo(activo)
    data = _request("/api/v3/ticker/24hr", {"symbol": simbolo})
    return Stats24h(
        simbolo=simbolo,
        precio_actual=float(data["lastPrice"]),
        cambio_pct_24h=float(data["priceChangePercent"]),
        volumen_24h=float(data["volume"]),
        volumen_quote_24h=float(data["quoteVolume"]),
        high_24h=float(data["highPrice"]),
        low_24h=float(data["lowPrice"]),
    )


def get_velas(
    activo: str,
    intervalo: Intervalo = "15m",
    limite: int = 100,
) -> list[Vela]:
    """
    Velas históricas (klines) para calcular indicadores técnicos.

    Args:
        activo: 'BTC', 'ETH', 'SOL'
        intervalo: '1m', '5m', '15m', '1h', '4h', '1d', etc.
        limite: cantidad de velas (máx 1000, default 100)

    Returns:
        Lista de Vela ordenada de más vieja a más nueva.

    Para nuestras estrategias usamos:
    - Mean Reversion / Bollinger: 15m, últimas 100 velas
    - EMA 9/21 cross: 15m, últimas 50 velas
    - Detección de movimientos fuertes: 5m, últimas 12 velas (1 hora)
    """
    simbolo = _resolver_simbolo(activo)
    data = _request("/api/v3/klines", {
        "symbol": simbolo,
        "interval": intervalo,
        "limit": min(limite, 1000),
    })
    return [
        Vela(
            timestamp=datetime.fromtimestamp(v[0] / 1000),
            open=float(v[1]),
            high=float(v[2]),
            low=float(v[3]),
            close=float(v[4]),
            volume=float(v[5]),
        )
        for v in data
    ]


def get_order_book(activo: str, profundidad: int = 20) -> OrderBook:
    """
    Snapshot del order book (bids y asks).

    Args:
        activo: 'BTC', 'ETH', 'SOL'
        profundidad: niveles a traer (5, 10, 20, 50, 100, 500, 1000, 5000)

    Returns:
        OrderBook con bids/asks ordenados por mejor precio.

    Uso futuro (Nivel 3): detectar paredes de liquidez institucional,
    calcular spread real, identificar absorción de órdenes.
    """
    simbolo = _resolver_simbolo(activo)
    data = _request("/api/v3/depth", {"symbol": simbolo, "limit": profundidad})
    return OrderBook(
        simbolo=simbolo,
        bids=[(float(p), float(q)) for p, q in data["bids"]],
        asks=[(float(p), float(q)) for p, q in data["asks"]],
    )


def get_snapshot_completo(activo: str) -> dict:
    """
    Snapshot completo de un activo: precio + stats 24h + velas + order book.
    Pensado para alimentar al agente IA en una sola llamada conceptual.

    Returns:
        dict con todo lo necesario para que analyst.py decida.
    """
    stats = get_stats_24h(activo)
    velas_15m = get_velas(activo, "15m", 100)
    velas_5m = get_velas(activo, "5m", 60)
    order_book = get_order_book(activo, 20)

    return {
        "activo": activo,
        "precio": stats.precio_actual,
        "stats_24h": {
            "cambio_pct": stats.cambio_pct_24h,
            "volumen_usdt": stats.volumen_quote_24h,
            "high": stats.high_24h,
            "low": stats.low_24h,
        },
        "velas_15m": [v.to_dict() for v in velas_15m],
        "velas_5m": [v.to_dict() for v in velas_5m],
        "order_book": {
            "mejor_bid": order_book.mejor_bid,
            "mejor_ask": order_book.mejor_ask,
            "spread_pct": order_book.spread_pct,
            "bids_top10": order_book.bids[:10],
            "asks_top10": order_book.asks[:10],
        },
        "timestamp": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────
# Testing rápido — correr con: python -m bot.binance
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Binance API client")
    print("=" * 60)

    # Test 1: precios actuales
    print("\n📊 Precios actuales:")
    for nombre, precio in get_precios_todos().items():
        print(f"  {nombre}: ${precio:,.2f} USDT")

    # Test 2: stats 24h de BTC
    print("\n📈 Stats 24h BTC:")
    stats = get_stats_24h("BTC")
    print(f"  Precio:     ${stats.precio_actual:,.2f}")
    print(f"  Cambio 24h: {stats.cambio_pct_24h:+.2f}%")
    print(f"  Volumen:    ${stats.volumen_quote_24h:,.0f} USDT")
    print(f"  High 24h:   ${stats.high_24h:,.2f}")
    print(f"  Low 24h:    ${stats.low_24h:,.2f}")

    # Test 3: velas 15m
    print("\n🕯️  Últimas 5 velas 15m de BTC:")
    velas = get_velas("BTC", "15m", 5)
    for v in velas:
        print(f"  {v.timestamp.strftime('%H:%M')} | "
              f"O: {v.open:,.0f} H: {v.high:,.0f} "
              f"L: {v.low:,.0f} C: {v.close:,.0f} | "
              f"Vol: {v.volume:.2f}")

    # Test 4: order book
    print("\n📖 Order book BTC (top 5):")
    ob = get_order_book("BTC", 5)
    print(f"  Mejor bid: ${ob.mejor_bid:,.2f}")
    print(f"  Mejor ask: ${ob.mejor_ask:,.2f}")
    print(f"  Spread:    {ob.spread_pct:.4f}%")

    # Test 5: snapshot completo
    print("\n📦 Snapshot completo de ETH:")
    snap = get_snapshot_completo("ETH")
    print(f"  Activo: {snap['activo']}")
    print(f"  Precio: ${snap['precio']:,.2f}")
    print(f"  Velas 15m: {len(snap['velas_15m'])}")
    print(f"  Velas 5m: {len(snap['velas_5m'])}")
    print(f"  Spread: {snap['order_book']['spread_pct']:.4f}%")
    print("\n✅ Todos los tests OK")