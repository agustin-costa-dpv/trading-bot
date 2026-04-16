"""
bot/binance.py — v3
Spot desde Binance (mirror). Derivados desde OKX (Binance Futures bloquea
desde muchas IPs incluyendo Render/Codespaces).
"""

import requests
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


BINANCE_BASE = "https://data-api.binance.vision"
OKX_BASE = "https://www.okx.com"

SIMBOLOS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}

OKX_SWAP = {
    "BTC": "BTC-USDT-SWAP",
    "ETH": "ETH-USDT-SWAP",
    "SOL": "SOL-USDT-SWAP",
}

OKX_CCY = {"BTC": "BTC", "ETH": "ETH", "SOL": "SOL"}

Intervalo = Literal["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]


@dataclass
class Vela:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open, "high": self.high, "low": self.low,
            "close": self.close, "volume": self.volume,
        }


@dataclass
class Stats24h:
    simbolo: str
    precio_actual: float
    cambio_pct_24h: float
    volumen_24h: float
    volumen_quote_24h: float
    high_24h: float
    low_24h: float


@dataclass
class OrderBook:
    simbolo: str
    bids: list[tuple[float, float]] = field(default_factory=list)
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


def _resolver_simbolo(activo: str) -> str:
    activo = activo.upper()
    if activo in SIMBOLOS:
        return SIMBOLOS[activo]
    if activo.endswith("USDT"):
        return activo
    raise ValueError(f"Activo no soportado: {activo}")


def _request(base: str, endpoint: str, params: dict | None = None) -> dict | list:
    url = f"{base}{endpoint}"
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Timeout: {endpoint}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP {r.status_code}: {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Red: {e}")


# ─────────────────────────────────────────────────────────────
# Spot (Binance)
# ─────────────────────────────────────────────────────────────

def get_precio(activo: str) -> float:
    simbolo = _resolver_simbolo(activo)
    data = _request(BINANCE_BASE, "/api/v3/ticker/price", {"symbol": simbolo})
    return float(data["price"])


def get_precios_todos() -> dict[str, float]:
    return {nombre: get_precio(nombre) for nombre in SIMBOLOS.keys()}


def get_stats_24h(activo: str) -> Stats24h:
    simbolo = _resolver_simbolo(activo)
    data = _request(BINANCE_BASE, "/api/v3/ticker/24hr", {"symbol": simbolo})
    return Stats24h(
        simbolo=simbolo,
        precio_actual=float(data["lastPrice"]),
        cambio_pct_24h=float(data["priceChangePercent"]),
        volumen_24h=float(data["volume"]),
        volumen_quote_24h=float(data["quoteVolume"]),
        high_24h=float(data["highPrice"]),
        low_24h=float(data["lowPrice"]),
    )


def get_velas(activo: str, intervalo: Intervalo = "15m", limite: int = 100) -> list[Vela]:
    simbolo = _resolver_simbolo(activo)
    data = _request(BINANCE_BASE, "/api/v3/klines", {
        "symbol": simbolo, "interval": intervalo, "limit": min(limite, 1000),
    })
    return [
        Vela(
            timestamp=datetime.fromtimestamp(v[0] / 1000),
            open=float(v[1]), high=float(v[2]), low=float(v[3]),
            close=float(v[4]), volume=float(v[5]),
        )
        for v in data
    ]


def get_order_book(activo: str, profundidad: int = 20) -> OrderBook:
    simbolo = _resolver_simbolo(activo)
    data = _request(BINANCE_BASE, "/api/v3/depth", {"symbol": simbolo, "limit": profundidad})
    return OrderBook(
        simbolo=simbolo,
        bids=[(float(p), float(q)) for p, q in data["bids"]],
        asks=[(float(p), float(q)) for p, q in data["asks"]],
    )


# ─────────────────────────────────────────────────────────────
# Derivados (OKX)
# ─────────────────────────────────────────────────────────────

def get_funding_rate(activo: str) -> float:
    activo = activo.upper()
    inst_id = OKX_SWAP.get(activo)
    if not inst_id:
        return 0.0
    try:
        data = _request(OKX_BASE, "/api/v5/public/funding-rate", {"instId": inst_id})
        if data.get("code") == "0" and data.get("data"):
            return float(data["data"][0].get("fundingRate", 0.0))
        return 0.0
    except (RuntimeError, KeyError, ValueError, IndexError):
        return 0.0


def get_long_short_ratio(activo: str, periodo: str = "5m") -> float:
    activo = activo.upper()
    ccy = OKX_CCY.get(activo)
    if not ccy:
        return 1.0
    try:
        data = _request(OKX_BASE, "/api/v5/rubik/stat/contracts/long-short-account-ratio", {
            "ccy": ccy, "period": periodo,
        })
        if data.get("code") == "0" and data.get("data"):
            return float(data["data"][0][1])
        return 1.0
    except (RuntimeError, KeyError, ValueError, IndexError):
        return 1.0


# ─────────────────────────────────────────────────────────────
# Snapshot completo
# ─────────────────────────────────────────────────────────────

def get_snapshot_completo(activo: str) -> dict:
    stats = get_stats_24h(activo)
    velas_15m = get_velas(activo, "15m", 100)
    velas_5m = get_velas(activo, "5m", 60)
    order_book = get_order_book(activo, 20)
    funding = get_funding_rate(activo)
    ls_ratio = get_long_short_ratio(activo, "5m")

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
        "funding_rate": funding,
        "long_short_ratio": ls_ratio,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Binance spot + OKX derivados")
    print("=" * 60)

    print("\n📊 Precios (Binance):")
    for nombre, precio in get_precios_todos().items():
        print(f"  {nombre}: ${precio:,.2f}")

    print("\n💸 Derivados (OKX):")
    for activo in ["BTC", "ETH", "SOL"]:
        f = get_funding_rate(activo)
        ls = get_long_short_ratio(activo)
        print(f"  {activo}: funding={f:+.6f} | L/S={ls:.3f}")

    print("\n📦 Snapshot BTC:")
    snap = get_snapshot_completo("BTC")
    print(f"  Precio: ${snap['precio']:,.2f}")
    print(f"  Velas: 15m={len(snap['velas_15m'])} 5m={len(snap['velas_5m'])}")
    print(f"  Funding: {snap['funding_rate']:+.6f}")
    print(f"  L/S: {snap['long_short_ratio']:.3f}")
