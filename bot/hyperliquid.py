"""
bot/hyperliquid.py
Cliente de Hyperliquid para el frente cripto del bot.

Arquitectura dual:
- Modo DEMO: simula órdenes, calcula PnL teórico, registra en Supabase con plataforma='hyperliquid_demo'
- Modo REAL: ejecuta órdenes firmadas con API wallet, registra con plataforma='hyperliquid_real'

Ambos modos corren en paralelo con la misma señal.

Flujo típico:
    1. main.py recibe señal del analyst (ej: BTC SUBE, prob 72%)
    2. main.py llama a ejecutar_apuesta(señal) → abre posiciones en demo Y real
    3. main.py llama a monitorear_posiciones() cada 30seg → cierra si target/stop
    4. Todo queda registrado en Supabase con PnL real vs simulado para comparar
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account


# ─────────────────────────────────────────────────────────────
# Carga de variables de entorno (debe ir ANTES de leer os.getenv)
# ─────────────────────────────────────────────────────────────
load_dotenv()


# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────

MAIN_ADDRESS = os.getenv("HYPERLIQUID_MAIN_ADDRESS")
API_PRIVATE_KEY = os.getenv("HYPERLIQUID_API_PRIVATE_KEY")
HYPERLIQUID_MODE = os.getenv("HYPERLIQUID_MODE", "mainnet")
CAPITAL_DEMO = float(os.getenv("CAPITAL_DEMO", "100"))
CAPITAL_REAL = float(os.getenv("CAPITAL_REAL", "50"))

API_URL = (
    constants.MAINNET_API_URL if HYPERLIQUID_MODE == "mainnet"
    else constants.TESTNET_API_URL
)

# Parámetros de trading
TAKE_PROFIT_PCT = 1.5   # Cerrar en ganancia cuando +1.5%
STOP_LOSS_PCT = 2.0     # Cerrar en pérdida cuando -2%
TAMANO_APUESTA_PCT = 0.04  # 4% del capital por apuesta (dentro del rango 3-5%)


# ─────────────────────────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────────────────────────

class LadoPosicion(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class EstadoPosicion(str, Enum):
    ABIERTA = "ABIERTA"
    CERRADA_GANANCIA = "CERRADA_GANANCIA"
    CERRADA_PERDIDA = "CERRADA_PERDIDA"
    CERRADA_MANUAL = "CERRADA_MANUAL"
    ERROR = "ERROR"


@dataclass
class Posicion:
    """Representa una posición abierta o cerrada."""
    activo: str                  # BTC, ETH, SOL
    lado: LadoPosicion
    tamano_usd: float            # cuánto USD en la posición
    precio_entrada: float
    precio_actual: float = 0.0
    precio_cierre: Optional[float] = None
    tp_precio: float = 0.0       # take profit target
    sl_precio: float = 0.0       # stop loss
    estado: EstadoPosicion = EstadoPosicion.ABIERTA
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    plataforma: str = "hyperliquid_demo"  # o 'hyperliquid_real'
    timestamp_apertura: str = field(default_factory=lambda: datetime.now().isoformat())
    timestamp_cierre: Optional[str] = None
    orden_id: Optional[str] = None        # ID de orden real de Hyperliquid
    razon_senal: str = ""

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["lado"] = self.lado.value
        d["estado"] = self.estado.value
        return d


# ─────────────────────────────────────────────────────────────
# Cliente Hyperliquid (lectura)
# ─────────────────────────────────────────────────────────────

_info_client: Optional[Info] = None


def get_info_client() -> Info:
    """Cliente de solo lectura. Singleton."""
    global _info_client
    if _info_client is None:
        _info_client = Info(API_URL, skip_ws=True)
    return _info_client


def get_precio_mark(activo: str) -> float:
    """Precio mark actual de Hyperliquid (referencia para PnL)."""
    info = get_info_client()
    mids = info.all_mids()
    if activo not in mids:
        raise ValueError(f"Activo {activo} no encontrado en Hyperliquid")
    return float(mids[activo])


def get_saldo_usdc() -> float:
    """Saldo USDC disponible en la cuenta real (modo Unified Account).

    En Unified Account de Hyperliquid, el balance USDC vive en spot_user_state
    y sirve como colateral para todas las posiciones de perpetuals.
    """
    info = get_info_client()
    spot_state = info.spot_user_state(MAIN_ADDRESS)
    for balance in spot_state.get("balances", []):
        if balance["coin"] == "USDC":
            return float(balance["total"])
    return 0.0


def get_posiciones_onchain() -> list[dict]:
    """Posiciones abiertas reales en Hyperliquid."""
    info = get_info_client()
    state = info.user_state(MAIN_ADDRESS)
    return state.get("assetPositions", [])


# ─────────────────────────────────────────────────────────────
# Cliente Hyperliquid (trading real con API wallet)
# ─────────────────────────────────────────────────────────────

_exchange_client: Optional[Exchange] = None


def get_exchange_client() -> Exchange:
    """Cliente para ejecutar órdenes reales. Usa la API wallet (no la main)."""
    global _exchange_client
    if _exchange_client is None:
        if not API_PRIVATE_KEY:
            raise RuntimeError("HYPERLIQUID_API_PRIVATE_KEY no está en el .env")
        wallet = Account.from_key(API_PRIVATE_KEY)
        _exchange_client = Exchange(
            wallet,
            API_URL,
            account_address=MAIN_ADDRESS,  # la master, no la agent
        )
    return _exchange_client


# ─────────────────────────────────────────────────────────────
# Ejecución: DEMO
# ─────────────────────────────────────────────────────────────

def abrir_posicion_demo(
    activo: str,
    lado: LadoPosicion,
    razon_senal: str = "",
) -> Posicion:
    """
    Simula apertura de posición. No toca blockchain.
    Usa precio mark actual de Hyperliquid como referencia realista.
    """
    precio = get_precio_mark(activo)
    tamano = CAPITAL_DEMO * TAMANO_APUESTA_PCT

    if lado == LadoPosicion.LONG:
        tp_precio = precio * (1 + TAKE_PROFIT_PCT / 100)
        sl_precio = precio * (1 - STOP_LOSS_PCT / 100)
    else:  # SHORT
        tp_precio = precio * (1 - TAKE_PROFIT_PCT / 100)
        sl_precio = precio * (1 + STOP_LOSS_PCT / 100)

    return Posicion(
        activo=activo,
        lado=lado,
        tamano_usd=tamano,
        precio_entrada=precio,
        precio_actual=precio,
        tp_precio=tp_precio,
        sl_precio=sl_precio,
        plataforma="hyperliquid_demo",
        razon_senal=razon_senal,
    )


# ─────────────────────────────────────────────────────────────
# Ejecución: REAL
# ─────────────────────────────────────────────────────────────

def abrir_posicion_real(
    activo: str,
    lado: LadoPosicion,
    razon_senal: str = "",
) -> Posicion:
    """
    Ejecuta orden MARKET real en Hyperliquid firmada con la API wallet.
    """
    exchange = get_exchange_client()
    precio = get_precio_mark(activo)
    tamano_usd = CAPITAL_REAL * TAMANO_APUESTA_PCT
    size = tamano_usd / precio  # cantidad en unidades del activo

    # Redondeo mínimo según reglas de Hyperliquid
    # BTC: 5 decimales, ETH: 4, SOL: 2 (aproximado, ajustar con meta)
    size = round(size, 5)

    try:
        # market_open: entrada a precio de mercado con slippage tolerable
        result = exchange.market_open(
            name=activo,
            is_buy=(lado == LadoPosicion.LONG),
            sz=size,
            slippage=0.01,  # 1% max slippage
        )

        if result["status"] != "ok":
            return Posicion(
                activo=activo,
                lado=lado,
                tamano_usd=tamano_usd,
                precio_entrada=precio,
                estado=EstadoPosicion.ERROR,
                plataforma="hyperliquid_real",
                razon_senal=f"Error API: {result}",
            )

        # Parseo del fill real
        statuses = result["response"]["data"]["statuses"]
        fill = statuses[0].get("filled", {})
        precio_real = float(fill.get("avgPx", precio))
        orden_id = str(fill.get("oid", ""))

        if lado == LadoPosicion.LONG:
            tp_precio = precio_real * (1 + TAKE_PROFIT_PCT / 100)
            sl_precio = precio_real * (1 - STOP_LOSS_PCT / 100)
        else:
            tp_precio = precio_real * (1 - TAKE_PROFIT_PCT / 100)
            sl_precio = precio_real * (1 + STOP_LOSS_PCT / 100)

        return Posicion(
            activo=activo,
            lado=lado,
            tamano_usd=tamano_usd,
            precio_entrada=precio_real,
            precio_actual=precio_real,
            tp_precio=tp_precio,
            sl_precio=sl_precio,
            plataforma="hyperliquid_real",
            orden_id=orden_id,
            razon_senal=razon_senal,
        )

    except Exception as e:
        return Posicion(
            activo=activo,
            lado=lado,
            tamano_usd=tamano_usd,
            precio_entrada=precio,
            estado=EstadoPosicion.ERROR,
            plataforma="hyperliquid_real",
            razon_senal=f"Excepción: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────
# Ejecución dual (demo + real en paralelo)
# ─────────────────────────────────────────────────────────────

def ejecutar_apuesta(
    activo: str,
    lado: LadoPosicion,
    razon_senal: str = "",
    modo: str = "dual",
) -> dict[str, Posicion]:
    """
    Ejecuta la misma señal en los modos configurados.

    Args:
        modo: 'demo' | 'real' | 'dual'
    """
    resultado = {}

    if modo in ("demo", "dual"):
        resultado["demo"] = abrir_posicion_demo(activo, lado, razon_senal)

    if modo in ("real", "dual"):
        resultado["real"] = abrir_posicion_real(activo, lado, razon_senal)

    return resultado


# ─────────────────────────────────────────────────────────────
# Monitoreo y cierre
# ─────────────────────────────────────────────────────────────

def calcular_pnl(posicion: Posicion) -> tuple[float, float]:
    """
    Calcula PnL en USD y % dado el precio actual.
    Returns: (pnl_usd, pnl_pct)
    """
    if posicion.lado == LadoPosicion.LONG:
        pct = (posicion.precio_actual - posicion.precio_entrada) / posicion.precio_entrada
    else:  # SHORT
        pct = (posicion.precio_entrada - posicion.precio_actual) / posicion.precio_entrada

    pnl_usd = posicion.tamano_usd * pct
    pnl_pct = pct * 100
    return pnl_usd, pnl_pct


def check_cierre(posicion: Posicion) -> Optional[EstadoPosicion]:
    """
    Chequea si hay que cerrar la posición por TP o SL.
    Returns: EstadoPosicion si hay que cerrar, None si sigue abierta.
    """
    if posicion.lado == LadoPosicion.LONG:
        if posicion.precio_actual >= posicion.tp_precio:
            return EstadoPosicion.CERRADA_GANANCIA
        if posicion.precio_actual <= posicion.sl_precio:
            return EstadoPosicion.CERRADA_PERDIDA
    else:  # SHORT
        if posicion.precio_actual <= posicion.tp_precio:
            return EstadoPosicion.CERRADA_GANANCIA
        if posicion.precio_actual >= posicion.sl_precio:
            return EstadoPosicion.CERRADA_PERDIDA

    return None


def cerrar_posicion_real(posicion: Posicion) -> bool:
    """
    Cierra una posición real en Hyperliquid con market order inversa.
    """
    if posicion.plataforma != "hyperliquid_real":
        return True  # demo no requiere cierre onchain

    exchange = get_exchange_client()
    size = posicion.tamano_usd / posicion.precio_entrada
    size = round(size, 5)

    try:
        result = exchange.market_close(name=posicion.activo)
        return result.get("status") == "ok"
    except Exception as e:
        print(f"⚠️  Error cerrando posición real {posicion.activo}: {e}")
        return False


def monitorear_posiciones(posiciones: list[Posicion]) -> list[Posicion]:
    """
    Para cada posición abierta:
    1. Actualiza precio actual
    2. Calcula PnL
    3. Chequea si hay que cerrar (TP / SL)
    4. Si sí, cierra (real o simulado)

    Returns: lista de posiciones actualizadas.
    """
    actualizadas = []

    for pos in posiciones:
        if pos.estado != EstadoPosicion.ABIERTA:
            actualizadas.append(pos)
            continue

        try:
            pos.precio_actual = get_precio_mark(pos.activo)
            pos.pnl_usd, pos.pnl_pct = calcular_pnl(pos)

            estado_cierre = check_cierre(pos)
            if estado_cierre:
                # Cerrar real si aplica
                if pos.plataforma == "hyperliquid_real":
                    cerrar_posicion_real(pos)

                pos.estado = estado_cierre
                pos.precio_cierre = pos.precio_actual
                pos.timestamp_cierre = datetime.now().isoformat()

        except Exception as e:
            print(f"⚠️  Error monitoreando {pos.activo}: {e}")

        actualizadas.append(pos)

    return actualizadas


# ─────────────────────────────────────────────────────────────
# Testing — python -m bot.hyperliquid
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Cliente Hyperliquid")
    print(f"Modo: {HYPERLIQUID_MODE} | Main: {MAIN_ADDRESS[:10]}...")
    print("=" * 60)

    # Test 1: precios mark
    print("\n📊 Precios mark actuales:")
    for activo in ["BTC", "ETH", "SOL"]:
        try:
            precio = get_precio_mark(activo)
            print(f"  {activo}: ${precio:,.2f}")
        except Exception as e:
            print(f"  {activo}: ERROR {e}")

    # Test 2: saldo
    print(f"\n💰 Saldo USDC en cuenta: ${get_saldo_usdc():.2f}")

    # Test 3: posiciones onchain
    posiciones = get_posiciones_onchain()
    print(f"📂 Posiciones abiertas onchain: {len(posiciones)}")

    # Test 4: simulación demo (sin tocar blockchain)
    print("\n🎮 Test apertura DEMO (sin tocar cuenta real):")
    demo_pos = abrir_posicion_demo(
        activo="BTC",
        lado=LadoPosicion.LONG,
        razon_senal="test manual",
    )
    print(f"  Activo: {demo_pos.activo}")
    print(f"  Lado: {demo_pos.lado.value}")
    print(f"  Tamaño: ${demo_pos.tamano_usd:.2f}")
    print(f"  Precio entrada: ${demo_pos.precio_entrada:,.2f}")
    print(f"  Take profit: ${demo_pos.tp_precio:,.2f}")
    print(f"  Stop loss: ${demo_pos.sl_precio:,.2f}")

    # Test 5: cálculo de PnL con movimiento simulado
    print("\n📈 Simulación de movimiento +0.5%:")
    demo_pos.precio_actual = demo_pos.precio_entrada * 1.005
    pnl_usd, pnl_pct = calcular_pnl(demo_pos)
    print(f"  Precio actual: ${demo_pos.precio_actual:,.2f}")
    print(f"  PnL: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)")
    estado = check_cierre(demo_pos)
    print(f"  ¿Cerrar? {'SÍ ('+estado.value+')' if estado else 'NO, sigue abierta'}")

    print("\n✅ Tests de lectura OK. Trading real no se ejecuta en este test.")
    print("   Para ejecutar real, cargá USDC y llamá ejecutar_apuesta() desde main.py")