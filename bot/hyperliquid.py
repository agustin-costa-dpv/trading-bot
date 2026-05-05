"""
bot/hyperliquid.py
Cliente de Hyperliquid para el frente cripto del bot.

Arquitectura dual hibrida:
- Modo DEMO: simula ordenes, calcula PnL teorico, monitoreo por codigo (check_cierre)
- Modo REAL: ejecuta orden de mercado + DOS ordenes trigger reduce-only (TP/SL nativas)
            el monitoreo solo detecta si el trigger ya cerro la posicion en el exchange

CAMBIOS v3.2 (fixes criticos):
- Formato correcto de precios para triggers (5 sig figs + szDecimals rules)
- Parsing robusto de respuestas (detecta error en statuses[0]["error"])
- get_saldo_usdc() usa spot_user_state (correcto en cuentas Unified)
- market_close usa coin=... (no name=...) — fix del bug del log
- _szdecimals_cache para no consultar meta() en cada trigger
- Logs mejorados: muestran respuesta cruda del exchange en errores
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, getcontext
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account


# ─────────────────────────────────────────────────────────────
# Carga de variables de entorno
# ─────────────────────────────────────────────────────────────
load_dotenv()

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Configuracion
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

# Parametros de trading (defaults — pueden ser sobrescritos por la senal)
TAKE_PROFIT_PCT = 1.5   # Cerrar en ganancia cuando +1.5%
STOP_LOSS_PCT = 2.0     # Cerrar en perdida cuando -2%
TAMANO_APUESTA_PCT = 0.25  # 25% del capital por apuesta (temp con CAPITAL_REAL=50, bajar a 0.04 cuando suba a $150+)


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
    """Representa una posicion abierta o cerrada."""
    activo: str
    lado: LadoPosicion
    tamano_usd: float
    precio_entrada: float
    precio_actual: float = 0.0
    precio_cierre: Optional[float] = None
    tp_precio: float = 0.0
    sl_precio: float = 0.0
    estado: EstadoPosicion = EstadoPosicion.ABIERTA
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    plataforma: str = "hyperliquid_demo"
    timestamp_apertura: str = field(default_factory=lambda: datetime.now().isoformat())
    timestamp_cierre: Optional[str] = None
    orden_id: Optional[str] = None
    razon_senal: str = ""
    tp_oid: Optional[str] = None
    sl_oid: Optional[str] = None
    db_id: Optional[int] = None
    estrategia: Optional[str] = None

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["lado"] = self.lado.value
        d["estado"] = self.estado.value
        return d


# ─────────────────────────────────────────────────────────────
# Clientes Hyperliquid (lectura + escritura) — singletons
# ─────────────────────────────────────────────────────────────

_info_client: Optional[Info] = None
_exchange_client: Optional[Exchange] = None
_szdecimals_cache: dict[str, int] = {}


def get_info_client() -> Info:
    global _info_client
    if _info_client is None:
        _info_client = Info(API_URL, skip_ws=True)
    return _info_client


def get_exchange_client() -> Exchange:
    """Cliente para ejecutar ordenes reales. Usa la API wallet (no la main)."""
    global _exchange_client
    if _exchange_client is None:
        if not API_PRIVATE_KEY:
            raise RuntimeError("HYPERLIQUID_API_PRIVATE_KEY no esta en el .env")
        if not MAIN_ADDRESS:
            raise RuntimeError("HYPERLIQUID_MAIN_ADDRESS no esta en el .env")
        wallet = Account.from_key(API_PRIVATE_KEY)
        _exchange_client = Exchange(
            wallet,
            API_URL,
            account_address=MAIN_ADDRESS,
        )
    return _exchange_client


def _get_szdecimals(activo: str) -> int:
    """Devuelve szDecimals del activo (cacheado)."""
    if activo not in _szdecimals_cache:
        info = get_info_client()
        meta = info.meta()
        for asset_info in meta.get("universe", []):
            if asset_info.get("name") == activo:
                _szdecimals_cache[activo] = asset_info.get("szDecimals", 5)
                return _szdecimals_cache[activo]
        # Fallback conservador
        _szdecimals_cache[activo] = 5
    return _szdecimals_cache[activo]


def get_precio_mark(activo: str) -> float:
    """Precio mark actual del activo."""
    info = get_info_client()
    mids = info.all_mids()
    return float(mids[activo])


def get_saldo_usdc() -> float:
    """
    Saldo USDC disponible para tradear.
    En cuentas Unified (default en app.hyperliquid.xyz), el balance esta
    en el spot clearinghouse, no en el perp. user_state() devolveria 0.
    """
    info = get_info_client()
    spot = info.spot_user_state(MAIN_ADDRESS)
    for b in spot.get("balances", []):
        if b.get("coin") == "USDC":
            total = float(b.get("total", 0))
            hold = float(b.get("hold", 0))
            return total - hold
    return 0.0


def get_posiciones_onchain() -> list[dict]:
    """Posiciones abiertas reales en Hyperliquid (formato on-chain)."""
    info = get_info_client()
    state = info.user_state(MAIN_ADDRESS)
    return state.get("assetPositions", [])


# ─────────────────────────────────────────────────────────────
# Helpers de formato (criticos para que Hyperliquid acepte ordenes)
# ─────────────────────────────────────────────────────────────

def _formatear_precio_hyperliquid(precio: float, sz_decimals: int = 5) -> float:
    """
    Hyperliquid exige reglas estrictas para precios:
      1. Maximo 5 digitos significativos
      2. Maximo (MAX_DECIMALS - szDecimals) decimales, donde MAX_DECIMALS=6 para perp

    Para BTC (szDecimals=5): max 1 decimal Y max 5 sig figs.
    Para ETH (szDecimals=4): max 2 decimales Y max 5 sig figs.

    Trunca (no redondea hacia arriba) para no superar precios que el exchange
    pudiera rechazar.
    """
    if precio <= 0:
        return precio

    getcontext().prec = 50
    d = Decimal(str(precio))

    # Cuantos digitos enteros tiene
    integer_digits = len(d.to_integral_value().__str__().lstrip("-"))

    # Decimales permitidos por regla 1 (sig figs)
    decimals_by_sigfigs = max(0, 5 - integer_digits)

    # Decimales permitidos por regla 2 (precio decimals)
    decimals_by_szdec = max(0, 6 - sz_decimals)

    # El mas restrictivo gana
    max_decimals = min(decimals_by_sigfigs, decimals_by_szdec)

    # Truncar al numero de decimales permitido
    quantize_str = "1" if max_decimals == 0 else "0." + "0" * max_decimals
    truncated = d.quantize(Decimal(quantize_str), rounding=ROUND_DOWN)

    return float(truncated)


def _parsear_respuesta_orden(result: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Parsea la respuesta de exchange.order() y devuelve (oid, error).

    Hyperliquid puede devolver:
      - {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 123}}]}}}
      - {"status": "ok", "response": {"data": {"statuses": [{"filled": {"oid": 123, ...}}]}}}
      - {"status": "ok", "response": {"data": {"statuses": [{"error": "msg"}]}}}
      - {"status": "err", "response": "mensaje de error"}

    Returns:
        (oid, error_msg). Si oid es None, error_msg explica por que.
    """
    if result.get("status") != "ok":
        return None, f"Status no-ok: {result.get('response', result)}"

    statuses = result.get("response", {}).get("data", {}).get("statuses", [])
    if not statuses:
        return None, f"Respuesta sin statuses: {result}"

    s = statuses[0]

    if "error" in s:
        return None, s["error"]

    if "resting" in s:
        oid = s["resting"].get("oid")
        if oid is None:
            return None, f"Resting sin oid: {s}"
        return str(oid), None

    if "filled" in s:
        oid = s["filled"].get("oid")
        if oid is None:
            return None, f"Filled sin oid: {s}"
        return str(oid), None

    return None, f"Status desconocido: {s}"


# ─────────────────────────────────────────────────────────────
# Ejecucion: DEMO
# ─────────────────────────────────────────────────────────────

def abrir_posicion_demo(
    activo: str,
    lado: LadoPosicion,
    razon_senal: str = "",
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
) -> Posicion:
    """
    Simula apertura de posicion. No toca blockchain.
    Usa precio mark actual de Hyperliquid como referencia realista.
    """
    precio = get_precio_mark(activo)
    tamano = CAPITAL_DEMO * TAMANO_APUESTA_PCT

    tp_pct_eff = tp_pct if tp_pct is not None else TAKE_PROFIT_PCT
    sl_pct_eff = sl_pct if sl_pct is not None else STOP_LOSS_PCT

    if lado == LadoPosicion.LONG:
        tp_precio = precio * (1 + tp_pct_eff / 100)
        sl_precio = precio * (1 - sl_pct_eff / 100)
    else:
        tp_precio = precio * (1 - tp_pct_eff / 100)
        sl_precio = precio * (1 + sl_pct_eff / 100)

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
# Ejecucion: REAL (con triggers nativas)
# ─────────────────────────────────────────────────────────────

def _colocar_triggers_real(
    activo: str,
    lado_apertura: LadoPosicion,
    sz: float,
    tp_precio: float,
    sl_precio: float,
) -> tuple[Optional[str], Optional[str], list[str]]:
    """
    Coloca DOS ordenes trigger reduce-only en Hyperliquid:
    - TP (take_profit market): se ejecuta cuando el precio cruza tp_precio
    - SL (stop_loss market):   se ejecuta cuando el precio cruza sl_precio

    Lado de las triggers: opuesto al de la apertura.
    LONG abre con is_buy=True  -> triggers cierran con is_buy=False
    SHORT abre con is_buy=False -> triggers cierran con is_buy=True

    Returns: (tp_oid, sl_oid, errores) - oids como string si OK, None si fallo.
    """
    exchange = get_exchange_client()
    is_buy_cierre = (lado_apertura == LadoPosicion.SHORT)
    errores = []

    sz_decimals = _get_szdecimals(activo)
    tp_precio_fmt = _formatear_precio_hyperliquid(tp_precio, sz_decimals)
    sl_precio_fmt = _formatear_precio_hyperliquid(sl_precio, sz_decimals)

    logger.info(
        f"  Triggers para {activo} (szDec={sz_decimals}): "
        f"TP {tp_precio:.2f} → {tp_precio_fmt}, SL {sl_precio:.2f} → {sl_precio_fmt}"
    )

    # ----- TP -----
    tp_oid = None
    try:
        result_tp = exchange.order(
            name=activo,
            is_buy=is_buy_cierre,
            sz=sz,
            limit_px=tp_precio_fmt,
            order_type={"trigger": {"isMarket": True, "triggerPx": tp_precio_fmt, "tpsl": "tp"}},
            reduce_only=True,
        )
        tp_oid, tp_err = _parsear_respuesta_orden(result_tp)
        if tp_err:
            errores.append(f"TP rechazado: {tp_err}")
            logger.error(f"  TP rechazado: {tp_err} | respuesta cruda: {result_tp}")
        else:
            logger.info(f"  TP trigger colocado: oid={tp_oid} px={tp_precio_fmt}")
    except Exception as e:
        errores.append(f"TP excepcion: {e}")
        logger.error(f"  TP excepcion: {e}", exc_info=True)

    # ----- SL -----
    sl_oid = None
    try:
        result_sl = exchange.order(
            name=activo,
            is_buy=is_buy_cierre,
            sz=sz,
            limit_px=sl_precio_fmt,
            order_type={"trigger": {"isMarket": True, "triggerPx": sl_precio_fmt, "tpsl": "sl"}},
            reduce_only=True,
        )
        sl_oid, sl_err = _parsear_respuesta_orden(result_sl)
        if sl_err:
            errores.append(f"SL rechazado: {sl_err}")
            logger.error(f"  SL rechazado: {sl_err} | respuesta cruda: {result_sl}")
        else:
            logger.info(f"  SL trigger colocado: oid={sl_oid} px={sl_precio_fmt}")
    except Exception as e:
        errores.append(f"SL excepcion: {e}")
        logger.error(f"  SL excepcion: {e}", exc_info=True)

    return tp_oid, sl_oid, errores


def _cerrar_market_real(activo: str) -> bool:
    """
    Cierra posicion con market order opuesta.
    NOTA: market_close() del SDK actual usa kwarg 'coin', no 'name'.
    """
    exchange = get_exchange_client()
    try:
        result = exchange.market_close(coin=activo)
        oid, err = _parsear_respuesta_orden(result)
        if err:
            logger.error(f"market_close {activo} fallo: {err} | cruda: {result}")
            return False
        logger.info(f"market_close {activo} OK: oid={oid}")
        return True
    except Exception as e:
        logger.error(f"market_close {activo} excepcion: {e}", exc_info=True)
        return False


def abrir_posicion_real(
    activo: str,
    lado: LadoPosicion,
    razon_senal: str = "",
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
) -> Posicion:
    """
    Ejecuta orden MARKET real en Hyperliquid + coloca triggers TP/SL nativas.
    Si los triggers fallan, cierra la posicion inmediatamente (no queda expuesta).
    """
    exchange = get_exchange_client()
    precio = get_precio_mark(activo)
    tamano_usd = CAPITAL_REAL * TAMANO_APUESTA_PCT
    sz_decimals = _get_szdecimals(activo)
    size = tamano_usd / precio
    size = round(size, sz_decimals)

    # Validacion de minimo nocional ($10 en Hyperliquid perps)
    notional = size * precio
    if notional < 10.0:
        logger.warning(
            f"Notional ${notional:.2f} < $10 minimo de Hyperliquid. "
            f"Aumentar CAPITAL_REAL o TAMANO_APUESTA_PCT. Skip real."
        )
        return Posicion(
            activo=activo, lado=lado, tamano_usd=tamano_usd,
            precio_entrada=precio, estado=EstadoPosicion.ERROR,
            plataforma="hyperliquid_real",
            razon_senal=f"Notional ${notional:.2f} debajo del minimo $10",
        )

    tp_pct_eff = tp_pct if tp_pct is not None else TAKE_PROFIT_PCT
    sl_pct_eff = sl_pct if sl_pct is not None else STOP_LOSS_PCT

    try:
        # 1) APERTURA: market order
        result = exchange.market_open(
            name=activo,
            is_buy=(lado == LadoPosicion.LONG),
            sz=size,
            slippage=0.01,
        )

        orden_id, err_open = _parsear_respuesta_orden(result)
        if err_open:
            logger.error(f"market_open {activo} fallo: {err_open} | cruda: {result}")
            return Posicion(
                activo=activo, lado=lado, tamano_usd=tamano_usd,
                precio_entrada=precio, estado=EstadoPosicion.ERROR,
                plataforma="hyperliquid_real",
                razon_senal=f"Error market_open: {err_open}",
            )

        # Extraer datos del fill (puede venir en "filled")
        statuses = result["response"]["data"]["statuses"]
        fill = statuses[0].get("filled", {})
        precio_real = float(fill.get("avgPx", precio))
        size_real = float(fill.get("totalSz", size))

        logger.info(f"REAL: {lado.value} {activo} sz={size_real} @ {precio_real} oid={orden_id}")

        # 2) CALCULAR TP/SL sobre el precio real de fill
        if lado == LadoPosicion.LONG:
            tp_precio = precio_real * (1 + tp_pct_eff / 100)
            sl_precio = precio_real * (1 - sl_pct_eff / 100)
        else:
            tp_precio = precio_real * (1 - tp_pct_eff / 100)
            sl_precio = precio_real * (1 + sl_pct_eff / 100)

        # 3) COLOCAR TRIGGERS NATIVAS
        tp_oid, sl_oid, errores = _colocar_triggers_real(
            activo=activo,
            lado_apertura=lado,
            sz=size_real,
            tp_precio=tp_precio,
            sl_precio=sl_precio,
        )

        # 4) FALLBACK CRITICO: si fallaron AMBOS triggers -> cerrar de inmediato
        # Si solo fallo uno, cancelar el otro y cerrar tambien (no queremos posicion semi-protegida)
        if errores:
            logger.error(
                f"⚠️  FALLBACK: triggers fallaron ({errores}). "
                f"Cerrando posicion inmediatamente para no quedar expuesto."
            )
            # Cancelar trigger sobreviviente si existe
            for oid_huerfana in (tp_oid, sl_oid):
                if oid_huerfana:
                    try:
                        exchange.cancel(activo, int(oid_huerfana))
                        logger.info(f"  Trigger huerfano cancelado: oid={oid_huerfana}")
                    except Exception as e_cancel:
                        logger.warning(f"  No se pudo cancelar trigger huerfano: {e_cancel}")
            # Cerrar la entrada con market opuesta
            _cerrar_market_real(activo)

            return Posicion(
                activo=activo, lado=lado, tamano_usd=tamano_usd,
                precio_entrada=precio_real, estado=EstadoPosicion.ERROR,
                plataforma="hyperliquid_real",
                orden_id=orden_id,
                razon_senal=f"Fallback: triggers fallaron, cerrada. {errores}",
            )

        # 5) OK: posicion abierta y protegida
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
            tp_oid=tp_oid,
            sl_oid=sl_oid,
            razon_senal=razon_senal,
        )

    except Exception as e:
        logger.error(f"abrir_posicion_real excepcion: {e}", exc_info=True)
        return Posicion(
            activo=activo, lado=lado, tamano_usd=tamano_usd,
            precio_entrada=precio, estado=EstadoPosicion.ERROR,
            plataforma="hyperliquid_real",
            razon_senal=f"Excepcion: {str(e)}",
        )


def ejecutar_apuesta(
    activo: str,
    lado: LadoPosicion,
    razon_senal: str = "",
    modo: str = "dual",
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
) -> dict[str, Posicion]:
    """
    Ejecuta la misma senal en los modos configurados.
    Args:
        modo: 'demo' | 'real' | 'dual'
        tp_pct/sl_pct: si vienen de la senal, override de los defaults
    """
    resultado = {}

    if modo in ("demo", "dual"):
        resultado["demo"] = abrir_posicion_demo(activo, lado, razon_senal, tp_pct, sl_pct)

    if modo in ("real", "dual"):
        resultado["real"] = abrir_posicion_real(activo, lado, razon_senal, tp_pct, sl_pct)

    return resultado


def calcular_pnl(posicion: Posicion) -> tuple[float, float]:
    """Calcula PnL en USD y porcentaje."""
    if posicion.lado == LadoPosicion.LONG:
        pnl_pct = ((posicion.precio_actual - posicion.precio_entrada) / posicion.precio_entrada) * 100
    else:
        pnl_pct = ((posicion.precio_entrada - posicion.precio_actual) / posicion.precio_entrada) * 100
    pnl_usd = (pnl_pct / 100) * posicion.tamano_usd
    return pnl_usd, pnl_pct


def check_cierre(posicion: Posicion) -> Optional[EstadoPosicion]:
    """
    DEMO ONLY: chequea si hay que cerrar la posicion por TP o SL.
    Para REAL ya no se usa (lo hace Hyperliquid via triggers).
    """
    if posicion.lado == LadoPosicion.LONG:
        if posicion.precio_actual >= posicion.tp_precio:
            return EstadoPosicion.CERRADA_GANANCIA
        if posicion.precio_actual <= posicion.sl_precio:
            return EstadoPosicion.CERRADA_PERDIDA
    else:
        if posicion.precio_actual <= posicion.tp_precio:
            return EstadoPosicion.CERRADA_GANANCIA
        if posicion.precio_actual >= posicion.sl_precio:
            return EstadoPosicion.CERRADA_PERDIDA
    return None


def cerrar_posicion_real(posicion: Posicion) -> bool:
    """
    Cierra una posicion real con market order inversa.
    Antes cancela los triggers TP/SL pendientes (si existen) para no dejar huerfanos.
    """
    if posicion.plataforma != "hyperliquid_real":
        return True

    exchange = get_exchange_client()

    # 1) Cancelar triggers pendientes
    for oid in (posicion.tp_oid, posicion.sl_oid):
        if oid:
            try:
                exchange.cancel(posicion.activo, int(oid))
                logger.info(f"  Trigger cancelado: oid={oid}")
            except Exception as e:
                logger.warning(f"  No se pudo cancelar trigger {oid}: {e}")

    # 2) Market close (usa coin=, no name=)
    return _cerrar_market_real(posicion.activo)


# ─────────────────────────────────────────────────────────────
# Monitoreo
# ─────────────────────────────────────────────────────────────

def _esta_abierta_onchain(activo: str) -> bool:
    """True si hay una posicion no-cero del activo en la cuenta real."""
    try:
        positions = get_posiciones_onchain()
        for p in positions:
            pos = p.get("position", {})
            if pos.get("coin") == activo:
                szi = float(pos.get("szi", "0"))
                if szi != 0:
                    return True
        return False
    except Exception as e:
        logger.warning(f"_esta_abierta_onchain fallo: {e}")
        # En caso de error de red, asumir que sigue abierta para no marcar cierre falso
        return True


def monitorear_posiciones(posiciones: list[Posicion]) -> list[Posicion]:
    """
    Recibe lista de OBJETOS Posicion (NO dicts on-chain).
    Para DEMO: chequea TP/SL via check_cierre y cierra si corresponde.
    Para REAL: chequea si la posicion sigue abierta on-chain. Si no, fue cerrada
              por trigger -> calcula motivo en base al precio actual y registra cierre.

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

            if pos.plataforma == "hyperliquid_real":
                # Real: verificar si el exchange ya la cerro via trigger
                if not _esta_abierta_onchain(pos.activo):
                    # Determinar motivo en base a donde esta el precio respecto a TP/SL
                    if pos.lado == LadoPosicion.LONG:
                        toco_tp = pos.precio_actual >= pos.tp_precio * 0.999
                        toco_sl = pos.precio_actual <= pos.sl_precio * 1.001
                    else:
                        toco_tp = pos.precio_actual <= pos.tp_precio * 1.001
                        toco_sl = pos.precio_actual >= pos.sl_precio * 0.999

                    if toco_tp:
                        pos.estado = EstadoPosicion.CERRADA_GANANCIA
                    elif toco_sl:
                        pos.estado = EstadoPosicion.CERRADA_PERDIDA
                    else:
                        pos.estado = EstadoPosicion.CERRADA_MANUAL

                    pos.precio_cierre = pos.precio_actual
                    pos.timestamp_cierre = datetime.now().isoformat()
                    logger.info(
                        f"REAL cerrada por trigger: {pos.activo} {pos.lado.value} "
                        f"estado={pos.estado.value} precio_cierre={pos.precio_cierre}"
                    )
            else:
                # Demo: chequear TP/SL por codigo
                estado_cierre = check_cierre(pos)
                if estado_cierre:
                    pos.estado = estado_cierre
                    pos.precio_cierre = pos.precio_actual
                    pos.timestamp_cierre = datetime.now().isoformat()
                    logger.info(
                        f"DEMO cerrada: {pos.activo} {pos.lado.value} "
                        f"estado={pos.estado.value} precio_cierre={pos.precio_cierre}"
                    )

        except Exception as e:
            logger.warning(f"Error monitoreando {pos.activo}: {e}")

        actualizadas.append(pos)

    return actualizadas
