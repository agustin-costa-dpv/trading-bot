"""
hyperliquid_client.py

Cliente robusto para Hyperliquid con:
- Pre-validacion local (min notional, decimales, tick size)
- Parsing estricto de respuestas (no mas oid=None silencioso)
- Apertura two-phase con rollback automatico
- Reconciliacion DB <-> exchange
- Sizing DUAL coherente

Diseñado para reemplazar la capa actual de bot.hyperliquid.
Compatible con hyperliquid-python-sdk >= 0.9.x
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

log = logging.getLogger("bot.hyperliquid")


# ─────────────────────────────────────────────────────────────────────
# Excepciones — jerarquia clara para que el caller pueda discriminar
# ─────────────────────────────────────────────────────────────────────

class HyperliquidError(Exception):
    """Base para todos los errores del cliente."""


class PreflightError(HyperliquidError):
    """Validacion local fallo. No se llamo al exchange. Es seguro reintentar
    con otros parametros."""


class OrderRejected(HyperliquidError):
    """El exchange acepto el request pero rechazo la orden (min size,
    margen insuficiente, etc.). NO se abrio nada."""


class ExchangeError(HyperliquidError):
    """Respuesta inesperada del exchange. Estado incierto — verificar."""


class RollbackFailed(HyperliquidError):
    """CRITICO: rollback no pudo cerrar posicion. Posible exposicion real."""


# ─────────────────────────────────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AssetSpec:
    """Specs de un asset, cargadas de info.meta() al inicio."""
    coin: str
    sz_decimals: int          # decimales permitidos en size
    max_leverage: int
    min_notional_usd: float   # Hyperliquid: $10 para perps

    def round_sz(self, sz: float) -> float:
        """Trunca size a los decimales permitidos (truncar, no redondear,
        para no superar el balance)."""
        factor = 10 ** self.sz_decimals
        return math.floor(sz * factor) / factor


class OrderStatus(str, Enum):
    RESTING = "resting"   # en el book, esperando fill
    FILLED = "filled"     # ejecutada (total o parcial)


@dataclass(frozen=True)
class OrderResult:
    oid: int
    status: OrderStatus
    fill_px: float | None = None     # solo si FILLED
    fill_sz: float | None = None     # solo si FILLED


@dataclass(frozen=True)
class TriggerResult:
    oid: int
    trigger_px: float


@dataclass(frozen=True)
class OpenPositionResult:
    """Resultado de open_position_protected. Todo o nada."""
    coin: str
    is_buy: bool
    entry_oid: int
    entry_px: float
    entry_sz: float
    tp_oid: int
    sl_oid: int
    tp_px: float
    sl_px: float


# ─────────────────────────────────────────────────────────────────────
# Cliente principal
# ─────────────────────────────────────────────────────────────────────

class HyperliquidClient:
    """
    Wrapper sobre la SDK oficial. Cachea specs al iniciar.
    Una instancia por entorno (mainnet/testnet).
    """

    # Buffer sobre el min notional del exchange para no quedar al limite
    MIN_NOTIONAL_BUFFER = 1.5  # 50% extra

    def __init__(
        self,
        exchange: Exchange,
        info: Info,
        address: str,
    ) -> None:
        self._exchange = exchange
        self._info = info
        self._address = address
        self._specs: dict[str, AssetSpec] = {}
        self._load_specs()

    # ─── Inicializacion ──────────────────────────────────────────────

    def _load_specs(self) -> None:
        """Carga specs de todos los assets desde info.meta(). Llamar al inicio."""
        meta = self._info.meta()
        # Hyperliquid mainnet perps: minNotional = $10
        # Lo dejamos parametrizable por si cambia o difiere por asset
        DEFAULT_MIN_NOTIONAL = 10.0

        for asset in meta["universe"]:
            coin = asset["name"]
            self._specs[coin] = AssetSpec(
                coin=coin,
                sz_decimals=asset["szDecimals"],
                max_leverage=asset["maxLeverage"],
                min_notional_usd=DEFAULT_MIN_NOTIONAL,
            )
        log.info(f"Specs cargadas para {len(self._specs)} assets")

    def get_spec(self, coin: str) -> AssetSpec:
        if coin not in self._specs:
            raise PreflightError(f"Asset desconocido: {coin}")
        return self._specs[coin]

    def min_size_for_notional(self, coin: str, px: float, target_usd: float) -> float:
        """
        Devuelve el size minimo valido para alcanzar target_usd al precio dado,
        o 0.0 si target_usd esta por debajo del minimo del exchange.
        """
        spec = self.get_spec(coin)
        floor_usd = spec.min_notional_usd * self.MIN_NOTIONAL_BUFFER
        effective_usd = max(target_usd, floor_usd) if target_usd > 0 else 0.0
        if effective_usd == 0.0:
            return 0.0
        sz = effective_usd / px
        return spec.round_sz(sz)

    # ─── Pre-validacion local ────────────────────────────────────────

    def validate_order(self, coin: str, sz: float, px: float) -> None:
        """Levanta PreflightError si la orden no va a pasar el exchange.
        Gratis y sin round-trip."""
        spec = self.get_spec(coin)

        if sz <= 0:
            raise PreflightError(f"{coin}: sz invalido {sz}")
        if px <= 0:
            raise PreflightError(f"{coin}: px invalido {px}")

        # Decimales de size
        rounded = spec.round_sz(sz)
        if rounded != sz:
            raise PreflightError(
                f"{coin}: sz {sz} excede {spec.sz_decimals} decimales "
                f"(usar {rounded})"
            )

        # Min notional con buffer
        notional = sz * px
        floor = spec.min_notional_usd * self.MIN_NOTIONAL_BUFFER
        if notional < floor:
            raise PreflightError(
                f"{coin}: notional ${notional:.2f} < piso ${floor:.2f} "
                f"(min exchange ${spec.min_notional_usd:.2f} + buffer "
                f"{self.MIN_NOTIONAL_BUFFER}x)"
            )

    # ─── Parsing estricto de respuestas ──────────────────────────────

    @staticmethod
    def _parse_order_response(resp: dict[str, Any]) -> OrderResult:
        """
        Parsea respuesta de exchange.order(). Levanta excepcion si:
        - status != "ok"
        - statuses vacios o con error
        - oid faltante
        """
        if resp.get("status") != "ok":
            raise ExchangeError(f"Status no-ok: {resp}")

        try:
            statuses = resp["response"]["data"]["statuses"]
        except (KeyError, TypeError) as e:
            raise ExchangeError(f"Respuesta malformada: {resp}") from e

        if not statuses:
            raise ExchangeError(f"Sin statuses en respuesta: {resp}")

        s = statuses[0]

        if "error" in s:
            raise OrderRejected(s["error"])

        if "resting" in s:
            oid = s["resting"].get("oid")
            if oid is None:
                raise ExchangeError(f"Resting sin oid: {s}")
            return OrderResult(oid=int(oid), status=OrderStatus.RESTING)

        if "filled" in s:
            f = s["filled"]
            oid = f.get("oid")
            if oid is None:
                raise ExchangeError(f"Filled sin oid: {s}")
            return OrderResult(
                oid=int(oid),
                status=OrderStatus.FILLED,
                fill_px=float(f["avgPx"]),
                fill_sz=float(f["totalSz"]),
            )

        raise ExchangeError(f"Status desconocido: {s}")

    # ─── Operaciones primitivas ──────────────────────────────────────

    def place_market_order(
        self,
        coin: str,
        is_buy: bool,
        sz: float,
        slippage: float = 0.01,
    ) -> OrderResult:
        """
        Orden market con slippage maximo (Hyperliquid usa IOC con limit
        agresivo internamente cuando llamas a market_open).

        slippage: 0.01 = 1% maximo de slippage tolerado.
        """
        # Necesitamos un precio de referencia para validar notional.
        ref_px = self._get_mid_price(coin)
        self.validate_order(coin, sz, ref_px)

        log.info(f"MARKET {('BUY' if is_buy else 'SELL')} {coin} sz={sz}")

        # market_open es la API correcta para market orders en Hyperliquid SDK
        # Firma: market_open(name, is_buy, sz, px=None, slippage=0.05)
        resp = self._exchange.market_open(coin, is_buy, sz, None, slippage)
        result = self._parse_order_response(resp)
        log.info(
            f"  → oid={result.oid} status={result.status.value}"
            + (f" fill_px={result.fill_px}" if result.fill_px else "")
        )
        return result

    def place_trigger_order(
        self,
        coin: str,
        is_buy: bool,
        sz: float,
        trigger_px: float,
        kind: Literal["tp", "sl"],
    ) -> TriggerResult:
        """
        Coloca una orden TP o SL como trigger order.
        Para una posicion LONG: tp es is_buy=False (cierre vendiendo arriba)
                                sl es is_buy=False (cierre vendiendo abajo)
        Para una posicion SHORT: invertido.
        """
        self.validate_order(coin, sz, trigger_px)

        # Hyperliquid trigger order spec
        order_type = {
            "trigger": {
                "triggerPx": str(trigger_px),
                "isMarket": True,         # se ejecuta a market al disparar
                "tpsl": kind,             # "tp" o "sl"
            }
        }

        log.info(
            f"TRIGGER {kind.upper()} {coin} "
            f"{'BUY' if is_buy else 'SELL'} sz={sz} @ {trigger_px}"
        )

        resp = self._exchange.order(
            coin,
            is_buy,
            sz,
            trigger_px,           # limit_px (se usa como referencia)
            order_type,
            reduce_only=True,     # CLAVE: solo reduce, nunca abre nuevo
        )
        result = self._parse_order_response(resp)
        log.info(f"  → trigger oid={result.oid}")
        return TriggerResult(oid=result.oid, trigger_px=trigger_px)

    def cancel_order(self, coin: str, oid: int) -> bool:
        """Cancela una orden. True si se cancelo, False si ya no existia."""
        try:
            resp = self._exchange.cancel(coin, oid)
            if resp.get("status") == "ok":
                log.info(f"Cancelada oid={oid}")
                return True
            log.warning(f"Cancel oid={oid} no-ok: {resp}")
            return False
        except Exception as e:
            log.warning(f"Cancel oid={oid} fallo: {e}")
            return False

    # ─── Apertura protegida (two-phase con rollback) ─────────────────

    def open_position_protected(
        self,
        coin: str,
        is_buy: bool,
        size_usd: float,
        tp_px: float,
        sl_px: float,
        slippage: float = 0.01,
    ) -> OpenPositionResult:
        """
        Abre una posicion con TP y SL en una transaccion logica.
        Si TP o SL fallan, cierra la entrada (rollback) y levanta excepcion.

        Pasos:
          1. Validar pre-condiciones (sizing, asset, etc.)
          2. Abrir entrada market
          3. Colocar TP
          4. Colocar SL  (si falla, cancelar TP y rollback)
          5. Devolver resultado

        Levanta:
          PreflightError    → no se hizo nada
          OrderRejected     → entrada rechazada, no se hizo nada
          ExchangeError     → estado incierto, revisar manualmente
          RollbackFailed    → CRITICO, posicion abierta sin proteccion
        """
        # 1. Validacion previa
        ref_px = self._get_mid_price(coin)
        sz = self.min_size_for_notional(coin, ref_px, size_usd)
        if sz == 0.0:
            raise PreflightError(
                f"{coin}: size_usd ${size_usd:.2f} < piso minimo, skip"
            )

        # Sanity check direcciones de TP/SL
        if is_buy:
            if not (sl_px < ref_px < tp_px):
                raise PreflightError(
                    f"LONG {coin}: SL/TP invalidos "
                    f"(SL={sl_px} px={ref_px} TP={tp_px})"
                )
        else:
            if not (tp_px < ref_px < sl_px):
                raise PreflightError(
                    f"SHORT {coin}: SL/TP invalidos "
                    f"(TP={tp_px} px={ref_px} SL={sl_px})"
                )

        # 2. Entrada
        log.info(f"━━ Apertura {coin} {'LONG' if is_buy else 'SHORT'} ━━")
        entry = self.place_market_order(coin, is_buy, sz, slippage)

        if entry.status != OrderStatus.FILLED:
            # No se ejecuto, cancelar y abortar
            self.cancel_order(coin, entry.oid)
            raise OrderRejected(
                f"Entrada {coin} no se llenó (status={entry.status.value})"
            )

        filled_sz = entry.fill_sz or sz
        entry_px = entry.fill_px or ref_px

        # 3. TP
        try:
            tp = self.place_trigger_order(
                coin, not is_buy, filled_sz, tp_px, "tp"
            )
        except (OrderRejected, ExchangeError) as e:
            log.error(f"TP fallo: {e}. Rollback de entrada.")
            self._rollback(coin, is_buy, filled_sz)
            raise

        # 4. SL
        try:
            sl = self.place_trigger_order(
                coin, not is_buy, filled_sz, sl_px, "sl"
            )
        except (OrderRejected, ExchangeError) as e:
            log.error(f"SL fallo: {e}. Cancelando TP y rollback.")
            self.cancel_order(coin, tp.oid)
            self._rollback(coin, is_buy, filled_sz)
            raise

        log.info(
            f"━━ Posicion abierta: {coin} sz={filled_sz} @ {entry_px} "
            f"TP={tp_px} SL={sl_px} ━━"
        )
        return OpenPositionResult(
            coin=coin,
            is_buy=is_buy,
            entry_oid=entry.oid,
            entry_px=entry_px,
            entry_sz=filled_sz,
            tp_oid=tp.oid,
            sl_oid=sl.oid,
            tp_px=tp_px,
            sl_px=sl_px,
        )

    def _rollback(self, coin: str, was_buy: bool, sz: float) -> None:
        """
        Cierra una posicion abierta enviando una orden market opuesta.
        NO usa market_close() (cuya firma cambia entre versiones del SDK).
        Si esto falla, levanta RollbackFailed — el caller debe alertar.
        """
        log.warning(f"ROLLBACK {coin} sz={sz} (cerrando con market opuesta)")
        try:
            resp = self._exchange.market_open(
                coin,
                not was_buy,        # direccion opuesta
                sz,
                None,
                0.05,               # slippage tolerante en rollback
            )
            self._parse_order_response(resp)
            log.info("Rollback exitoso")
        except Exception as e:
            log.critical(
                f"ROLLBACK FALLO {coin} sz={sz}: {e}. POSIBLE EXPOSICION."
            )
            raise RollbackFailed(
                f"{coin} sz={sz} quedo posiblemente abierto: {e}"
            ) from e

    # ─── Cierre limpio ───────────────────────────────────────────────

    def close_position(
        self,
        coin: str,
        is_buy_was: bool,
        sz: float,
        tp_oid: int | None = None,
        sl_oid: int | None = None,
    ) -> OrderResult:
        """Cierra una posicion abierta. Cancela TP/SL pendientes primero."""
        if tp_oid is not None:
            self.cancel_order(coin, tp_oid)
        if sl_oid is not None:
            self.cancel_order(coin, sl_oid)

        log.info(f"Cerrando {coin} sz={sz}")
        return self.place_market_order(coin, not is_buy_was, sz, slippage=0.05)

    # ─── Reconciliacion ──────────────────────────────────────────────

    def get_open_positions(self) -> dict[str, dict[str, Any]]:
        """Estado real de posiciones en el exchange. Key = coin."""
        state = self._info.user_state(self._address)
        result = {}
        for ap in state.get("assetPositions", []):
            pos = ap.get("position", {})
            coin = pos.get("coin")
            sz = float(pos.get("szi", 0))
            if coin and sz != 0:
                result[coin] = {
                    "sz": abs(sz),
                    "is_buy": sz > 0,
                    "entry_px": float(pos.get("entryPx", 0)),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                }
        return result

    def get_open_orders(self) -> list[dict[str, Any]]:
        """Ordenes pendientes (incluye triggers no disparados)."""
        return self._info.open_orders(self._address)

    def reconcile(self, db_open_positions: list[Any]) -> dict[str, Any]:
        """
        Compara DB con exchange. Devuelve dict con discrepancias.

        db_open_positions: lista de objetos con .coin y .estado
        Retorna:
          {
            "ghosts_in_db": [...],    # DB dice abierta, exchange no
            "orphans_in_ex": [...],   # exchange tiene, DB no
            "matched": [...]
          }
        """
        ex_positions = self.get_open_positions()
        db_by_coin = {p.coin: p for p in db_open_positions}

        ghosts = [p for p in db_open_positions if p.coin not in ex_positions]
        orphans = [
            {"coin": c, **data}
            for c, data in ex_positions.items()
            if c not in db_by_coin
        ]
        matched = [p for p in db_open_positions if p.coin in ex_positions]

        if ghosts:
            log.warning(f"Reconciliacion: {len(ghosts)} fantasmas en DB")
        if orphans:
            log.warning(f"Reconciliacion: {len(orphans)} huerfanos en exchange")

        return {
            "ghosts_in_db": ghosts,
            "orphans_in_ex": orphans,
            "matched": matched,
        }

    # ─── Helpers ─────────────────────────────────────────────────────

    def _get_mid_price(self, coin: str) -> float:
        """Mid price actual del orderbook."""
        mids = self._info.all_mids()
        if coin not in mids:
            raise ExchangeError(f"No hay mid price para {coin}")
        return float(mids[coin])


# ─────────────────────────────────────────────────────────────────────
# Sizing DUAL coherente
# ─────────────────────────────────────────────────────────────────────

def calc_dual_sizing(
    client: HyperliquidClient,
    coin: str,
    px: float,
    demo_notional_usd: float,
    real_ratio: float = 0.5,
) -> tuple[float, float]:
    """
    Devuelve (demo_usd, real_usd) coherentes.

    Regla:
      - demo siempre usa demo_notional_usd
      - real = demo * real_ratio si supera el piso del exchange
      - real = 0 (skip) si no llega al piso

    Esto evita el problema de "real_usd demasiado chico, todas las
    ordenes rechazadas silenciosamente".
    """
    spec = client.get_spec(coin)
    floor = spec.min_notional_usd * client.MIN_NOTIONAL_BUFFER

    real_target = demo_notional_usd * real_ratio
    if real_target < floor:
        log.warning(
            f"Real sizing ${real_target:.2f} < piso ${floor:.2f} para {coin}, "
            f"skip real (demo sigue)"
        )
        return demo_notional_usd, 0.0

    return demo_notional_usd, real_target
