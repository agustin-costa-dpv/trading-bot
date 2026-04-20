"""
main.py — Orquestador principal del bot de trading (v3)

Alineado con analyst.py v3 (configuración validada por backtest):
  - BTC: TREND_FOLLOWING
  - ETH: TREND_FOLLOWING + ARBITRAJE
  - SOL: fuera del universo
  - TP/SL/horizonte por estrategia (vienen en la Senal)

Dos loops:
  - Loop lento:  cada 30 minutos → análisis + señales
  - Loop rápido: cada 30 segundos → monitoreo de posiciones
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("main")

_ciclos_sin_senal_deportiva = 0


# ─── Módulos del proyecto ────────────────────────────────────────────────────
from agent.sessions import get_sesion_actual, puede_operar_ahora
from agent.analyst import analizar, Direccion
from agent.analyst_sports import analizar_todos_los_partidos
from bot.hyperliquid import (
    get_precio_mark,
    get_posiciones_onchain,
    ejecutar_apuesta,
    monitorear_posiciones,
    abrir_posicion_demo,
    LadoPosicion,
)
from bot.risk import verificar_stop_loss, obtener_estado_capital
from database.models import (
    obtener_capital,
    crear_capital_inicial,
    registrar_apuesta,
    registrar_senal,
)

# ─── Config ──────────────────────────────────────────────────────────────────
MODE           = os.getenv("MODE", "demo")
CAPITAL_DEMO   = float(os.getenv("CAPITAL_DEMO", 100))
CAPITAL_REAL   = float(os.getenv("CAPITAL_REAL", 50))
CAPITAL_ACTUAL = CAPITAL_DEMO if MODE == "demo" else CAPITAL_REAL

# v3: SOL descartado por backtest
PARES_CRIPTO = ["BTC", "ETH"]

INTERVALO_LENTO   = 30 * 60   # 30 min
INTERVALO_RAPIDO  = 30        # 30 seg

# Umbrales validados por backtest
PROBABILIDAD_MIN_EJECUCION = float(os.getenv("PROBABILIDAD_MIN_EJECUCION", 0.58))

TAMANO_APUESTA = 0.04    # 4% del capital por trade
LEVERAGE       = 1


# ─── Inicialización ──────────────────────────────────────────────────────────

def inicializar_capital():
    for modo in ("demo", "real"):
        capital = obtener_capital(modo)
        if not capital:
            saldo = CAPITAL_DEMO if modo == "demo" else CAPITAL_REAL
            crear_capital_inicial(modo=modo, saldo=saldo)
            logger.info(f"Capital inicial creado — modo={modo} saldo=${saldo}")
        else:
            logger.info(f"Capital existente — modo={modo} saldo=${capital.get('saldo', '?')}")


# ─── Loop lento ──────────────────────────────────────────────────────────────

async def ciclo_analisis():
    logger.info("═" * 60)
    logger.info(f"CICLO DE ANÁLISIS — {datetime.now().strftime('%H:%M:%S')} — modo={MODE.upper()}")

    if not puede_operar_ahora():
        logger.info("⏸  Sesión con baja liquidez — análisis omitido")
        return

    sesion = get_sesion_actual()
    prioridad = sesion.prioridad.value if hasattr(sesion, "prioridad") else sesion.get("prioridad", "EVITAR")
    nombre_sesion = sesion.nombre if hasattr(sesion, "nombre") else sesion.get("nombre", "desconocida")
    logger.info(f"Sesión actual: {nombre_sesion} [{prioridad}]")

    estado = obtener_estado_capital(MODE)
    if estado:
        bloqueado, razon = verificar_stop_loss(estado)
        if bloqueado:
            logger.warning(f"🛑 {razon} — operaciones bloqueadas")
            return

    capital = obtener_capital(MODE)

    if prioridad in ("ALTA", "MEDIA"):
        await _ciclo_cripto(prioridad, capital)

    if prioridad == "ALTA":
        await _ciclo_deportivo()


async def _ciclo_cripto(prioridad: str, capital):
    """Analiza pares cripto con analyst v3 y ejecuta si hay señal validada."""
    logger.info("── Análisis CRIPTO ──────────────────────────────────")

    for par in PARES_CRIPTO:
        try:
            logger.info(f"Analizando {par}...")
            senal = analizar(par)

            if not senal:
                logger.info(f"{par}: sin señal")
                continue

            # analyst v3 devuelve objeto Senal con:
            # direccion (SUBE/BAJA), probabilidad, confianza, razon, estrategia,
            # regimen, tp_pct, sl_pct, horizonte_min
            precio = get_precio_mark(par)

            logger.info(
                f"{par}: {senal.direccion.value} | "
                f"estrategia={senal.estrategia.value} | "
                f"régimen={senal.regimen} | "
                f"prob={senal.probabilidad:.2f} [{senal.confianza}] | "
                f"precio={precio} | TP={senal.tp_pct}% SL={senal.sl_pct}%"
            )
            logger.info(f"  Razón: {senal.razon}")

            # Registrar señal en DB
            registrar_senal(
                fuente=f"analyst_cripto_{par}",
                contenido=senal.razon,
                score=senal.probabilidad,
                accion=senal.direccion.value,
                mercado_id=par,
            )

            # Ejecutar solo si prob >= umbral Y sesión ALTA
            if prioridad == "ALTA" and senal.probabilidad >= PROBABILIDAD_MIN_EJECUCION:
                monto = _calcular_monto(capital)
                if monto <= 0:
                    logger.warning(f"{par}: capital insuficiente")
                    continue
                _ejecutar_trade(par, senal, monto, precio)
            else:
                logger.info(f"{par}: señal registrada sin ejecutar (prioridad={prioridad})")

        except Exception as e:
            logger.error(f"Error analizando {par}: {e}", exc_info=True)


async def _ciclo_deportivo():
    global _ciclos_sin_senal_deportiva
    if _ciclos_sin_senal_deportiva > 0 and _ciclos_sin_senal_deportiva % 2 != 0:
        _ciclos_sin_senal_deportiva += 1
        logger.info(f"── Deportivo salteado (ciclos secos: {_ciclos_sin_senal_deportiva}) ──")
        return
    logger.info("── Análisis DEPORTIVO ──────────────────────────────")
    try:
        señales = analizar_todos_los_partidos()

        if not señales:
            _ciclos_sin_senal_deportiva += 1
            logger.info(f"Deportivo: sin señales (ciclos secos: {_ciclos_sin_senal_deportiva})")
            return

        _ciclos_sin_senal_deportiva = 0

        for s in señales:
            partido    = s.partido    if hasattr(s, "partido")    else s.get("partido", "")
            seleccion  = s.seleccion  if hasattr(s, "seleccion")  else s.get("seleccion", "")
            odds       = s.odds       if hasattr(s, "odds")       else s.get("odds", 0)
            value_pct  = s.value_pct  if hasattr(s, "value_pct")  else s.get("value_pct", 0)
            mercado_id = s.mercado_id if hasattr(s, "mercado_id") else s.get("mercado_id", "")

            logger.info(f"AZURO value: {partido} | {seleccion} @ {odds} | value={value_pct:.1f}%")

            registrar_apuesta(
                mercado_id=mercado_id,
                descripcion=partido,
                monto=0,
                odds=odds,
                modo=MODE,
            )
            registrar_senal(
                fuente="analyst_sports_azuro",
                contenido=str(s),
                score=value_pct / 100,
                accion=seleccion,
                mercado_id=mercado_id,
            )

    except Exception as e:
        logger.error(f"Error en análisis deportivo: {e}", exc_info=True)


# ─── Loop rápido ─────────────────────────────────────────────────────────────

async def ciclo_monitoreo():
    try:
        posiciones = get_posiciones_onchain()
        if not posiciones:
            return

        logger.debug(f"Monitoreando {len(posiciones)} posición(es)")
        posiciones_actualizadas = monitorear_posiciones(posiciones)

        for pos in posiciones_actualizadas:
            estado_pos = pos.get("estado") if isinstance(pos, dict) else getattr(pos, "estado", None)
            coin       = pos.get("coin")   if isinstance(pos, dict) else getattr(pos, "coin", "")

            if estado_pos in ("TP", "SL"):
                emoji = "✅" if estado_pos == "TP" else "🛑"
                logger.info(f"{emoji} {estado_pos} ejecutado — {coin}")
                registrar_senal(
                    fuente=f"monitoreo_{estado_pos.lower()}",
                    contenido=str(pos),
                    score=1.0 if estado_pos == "TP" else 0.0,
                    accion=f"CERRAR_{estado_pos}",
                    mercado_id=coin,
                )

    except Exception as e:
        logger.debug(f"Monitoreo: {e}")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _calcular_monto(capital) -> float:
    if not capital:
        return 0.0
    saldo = capital.get("saldo", 0) if isinstance(capital, dict) else getattr(capital, "saldo", 0)
    monto = round(float(saldo) * TAMANO_APUESTA, 2)
    return monto if monto >= 1.0 else 0.0


def _ejecutar_trade(par: str, senal, monto: float, precio: float):
    """
    v3: usa TP/SL específicos de la señal (vienen de la estrategia):
    - TREND_FOLLOWING: TP 2.0% / SL 1.0%
    - ARBITRAJE:       TP 1.5% / SL 0.7%
    """
    es_long = senal.direccion == Direccion.SUBE
    side = "buy" if es_long else "sell"
    accion_log = "LONG" if es_long else "SHORT"
    emoji = "🟢" if es_long else "🔴"
    prefijo = "[DEMO]" if MODE == "demo" else "[REAL]"

    tp_pct = senal.tp_pct / 100   # convertir a decimal
    sl_pct = senal.sl_pct / 100

    if es_long:
        take_profit = precio * (1 + tp_pct)
        stop_loss   = precio * (1 - sl_pct)
    else:
        take_profit = precio * (1 - tp_pct)
        stop_loss   = precio * (1 + sl_pct)

    logger.info(
        f"{prefijo} {emoji} {accion_log} {par} | ${monto} @ {precio} | "
        f"TP={take_profit:.2f} ({senal.tp_pct}%) | SL={stop_loss:.2f} ({senal.sl_pct}%) | "
        f"estrategia={senal.estrategia.value}"
    )

    if MODE == "demo":
        lado = LadoPosicion.LONG if es_long else LadoPosicion.SHORT
        resultado = abrir_posicion_demo(
            activo=par,
            lado=lado,
            razon_senal=senal.razon,
        )
        registrar_apuesta(
            mercado_id=par,
            descripcion=f"{accion_log} {par} demo @ {precio} ({senal.estrategia.value})",
            monto=monto,
            odds=1.0,
            modo="demo",
        )
        logger.info(f"Demo trade registrado: {resultado}")
        return

    try:
        lado = LadoPosicion.LONG if es_long else LadoPosicion.SHORT
        resultado = ejecutar_apuesta(
            activo=par,
            lado=lado,
            tp_pct=senal.tp_pct,
            sl_pct=senal.sl_pct,
            razon_senal=senal.razon,
            modo="real",
        )
        )
        logger.info(f"Order ejecutada: {resultado}")
        registrar_apuesta(
            mercado_id=par,
            descripcion=f"{accion_log} {par} real @ {precio} ({senal.estrategia.value})",
            monto=monto,
            odds=1.0,
            modo="real",
        )
    except Exception as e:
        logger.error(f"Error ejecutando order {par}: {e}", exc_info=True)


# ─── Runner ─────────────────────────────────────────────────────────────────

async def main():
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║        TRADING BOT v3 — ARRANCANDO                  ║")
    logger.info(f"║  modo={MODE.upper():<6} capital=${CAPITAL_ACTUAL:<8}                   ║")
    logger.info(f"║  pares cripto: {', '.join(PARES_CRIPTO):<38}║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    inicializar_capital()

    ultimo_analisis = 0

    while True:
        ahora = time.time()

        if ahora - ultimo_analisis >= INTERVALO_LENTO:
            try:
                await ciclo_analisis()
            except Exception as e:
                logger.error(f"Error en ciclo_analisis: {e}", exc_info=True)
            ultimo_analisis = time.time()

        try:
            await ciclo_monitoreo()
        except Exception as e:
            logger.debug(f"Error en ciclo_monitoreo: {e}")

        await asyncio.sleep(INTERVALO_RAPIDO)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot detenido manualmente (Ctrl+C)")
