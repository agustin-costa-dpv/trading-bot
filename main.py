"""
main.py — Orquestador principal del bot de trading
Corre en dos loops paralelos:
  - Loop lento:  cada 30 minutos → análisis completo + señales
  - Loop rápido: cada 30 segundos → monitoreo de posiciones abiertas
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

# ─── Módulos del proyecto ────────────────────────────────────────────────────
from agent.sessions import get_sesion_actual, puede_operar_ahora, get_multiplicador_capital
from agent.analyst import analizar
from agent.analyst_sports import analizar_todos_los_partidos
from bot.hyperliquid import (
    get_precio_mark,
    get_saldo_usdc,
    get_posiciones_onchain,
    ejecutar_apuesta,
    monitorear_posiciones,
    abrir_posicion_demo,
)
from bot.risk import verificar_stop_loss, obtener_estado_capital, evaluar_apuesta
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

PARES_CRIPTO      = ["BTC", "ETH", "SOL"]
INTERVALO_LENTO   = 30 * 60   # 30 minutos
INTERVALO_RAPIDO  = 30        # 30 segundos

TAKE_PROFIT_PCT = 0.015   # +1.5%
STOP_LOSS_PCT   = 0.020   # -2.0%
TAMANO_APUESTA  = 0.04    # 4% del capital por trade
LEVERAGE        = 1


# ─── Inicialización ──────────────────────────────────────────────────────────

def inicializar_capital():
    """Crea el registro de capital inicial si no existe."""
    for modo in ("demo", "real"):
        capital = obtener_capital(modo)
        if not capital:
            saldo = CAPITAL_DEMO if modo == "demo" else CAPITAL_REAL
            crear_capital_inicial(modo=modo, saldo=saldo)
            logger.info(f"Capital inicial creado — modo={modo} saldo=${saldo}")
        else:
            logger.info(f"Capital existente — modo={modo} saldo=${capital.get('saldo', '?')}")


# ─── Loop lento: análisis cada 30 minutos ────────────────────────────────────

async def ciclo_analisis():
    logger.info("═" * 60)
    logger.info(f"CICLO DE ANÁLISIS — {datetime.now().strftime('%H:%M:%S')} — modo={MODE.upper()}")

    # 1. Verificar sesión horaria
    if not puede_operar_ahora():
        logger.info("⏸  Sesión con baja liquidez — análisis omitido")
        return

    sesion = get_sesion_actual()
    # Sesion puede ser dataclass o dict — manejamos ambos casos
    prioridad     = sesion.prioridad.value if hasattr(sesion, "prioridad") else sesion.get("prioridad", "EVITAR")
    nombre_sesion = sesion.nombre if hasattr(sesion, "nombre") else sesion.get("nombre", "desconocida")
    logger.info(f"Sesión actual: {nombre_sesion} [{prioridad}]")

    # 2. Verificar stop loss diario
    estado = obtener_estado_capital(MODE)
    if estado:
        bloqueado, razon = verificar_stop_loss(estado)
        if bloqueado:
            logger.warning(f"🛑 {razon} — operaciones bloqueadas")
            return

    capital = obtener_capital(MODE)

    # 3. Análisis cripto
    if prioridad in ("ALTA", "MEDIA"):
        await _ciclo_cripto(prioridad, capital)

    # 4. Análisis deportivo — solo sesión ALTA
    if prioridad == "ALTA":
        await _ciclo_deportivo()


async def _ciclo_cripto(prioridad: str, capital):
    """Analiza los pares cripto con analyst.analizar() y ejecuta/registra."""
    logger.info("── Análisis CRIPTO ──────────────────────────────────")

    for par in PARES_CRIPTO:
        try:
            logger.info(f"Analizando {par}...")
            senal = analizar(par)   # Devuelve Optional[Senal]

            if not senal:
                logger.info(f"{par}: sin señal")
                continue

            accion  = senal.accion  if hasattr(senal, "accion")  else senal.get("accion", "ESPERAR")
            score   = senal.score   if hasattr(senal, "score")   else senal.get("score", 0)
            resumen = senal.resumen if hasattr(senal, "resumen") else senal.get("resumen", "")
            precio  = get_precio_mark(par)

            if accion == "ESPERAR":
                logger.info(f"{par}: sin señal")
                continue

            logger.info(f"{par}: {accion} | score={score:.2f} | precio={precio}")

            registrar_senal(
                fuente=f"analyst_cripto_{par}",
                contenido=resumen,
                score=score,
                accion=accion,
                mercado_id=par,
            )

            if prioridad == "ALTA" and score >= 0.65:
                monto = _calcular_monto(capital)
                if monto <= 0:
                    logger.warning(f"{par}: capital insuficiente")
                    continue
                _ejecutar_trade(par, accion, monto, precio)

        except Exception as e:
            logger.error(f"Error analizando {par}: {e}", exc_info=True)


async def _ciclo_deportivo():
    """Analiza mercados Azuro y registra señales (sin ejecución real en Nivel 1)."""
    logger.info("── Análisis DEPORTIVO ──────────────────────────────")
    try:
        señales = analizar_todos_los_partidos()

        if not señales:
            logger.info("Deportivo: sin señales con value suficiente")
            return

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


# ─── Loop rápido: monitoreo cada 30 segundos ─────────────────────────────────

async def ciclo_monitoreo():
    """Verifica posiciones abiertas y aplica TP/SL."""
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


def _ejecutar_trade(par: str, accion: str, monto: float, precio: float):
    side   = "buy" if accion == "LONG" else "sell"
    emoji  = "🟢" if side == "buy" else "🔴"
    prefijo = "[DEMO]" if MODE == "demo" else "[REAL]"
    logger.info(f"{prefijo} {emoji} {accion} {par} | monto=${monto} | precio={precio}")

    if MODE == "demo":
        resultado = abrir_posicion_demo(
            activo=par,
            side=side,
            monto_usdc=monto,
            precio_entrada=precio,
            take_profit=precio * (1 + TAKE_PROFIT_PCT) if side == "buy" else precio * (1 - TAKE_PROFIT_PCT),
            stop_loss=precio * (1 - STOP_LOSS_PCT)     if side == "buy" else precio * (1 + STOP_LOSS_PCT),
        )
        registrar_apuesta(
            mercado_id=par,
            descripcion=f"{accion} {par} demo @ {precio}",
            monto=monto,
            odds=1.0,
            modo="demo",
        )
        logger.info(f"Demo trade registrado: {resultado}")
        return

    try:
        resultado = ejecutar_apuesta(
            activo=par,
            side=side,
            monto_usdc=monto,
            leverage=LEVERAGE,
        )
        logger.info(f"Order ejecutada: {resultado}")
        registrar_apuesta(
            mercado_id=par,
            descripcion=f"{accion} {par} real @ {precio}",
            monto=monto,
            odds=1.0,
            modo="real",
        )
    except Exception as e:
        logger.error(f"Error ejecutando order {par}: {e}", exc_info=True)


# ─── Runner principal ─────────────────────────────────────────────────────────

async def main():
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║        TRADING BOT — ARRANCANDO                     ║")
    logger.info(f"║  modo={MODE.upper():<6} capital=${CAPITAL_ACTUAL:<8}                   ║")
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