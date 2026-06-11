import logging
from config import PERDIDA_DIARIA_MAXIMA, CAPITAL_INICIAL_DEMO
from database.models import obtener_capital, crear_capital_inicial

logger = logging.getLogger(__name__)

def obtener_estado_capital(modo="demo"):
    estado = obtener_capital(modo)
    if not estado:
        logger.info("Sin capital registrado, creando inicial...")
        crear_capital_inicial(modo, CAPITAL_INICIAL_DEMO)
        estado = obtener_capital(modo)
    return estado

def verificar_stop_loss(estado):
    if not estado:
        return True, "Sin datos de capital"
    if estado.get("stop_loss_activo"):
        return True, "Stop loss diario activo"
    perdida_dia = float(estado["perdida_dia"])
    capital = float(estado["saldo"])
    if perdida_dia >= capital * PERDIDA_DIARIA_MAXIMA:
        return True, f"Pérdida diaria máxima alcanzada: {perdida_dia:.2f}"
    return False, "OK"
