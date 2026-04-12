import logging
from config import STOP_LOSS_POR_APUESTA, PERDIDA_DIARIA_MAXIMA, TAMANO_MAXIMO_APUESTA, CAPITAL_INICIAL_DEMO
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

def calcular_value(odds, prob_estimada):
    return round((odds * prob_estimada) - 1, 4)

def evaluar_apuesta(mercado, prob_estimada, modo="demo"):
    estado = obtener_estado_capital(modo)
    bloqueado, razon = verificar_stop_loss(estado)
    if bloqueado:
        return {"aprobar": False, "razon": razon, "monto": 0, "value": 0}

    capital = float(estado["saldo"])
    odds_max = max(o["odds"] for o in mercado["outcomes"])
    value = calcular_value(odds_max, prob_estimada)

    if value < 0.05:
        return {"aprobar": False, "razon": f"Value insuficiente: {value:.2%}", "monto": 0, "value": value}

    monto = min(capital * TAMANO_MAXIMO_APUESTA, capital * STOP_LOSS_POR_APUESTA)
    return {
        "aprobar": True,
        "razon": f"Value: {value:.2%}",
        "monto": round(monto, 2),
        "value": value,
        "capital": capital
    }
