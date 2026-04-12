from database.client import select, insert, update
import logging

logger = logging.getLogger(__name__)

def obtener_capital(modo="demo"):
    try:
        rows = select("capital", filters={"modo": f"eq.{modo}", "order": "fecha.desc", "limit": "1"})
        return rows[0] if rows else None
    except Exception as e:
        logger.error(f"Error obteniendo capital: {e}")
        return None

def crear_capital_inicial(modo="demo", saldo=1000.0):
    return insert("capital", {"saldo": saldo, "perdida_dia": 0, "stop_loss_activo": False, "modo": modo})

def registrar_apuesta(mercado_id, descripcion, monto, odds, modo="demo"):
    return insert("apuestas", {
        "mercado_id": mercado_id,
        "mercado_descripcion": descripcion,
        "monto": monto,
        "odds": odds,
        "modo": modo,
        "resultado": "pendiente"
    })

def registrar_senal(fuente, contenido, score, accion, mercado_id=None):
    return insert("senales", {
        "fuente": fuente,
        "contenido": contenido,
        "score_sentimiento": score,
        "accion_tomada": accion,
        "mercado_id": mercado_id
    })
