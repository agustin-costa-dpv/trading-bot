from database.client import select, insert, update
import logging

logger = logging.getLogger(__name__)

# ============================================================
# Existentes (no se tocan)
# ============================================================

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


# ============================================================
# NUEVO: posiciones_activas
# ============================================================

def crear_posicion_activa(posicion_dict: dict) -> dict | None:
    """
    Crea una nueva posicion_activa en Supabase.
    Recibe dict (Posicion.to_dict() del bot, ya serializado).
    Devuelve la fila creada (con id) o None si fallo.
    """
    # Mapear campos del dict de Posicion a las columnas de la tabla
    payload = {
        "activo": posicion_dict.get("activo"),
        "lado": posicion_dict.get("lado"),
        "tamano_usd": posicion_dict.get("tamano_usd"),
        "precio_entrada": posicion_dict.get("precio_entrada"),
        "precio_actual": posicion_dict.get("precio_actual"),
        "tp_precio": posicion_dict.get("tp_precio"),
        "sl_precio": posicion_dict.get("sl_precio"),
        "estado": posicion_dict.get("estado", "ABIERTA"),
        "pnl_usd": posicion_dict.get("pnl_usd", 0),
        "pnl_pct": posicion_dict.get("pnl_pct", 0),
        "plataforma": posicion_dict.get("plataforma"),
        "orden_id": posicion_dict.get("orden_id"),
        "tp_oid": posicion_dict.get("tp_oid"),
        "sl_oid": posicion_dict.get("sl_oid"),
        "estrategia": posicion_dict.get("estrategia"),
        "razon_senal": posicion_dict.get("razon_senal"),
    }
    # Limpiar None para que postgres aplique defaults
    payload = {k: v for k, v in payload.items() if v is not None}

    try:
        result = insert("posiciones_activas", payload)
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error creando posicion_activa: {e}")
        return None


def listar_posiciones_abiertas(plataforma: str | None = None) -> list[dict]:
    """
    Lista todas las posiciones con estado=ABIERTA.
    Si se pasa plataforma, filtra por ella (hyperliquid_demo / hyperliquid_real).
    """
    filters = {"estado": "eq.ABIERTA"}
    if plataforma:
        filters["plataforma"] = f"eq.{plataforma}"
    try:
        return select("posiciones_activas", filters=filters)
    except Exception as e:
        logger.error(f"Error listando posiciones abiertas: {e}")
        return []


def actualizar_posicion_activa(posicion_id: int, cambios: dict) -> bool:
    """
    Actualiza campos de una posicion_activa por id.
    Retorna True si OK, False si fallo.
    """
    try:
        update("posiciones_activas", cambios, {"id": f"eq.{posicion_id}"})
        return True
    except Exception as e:
        logger.error(f"Error actualizando posicion_activa {posicion_id}: {e}")
        return False


def cerrar_posicion_activa(
    posicion_id: int,
    precio_cierre: float,
    motivo: str,
    pnl_usd: float = 0,
    pnl_pct: float = 0,
) -> bool:
    """
    Marca una posicion_activa como cerrada.
    motivo: 'TP' | 'SL' | 'MANUAL' | 'ERROR'
    """
    motivo = motivo.upper()
    estado_map = {
        "TP": "CERRADA_GANANCIA",
        "SL": "CERRADA_PERDIDA",
        "MANUAL": "CERRADA_MANUAL",
        "ERROR": "ERROR",
    }
    estado = estado_map.get(motivo, "CERRADA_MANUAL")

    from datetime import datetime, timezone
    cambios = {
        "estado": estado,
        "precio_cierre": precio_cierre,
        "motivo_cierre": motivo,
        "pnl_usd": pnl_usd,
        "pnl_pct": pnl_pct,
        "timestamp_cierre": datetime.now(timezone.utc).isoformat(),
    }
    return actualizar_posicion_activa(posicion_id, cambios)
