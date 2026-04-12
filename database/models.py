from database.client import get_client
import logging

logger = logging.getLogger(__name__)

def crear_tablas():
    """Crea las tablas necesarias en Supabase si no existen."""
    client = get_client()
    
    # Las tablas se crean via SQL en Supabase
    # Este archivo define la estructura esperada como referencia
    
    TABLAS = {
        "capital": """
            id bigint generated always as identity primary key,
            fecha timestamptz default now(),
            saldo numeric(12,2) not null,
            perdida_dia numeric(12,2) default 0,
            stop_loss_activo boolean default false,
            modo text default 'demo'
        """,
        "apuestas": """
            id bigint generated always as identity primary key,
            fecha timestamptz default now(),
            mercado_id text,
            mercado_descripcion text,
            monto numeric(12,2),
            odds numeric(8,4),
            resultado text default 'pendiente',
            ganancia_perdida numeric(12,2) default 0,
            modo text default 'demo',
            plataforma text default 'azuro'
        """,
        "senales": """
            id bigint generated always as identity primary key,
            fecha timestamptz default now(),
            fuente text,
            contenido text,
            score_sentimiento numeric(4,2),
            accion_tomada text,
            mercado_id text
        """,
        "odds_historial": """
            id bigint generated always as identity primary key,
            fecha timestamptz default now(),
            mercado_id text,
            descripcion text,
            odds numeric(8,4),
            plataforma text default 'azuro'
        """
    }
    
    logger.info(f"Estructura de tablas definida: {list(TABLAS.keys())}")
    return TABLAS

def registrar_apuesta(mercado_id, descripcion, monto, odds, modo="demo"):
    client = get_client()
    data = {
        "mercado_id": mercado_id,
        "mercado_descripcion": descripcion,
        "monto": monto,
        "odds": odds,
        "modo": modo,
        "resultado": "pendiente"
    }
    result = client.table("apuestas").insert(data).execute()
    logger.info(f"Apuesta registrada: {descripcion} | odds: {odds} | monto: {monto}")
    return result

def registrar_senal(fuente, contenido, score, accion, mercado_id=None):
    client = get_client()
    data = {
        "fuente": fuente,
        "contenido": contenido,
        "score_sentimiento": score,
        "accion_tomada": accion,
        "mercado_id": mercado_id
    }
    result = client.table("senales").insert(data).execute()
    return result

def obtener_capital_actual(modo="demo"):
    client = get_client()
    result = client.table("capital")\
        .select("*")\
        .eq("modo", modo)\
        .order("fecha", desc=True)\
        .limit(1)\
        .execute()
    return result.data[0] if result.data else None
