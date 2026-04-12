from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_ANON_KEY
import logging

logger = logging.getLogger(__name__)

_client: Client = None

def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            raise ValueError("Faltan credenciales de Supabase en el .env")
        _client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        logger.info("Conexión a Supabase establecida")
    return _client
