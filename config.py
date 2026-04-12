import os
from dotenv import load_dotenv

load_dotenv()

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Binance
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# Modo de operación
MODE = os.getenv("MODE", "demo")  # "demo" o "real"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Reglas de riesgo
STOP_LOSS_POR_APUESTA = 0.05     # 5% del capital por apuesta
PERDIDA_DIARIA_MAXIMA = 0.15     # 15% del capital total por día
TAMANO_MAXIMO_APUESTA = 0.05     # 5% del capital por apuesta
CAPITAL_INICIAL_DEMO = 1000.0    # USD simulados para modo demo

# Azuro Protocol
AZURO_GRAPHQL_URL = "https://thegraph-1.onchainfeed.org/subgraphs/name/azuro-protocol/azuro-data-feed-base"
AZURO_CHAIN = "polygon"
