"""
scripts/test_env.py
Verifica que todas las variables de Hyperliquid estén bien cargadas.
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("🔍 Verificando .env para Hyperliquid")
print("=" * 60)

variables = [
    "HYPERLIQUID_MAIN_ADDRESS",
    "HYPERLIQUID_API_ADDRESS",
    "HYPERLIQUID_API_PRIVATE_KEY",
    "HYPERLIQUID_MODE",
    "CAPITAL_REAL",
    "CAPITAL_DEMO",
    "ANTHROPIC_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY",
]

for var in variables:
    valor = os.getenv(var)
    if not valor:
        print(f"❌ {var}: FALTA")
        continue

    # Enmascarar para no exponer en pantalla
    if "PRIVATE_KEY" in var or "SUPABASE_KEY" in var or "ANTHROPIC" in var:
        display = f"{valor[:6]}...{valor[-4:]} ({len(valor)} chars)"
    elif "ADDRESS" in var:
        # Validar formato address (0x + 40 hex)
        if valor.startswith("0x") and len(valor) == 42:
            display = f"{valor[:6]}...{valor[-4:]} ✓ (formato OK)"
        else:
            display = f"{valor[:6]}...{valor[-4:]} ⚠️  formato incorrecto (debe ser 0x + 40 chars)"
    else:
        display = valor

    print(f"✅ {var}: {display}")

print("\n" + "=" * 60)

# Chequeo extra: que MAIN y API address sean distintas
main = os.getenv("HYPERLIQUID_MAIN_ADDRESS", "").lower()
api = os.getenv("HYPERLIQUID_API_ADDRESS", "").lower()
if main and api:
    if main == api:
        print("⚠️  ALERTA: MAIN_ADDRESS y API_ADDRESS son iguales, deben ser distintas")
    else:
        print("✅ MAIN y API addresses son distintas (correcto)")

print("=" * 60)