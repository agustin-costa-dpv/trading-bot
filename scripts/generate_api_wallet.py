"""
scripts/generate_api_wallet.py
Genera una API wallet para Hyperliquid completamente offline.
La private key se crea localmente con entropía del sistema operativo
y NUNCA sale de tu máquina.

Uso:
    python -m scripts.generate_api_wallet
"""

from eth_account import Account
import secrets


def generar_api_wallet() -> tuple[str, str]:
    """
    Genera un par (private_key, address) criptográficamente seguro.
    Usa secrets.token_hex() que lee entropía del OS (urandom).
    """
    # 32 bytes = 256 bits de entropía
    private_key = "0x" + secrets.token_hex(32)
    account = Account.from_key(private_key)
    return private_key, account.address


if __name__ == "__main__":
    print("=" * 70)
    print("🔐 GENERANDO API WALLET PARA HYPERLIQUID")
    print("=" * 70)

    private_key, address = generar_api_wallet()

    print(f"\n📋 API Wallet Address (PÚBLICA, pegar en Hyperliquid):")
    print(f"   {address}")

    print(f"\n🔑 API Wallet Private Key (PRIVADA, NUNCA COMPARTIR):")
    print(f"   {private_key}")

    print("\n" + "=" * 70)
    print("⚠️  GUARDÁ LA PRIVATE KEY EN UN LUGAR SEGURO AHORA")
    print("=" * 70)

    print("""
📝 Próximos pasos:

1. Copiá estos valores a tu .env:

   HYPERLIQUID_API_ADDRESS=<pegá la address>
   HYPERLIQUID_API_PRIVATE_KEY=<pegá la private key>

2. En Hyperliquid (https://app.hyperliquid.xyz/API):
   - Nombre: trading-bot
   - Pegá la Address en el segundo campo
   - Click en "Authorize API Wallet"
   - Firmá con MetaMask

3. Guardá este archivo cerrándolo SIN mostrar las keys en terminal.
   Podés correrlo de nuevo si necesitás crear otra API wallet.
""")