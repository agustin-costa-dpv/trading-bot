import logging
from datetime import datetime
from bot.azuro import listar_mercados
from bot.risk import evaluar_apuesta, obtener_estado_capital
from database.models import registrar_apuesta, registrar_senal
from database.client import insert, select, update

logger = logging.getLogger(__name__)

def ejecutar_ciclo_demo(prob_estimada=0.55):
    """
    Ciclo completo demo:
    1. Obtiene mercados de Azuro
    2. Evalúa cada uno con las reglas de riesgo
    3. Registra apuestas simuladas en Supabase
    """
    logger.info("=== CICLO DEMO INICIADO ===")

    estado = obtener_estado_capital("demo")
    logger.info(f"Capital: {estado['saldo']} | Pérdida día: {estado['perdida_dia']} | Stop: {estado['stop_loss_activo']}")

    mercados = listar_mercados()
    logger.info(f"Mercados disponibles: {len(mercados)}")

    apuestas_realizadas = 0
    apuestas_rechazadas = 0

    for mercado in mercados[:50]:  # máximo 50 por ciclo
        decision = evaluar_apuesta(mercado, prob_estimada, modo="demo")

        if decision["aprobar"]:
            # Elegir el outcome con mejor odds
            mejor_outcome = max(mercado["outcomes"], key=lambda o: o["odds"])

            try:
                registrar_apuesta(
                    mercado_id=f"{mercado['condition_id']}_{mejor_outcome['id']}",
                    descripcion=f"{mercado['titulo']} | outcome {mejor_outcome['id']}",
                    monto=decision["monto"],
                    odds=mejor_outcome["odds"],
                    modo="demo"
                )
                apuestas_realizadas += 1
                logger.info(
                    f"  ✓ APUESTA: {mercado['titulo']} | "
                    f"odds: {mejor_outcome['odds']} | "
                    f"monto: {decision['monto']} | "
                    f"value: {decision['value']:.2%}"
                )
            except Exception as e:
                logger.error(f"Error registrando apuesta: {e}")
        else:
            apuestas_rechazadas += 1

    logger.info(f"=== CICLO DEMO FINALIZADO ===")
    logger.info(f"Apuestas realizadas: {apuestas_realizadas}")
    logger.info(f"Apuestas rechazadas: {apuestas_rechazadas}")
    return apuestas_realizadas

def ver_historial(limit=10):
    """Muestra el historial de apuestas demo."""
    try:
        rows = select("apuestas", filters={"modo": "eq.demo", "order": "fecha.desc", "limit": str(limit)})
        if not rows:
            print("Sin apuestas registradas")
            return
        print(f"\n{'='*60}")
        print(f"{'FECHA':<20} {'DESCRIPCIÓN':<30} {'MONTO':>7} {'ODDS':>6} {'RESULTADO'}")
        print(f"{'='*60}")
        for a in rows:
            fecha = a["fecha"][:16].replace("T", " ")
            desc = a["mercado_descripcion"][:28]
            print(f"{fecha:<20} {desc:<30} {a['monto']:>7.2f} {a['odds']:>6.2f} {a['resultado']}")
        print(f"{'='*60}\n")
    except Exception as e:
        logger.error(f"Error obteniendo historial: {e}")

def resumen_pnl():
    """Calcula el P&L total del modo demo."""
    try:
        rows = select("apuestas", filters={"modo": "eq.demo"})
        if not rows:
            print("Sin apuestas registradas")
            return

        total = len(rows)
        pendientes = sum(1 for a in rows if a["resultado"] == "pendiente")
        ganadas = sum(1 for a in rows if a["resultado"] == "ganada")
        perdidas = sum(1 for a in rows if a["resultado"] == "perdida")
        pnl = sum(float(a["ganancia_perdida"] or 0) for a in rows)
        invertido = sum(float(a["monto"]) for a in rows)

        print(f"\n{'='*40}")
        print(f"  RESUMEN P&L DEMO")
        print(f"{'='*40}")
        print(f"  Total apuestas:  {total}")
        print(f"  Pendientes:      {pendientes}")
        print(f"  Ganadas:         {ganadas}")
        print(f"  Perdidas:        {perdidas}")
        print(f"  Total invertido: ${invertido:.2f}")
        print(f"  P&L neto:        ${pnl:.2f}")
        print(f"{'='*40}\n")
    except Exception as e:
        logger.error(f"Error calculando P&L: {e}")
