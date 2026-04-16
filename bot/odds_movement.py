"""
bot/odds_movement.py
Mejora 4 — Momentum + movimiento de odds.

Lógica:
- Guarda las odds de cada partido en Supabase al inicio de cada ciclo.
- Al analizar un partido, compara odds actuales vs odds de apertura (primera vez que se vio).
- Si las odds de un equipo SUBIERON (el mercado se alejó) pero los datos
  fundamentales (xG) lo favorecen → hay value confirmado por divergencia.

Ejemplo de edge:
  - Ajax tenía odds 2.10 al abrir → ahora están en 2.40
  - Pero su xG de los últimos 5 partidos es superior al rival
  - El mercado se equivocó al alejar las odds → apostar Ajax tiene value doble

Uso:
    from bot.odds_movement import guardar_odds, get_movimiento_odds
"""

import logging
from typing import Optional
from dataclasses import dataclass

from database.client import select, insert

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────────────────────────

@dataclass
class MovimientoOdds:
    """Describe cómo se movieron las odds de un partido desde la apertura."""
    game_id: str
    titulo: str

    # Odds de apertura (primera vez que se vio el partido)
    odds_apertura_home: float
    odds_apertura_draw: float
    odds_apertura_away: float

    # Odds actuales
    odds_actual_home: float
    odds_actual_draw: float
    odds_actual_away: float

    # Movimiento porcentual (positivo = odds subieron = equipo menos favorito ahora)
    mov_home: float   # (actual - apertura) / apertura
    mov_draw: float
    mov_away: float

    registros: int    # cuántas veces se vio este partido (proxy de antigüedad)

    def resumen(self) -> str:
        """Texto para incluir en el prompt de Claude."""
        lineas = ["📈 MOVIMIENTO DE ODDS (apertura → actual):"]
        lineas.append(
            f"  Local:     {self.odds_apertura_home:.2f} → {self.odds_actual_home:.2f} "
            f"({self.mov_home:+.1%})"
        )
        lineas.append(
            f"  Empate:    {self.odds_apertura_draw:.2f} → {self.odds_actual_draw:.2f} "
            f"({self.mov_draw:+.1%})"
        )
        lineas.append(
            f"  Visitante: {self.odds_apertura_away:.2f} → {self.odds_actual_away:.2f} "
            f"({self.mov_away:+.1%})"
        )
        lineas.append(f"  (basado en {self.registros} ciclos de observación)")
        return "\n".join(lineas)

    def hay_divergencia(self, seleccion: str, xg_favorece: bool) -> bool:
        """
        Detecta divergencia: odds subieron (mercado se alejó) pero xG favorece al equipo.
        Eso es edge doble — el mercado está equivocado Y los datos lo confirman.

        seleccion: "HOME" | "DRAW" | "AWAY"
        xg_favorece: True si el xG del equipo es superior al rival
        """
        if not xg_favorece:
            return False

        mov = {
            "HOME": self.mov_home,
            "DRAW": self.mov_draw,
            "AWAY": self.mov_away,
        }.get(seleccion, 0.0)

        # Odds subieron más de 3% → mercado se alejó → posible value
        return mov > 0.03


# ─────────────────────────────────────────────────────────────
# Guardar odds en Supabase
# ─────────────────────────────────────────────────────────────

def guardar_odds(mercados: list) -> None:
    """
    Guarda las odds actuales de una lista de Mercado1X2 en odds_historicas.
    Llamar una vez por ciclo, antes de analizar partidos.
    """
    guardados = 0
    for m in mercados:
        try:
            insert("odds_historicas", {
                "game_id": m.game_id,
                "titulo": m.titulo,
                "liga": m.liga,
                "odds_home": m.odds_home,
                "odds_draw": m.odds_draw,
                "odds_away": m.odds_away,
            })
            guardados += 1
        except Exception as e:
            logger.warning(f"Error guardando odds para {m.titulo}: {e}")

    logger.info(f"Odds guardadas: {guardados}/{len(mercados)} partidos")


# ─────────────────────────────────────────────────────────────
# Obtener movimiento de odds
# ─────────────────────────────────────────────────────────────

def get_movimiento_odds(game_id: str, odds_actual_home: float,
                        odds_actual_draw: float, odds_actual_away: float) -> Optional[MovimientoOdds]:
    """
    Busca el historial de odds de un partido en Supabase y calcula el movimiento.
    Devuelve None si no hay suficiente historial (menos de 2 registros).
    """
    try:
        rows = select(
            "odds_historicas",
            filters={"game_id": f"eq.{game_id}"},
            order="fecha.asc",
        )
    except Exception as e:
        logger.warning(f"Error consultando odds_historicas para {game_id}: {e}")
        return None

    if not rows or len(rows) < 2:
        # Primera vez que vemos este partido — no hay historial para comparar
        return None

    # Odds de apertura = primer registro
    apertura = rows[0]
    titulo = apertura.get("titulo", "")

    odds_ap_home = float(apertura.get("odds_home", odds_actual_home))
    odds_ap_draw = float(apertura.get("odds_draw", odds_actual_draw))
    odds_ap_away = float(apertura.get("odds_away", odds_actual_away))

    # Calcular movimiento porcentual
    def mov_pct(apertura_val: float, actual_val: float) -> float:
        if apertura_val == 0:
            return 0.0
        return (actual_val - apertura_val) / apertura_val

    return MovimientoOdds(
        game_id=game_id,
        titulo=titulo,
        odds_apertura_home=odds_ap_home,
        odds_apertura_draw=odds_ap_draw,
        odds_apertura_away=odds_ap_away,
        odds_actual_home=odds_actual_home,
        odds_actual_draw=odds_actual_draw,
        odds_actual_away=odds_actual_away,
        mov_home=mov_pct(odds_ap_home, odds_actual_home),
        mov_draw=mov_pct(odds_ap_draw, odds_actual_draw),
        mov_away=mov_pct(odds_ap_away, odds_actual_away),
        registros=len(rows),
    )


# ─────────────────────────────────────────────────────────────
# Testing — python -m bot.odds_movement
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test: simular movimiento
    mov = MovimientoOdds(
        game_id="test_123",
        titulo="Ajax vs PSV",
        odds_apertura_home=2.10,
        odds_apertura_draw=3.20,
        odds_apertura_away=3.50,
        odds_actual_home=2.40,
        odds_actual_draw=3.10,
        odds_actual_away=3.20,
        mov_home=0.143,
        mov_draw=-0.031,
        mov_away=-0.086,
        registros=4,
    )
    print(mov.resumen())
    print(f"\nDivergencia HOME (xG favorece): {mov.hay_divergencia('HOME', True)}")
    print(f"Divergencia HOME (xG no favorece): {mov.hay_divergencia('HOME', False)}")
