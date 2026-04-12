import requests
import time
import logging
from config import AZURO_GRAPHQL_URL

logger = logging.getLogger(__name__)

def obtener_mercados():
    ahora = str(int(time.time()))
    query = """
    {
      games(
        first: 20
        orderBy: startsAt
        orderDirection: asc
        where: { startsAt_gt: \"%s\" }
      ) {
        id
        title
        startsAt
        sport { name }
        league { name country { name } }
        conditions(where: { state: Active }) {
          conditionId
          margin
          outcomes {
            outcomeId
            currentOdds
          }
        }
      }
    }
    """ % ahora

    try:
        r = requests.post(
            AZURO_GRAPHQL_URL,
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            logger.error(f"GraphQL error: {data['errors']}")
            return []
        games = data.get("data", {}).get("games", [])
        logger.info(f"Juegos obtenidos: {len(games)}")
        return games
    except Exception as e:
        logger.error(f"Error: {e}")
        return []

def listar_mercados():
    games = obtener_mercados()
    mercados = []
    for g in games:
        for c in g.get("conditions", []):
            outcomes = c.get("outcomes", [])
            if len(outcomes) < 2:
                continue
            mercados.append({
                "game_id": g["id"],
                "titulo": g["title"],
                "deporte": g.get("sport", {}).get("name", ""),
                "liga": g.get("league", {}).get("name", ""),
                "pais": g.get("league", {}).get("country", {}).get("name", ""),
                "condition_id": c["conditionId"],
                "margen": float(c.get("margin", 0)),
                "outcomes": [
                    {"id": o["outcomeId"], "odds": float(o["currentOdds"])}
                    for o in outcomes
                ]
            })
    return mercados
