LIGAS_PRIORITARIAS = {
    "eredivisie", "liga portugal", "championship",
    "superliga argentina", "bundesliga 2", "bundesliga ii",
    "ligue 2", "serie b", "segunda division"
}

ODDS_MIN = 1.50
ODDS_MAX = 4.00
MINUTOS_MINIMOS_ANTES_DEL_PARTIDO = 30

def calcular_prob_implicita(odds: float) -> float:
    if odds <= 0:
        return 0.0
    return 1.0 / odds

def tiene_value_bruto(odds: float, prob_minima: float = 0.30) -> bool:
    prob = calcular_prob_implicita(odds)
    return prob >= prob_minima

def pre_filtrar_partido(partido: dict) -> tuple[bool, str]:
    liga = partido.get("liga", "").lower()
    odds_local = float(partido.get("odds_local", 0))
    odds_visitante = float(partido.get("odds_visitante", 0))
    odds_empate = float(partido.get("odds_empate", 0))
    minutos = partido.get("minutos_para_inicio", 999)

    if minutos < MINUTOS_MINIMOS_ANTES_DEL_PARTIDO:
        return False, f"partido en {minutos} min"

    odds_validas = [o for o in [odds_local, odds_empate, odds_visitante] if o > 0 and ODDS_MIN <= o <= ODDS_MAX]
    if not odds_validas:
        return False, "odds fuera de rango"

    if not any(tiene_value_bruto(o) for o in odds_validas):
        return False, "sin prob implicita razonable"

    es_prioritaria = any(lp in liga for lp in LIGAS_PRIORITARIAS)
    return True, "liga prioritaria" if es_prioritaria else "liga estándar"

def ordenar_partidos_por_prioridad(partidos: list[dict]) -> list[dict]:
    def score(p):
        liga = p.get("liga", "").lower()
        es_prioritaria = any(lp in liga for lp in LIGAS_PRIORITARIAS)
        return (0 if es_prioritaria else 1, p.get("minutos_para_inicio", 999))
    return sorted(partidos, key=score)

def aplicar_prefiltro_lista(partidos: list[dict]) -> tuple[list[dict], dict]:
    pasaron = []
    descartados = 0
    for partido in partidos:
        pasar, razon = pre_filtrar_partido(partido)
        if pasar:
            partido["_prioridad"] = razon
            pasaron.append(partido)
        else:
            descartados += 1
    total = len(partidos)
    ahorro_pct = round((descartados / total * 100) if total > 0 else 0, 1)
    return ordenar_partidos_por_prioridad(pasaron), {"total": total, "pasaron": len(pasaron), "descartados": descartados, "ahorro_pct": ahorro_pct}
