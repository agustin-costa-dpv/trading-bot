# CHANGELOG

> Bitácora de sesiones de trabajo sobre el bot de cripto v3.
> Convención: cada sesión es una entrada con fecha. Las entradas se agregan al
> tope (más reciente arriba). Al iniciar una sesión nueva, leer las últimas
> 5–10 entradas para contexto.

---

## 2026-05-07 — Activación del modo dual + diagnóstico falso positivo

### Estado del bot al cierre de la sesión

| Componente | Valor |
| --- | --- |
| Branch productivo | `main` |
| Último commit deployado | `5346578` |
| Modo de operación | `MODE=dual` (env var Render) |
| Capital real | $100 USDC en Hyperliquid |
| Capital demo | $100 USD virtuales (Supabase) |
| Leverage Hyperliquid | 5x isolated (BTC y ETH), seteado en UI del exchange |
| `TAMANO_APUESTA_PCT` | 0.10 (sizing real, en `bot/hyperliquid.py:60`) |
| `TAMANO_APUESTA` | 0.04 (solo logging y Supabase, en `main.py:79`) |
| Pares activos | BTC, ETH |
| Estrategias activas | TREND_FOLLOWING (ambos), ARBITRAJE (solo ETH) |
| TP/SL | 2.0% / 1.0% (TREND_FOLLOWING) |
| Floor de Hyperliquid | $10 nominal |
| Última posición real abierta | ETH SHORT @ 2295.85, oid=415021083916, abierta 14:44:51 |

### Cambios aplicados

1. **Render env var:** `MODE` cambiada a `dual` (estaba en `real` o `demo`, no
   recordado con certeza).
2. **Render env var:** `CAPITAL_REAL` cambiada de `50` a `100`.
3. **Depósito en Hyperliquid:** $47 USDC adicionales para llegar a $100 USDC
   en la cuenta real.
4. **Código (`bot/hyperliquid.py:60`):** `TAMANO_APUESTA_PCT` cambiada de
   `0.25` a `0.10`. Commit `5346578`.

### Validaciones empíricas

- **Modo dual funciona en producción.** Primera operación dual confirmada el
  7/5/2026 a las 14:44:51 — demo y real abrieron simultáneamente la misma
  señal SHORT ETH, con TP/SL nativos colocados correctamente en el exchange.
  Logs muestran `Order ejecutada: {'demo': ..., 'real': ...}` con ambos
  diccionarios completos.
- **Leverage 5x isolated confirmado empíricamente.** UI de Hyperliquid mostró
  Position Value $10.10 USDC con Margin $2.02 (Isolated) para una posición
  con `tamano_usd=10.0`. Esto resuelve la ambigüedad sobre si `tamano_usd`
  era nominal o margen: **es nominal**.
- **Logs nuevos esperados:** `[REAL] LONG/SHORT ...`, `Order ejecutada: ...`,
  `REAL: SHORT ETH sz=... @ ... oid=...`, `TP trigger colocado: oid=...`,
  `SL trigger colocado: oid=...`. Si en futuras sesiones no aparecen, hay
  que diagnosticar.

### Bugs y hallazgos descubiertos (no resueltos)

#### 1. Doble variable de sizing — Severidad: ALTA
- `TAMANO_APUESTA = 0.04` en `main.py:79` se usa para `monto` en logs y
  `registrar_apuesta()` (Supabase).
- `TAMANO_APUESTA_PCT = 0.10` en `bot/hyperliquid.py:60` se usa para sizing
  real en `abrir_posicion_demo` (L284) y `abrir_posicion_real` (L421).
- **Consecuencia:** Supabase registra "monto $4" pero el bot abre posición
  de $10 nominal. El histórico de Supabase del bot subreporta ~60% el
  sizing real.
- **Impacto en validación:** cualquier análisis de PnL/Sharpe/PF sobre la
  tabla `apuestas` de Supabase está distorsionado. **Usar `tamano_usd` de
  la tabla `posiciones_activas` para análisis serios.**
- Pendiente refactor: unificar a una sola variable.

#### 2. Floor de BTC en sizing actual — Severidad: MEDIA
- Con `CAPITAL_REAL=$100` y `TAMANO_APUESTA_PCT=0.10`, nominal = $10.
- Para BTC a $80,000 con `szDecimals=5`, `round(0.000125, 5)` puede truncar
  a `0.00012`, dando notional `$9.60` que falla el floor de Hyperliquid.
- **Confirmado en producción:** 15:45:37 del 7/5/2026, BTC SHORT skipeado
  con warning `Notional $9.60 < $10 minimo`.
- ETH no tiene el problema (precio bajo, más granularidad de szDecimals).
- Estimación: ~50% de las señales BTC van a fallar el floor mientras
  estemos en este sizing.

#### 3. Bug histórico documentado — bug `ciclo_monitoreo` key paths
- `pos.get("coin")` debería ser `pos.get("position", {}).get("coin")`.
- Hyperliquid sigue ejecutando TP/SL del lado del exchange, pero el log
  interno está roto.
- Ver memoria persistente para detalle.

#### 4. Bug histórico documentado — ETH silent execution failure
- Cuando hay BTC abierto, ETH puede fallar con `orden_id=''` por margin
  saturation o parsing mal de `statuses[0]`.
- Esto NO se observó hoy en la operación dual del 14:44, pero sigue como
  potencial.

#### 5. Confusión de nomenclatura — Severidad: BAJA
- En logs aparece `[ALTA]/[MEDIA]/[BAJA]` para describir DOS cosas:
  prioridad de sesión (en banner del ciclo) y probabilidad de señal (en
  `agent.analyst`). Mismo vocabulario, conceptos distintos.
- Ejemplo confuso del 7/5/2026 16:16: ETH señal con `prob=0.82 [ALTA]`
  pero "no ejecutada" porque la sesión POST_OVERLAP es `[MEDIA]`.
- No es bug funcional, pero confunde el diagnóstico.

### Decisiones pendientes (próxima sesión)

1. **Refactor de sizing:** unificar `TAMANO_APUESTA` y `TAMANO_APUESTA_PCT`.
   Definir valor único, decidir si lectura desde env var o constante.
2. **Resolución del floor de BTC:** subir `TAMANO_APUESTA_PCT` a 0.15, o
   ajustar el cálculo para que respete el floor con margen.
3. **Manejo del histórico de Supabase inconsistente:** ¿flag de versión?
   ¿tabla nueva? ¿aceptar y empezar análisis fresh desde 7/5/2026?
4. **Separación del bot de Azuro:** crear repo `trading-bot-azuro` cuando
   v3 cripto esté validado. Mientras tanto, no tocar módulos de Azuro en
   este repo.
5. **Criterio cuantitativo de validación:** reemplazar la regla "30 días"
   por "N señales ejecutadas con PF > 1.25 y Sharpe > 0.8 en live".

### Lecciones de proceso (no técnicas)

#### Lección 1: cierres no son aperturas
Cuando aparezca un log `DEMO cerrada: ...` o similar, **NO asumir que hubo
apertura reciente**. Verificar timestamp de apertura primero. Las posiciones
huérfanas heredadas de versiones anteriores del código pueden cerrarse y
generar logs que parecen actividad nueva.

**Costo de no haberlo hecho hoy:** ~2-3 horas de diagnóstico falso buscando
un bug que no existía en "real no abre".

#### Lección 2: leer git log antes de proponer arquitectura
Cuando el comportamiento del bot no coincide con lo esperado, primero
revisar:
- `git log --oneline -20 -- main.py` (¿qué cambió recientemente?)
- `git log --since="X days ago"` (¿hay commits nuevos no deployados?)
- Verificar último arranque del bot en logs (`grep ARRANCANDO`).

**Costo de no haberlo hecho hoy:** propuesta de "implementar modo dual
desde cero" cuando ya estaba implementado correctamente.

#### Lección 3: cuestionamiento mutuo es sano
Hubo dos momentos en la sesión donde Agustín cuestionó propuestas y los
cuestionamientos eran correctos:
1. "¿Por qué tenemos demo y real en paralelo?" → reconoció que era ruido
   emocional post-descubrimiento de bug, no señal. Decidió mantener.
2. "¿El bot apalanca x1 o x5?" → forzó verificación que reveló
   desalineación entre código (LEVERAGE=1) y realidad (5x isolated en UI).
3. "¿Por qué nunca propusiste separar Azuro del cripto?" → crítica válida,
   mezcla generaba complejidad oculta.

**Aprendizaje:** los cuestionamientos del usuario son mecanismo de
corrección, no obstáculo a mover rápido.

#### Lección 4: documentar es parte del trabajo
Esta sesión arrancó con 2-3 horas perdidas porque no había memoria
persistente sobre fixes recientes. Antes de cerrar cada sesión:
- Actualizar memoria persistente con descubrimientos clave
- Agregar entrada en CHANGELOG.md
- Idealmente: actualizar RUNBOOK.md (pendiente para próxima sesión)

### Observaciones del mercado durante la sesión

- Régimen INDEFINIDO la mayor parte de la mañana (BTC ADX 18-22, ETH 18-22).
- Régimen TENDENCIA bajista a partir de 14:14 con ADX subiendo (28→50).
- Señales SHORT con RSI en zona de sobreventa extrema (25-32) — el analyst
  de Claude las marca como "riesgo de rebote técnico" pero ejecuta porque
  pasan el umbral de probabilidad. Patrón a observar en próximas sesiones.
- Operación abierta al cierre: ETH SHORT 0.0044 @ 2296.0, en ganancia leve
  (+$0.01).

---

<!-- Próximas entradas se agregan ARRIBA de esta línea -->
