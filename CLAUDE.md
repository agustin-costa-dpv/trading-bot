# CLAUDE.md — Trading Bot v3 (Hyperliquid)

## Contexto del proyecto

Bot de trading automatizado de perpetuales BTC/ETH en Hyperliquid, 5x apalancamiento aislado, modo dual (demo + real simultáneo). Desplegado en Render como Background Worker con auto-deploy desde GitHub. Estado en Supabase. Validación de señales vía API de Claude.

**Criterio de validación activo:** 50 señales reales cerradas, Profit Factor >1.25, Sharpe >0.8. Hasta que se cumpla, NO se agregan estrategias, plataformas ni parámetros nuevos.

## Reglas de trabajo (NO negociables)

1. **Un cambio por sesión.** Nunca mezclar dos cambios en un mismo commit o sesión. Flujo: backup → premortem → implementar → observar 48–72 hs o 10 señales nuevas antes del siguiente cambio.
2. **NUNCA hacer commit ni push sin confirmación explícita del usuario.** Cada push a `main` dispara un deploy automático en Render sobre el bot en producción con capital real.
3. **Premortem obligatorio antes de tocar zonas core:** lógica de ejecución (`ejecutar_apuesta` en `bot/hyperliquid.py` ~línea 540), sizing, persistencia en Supabase.
4. **Backup antes de editar:** copiar el archivo original antes de cualquier modificación.
5. **No operar cambios durante ventana de validación activa** salvo pedido explícito del usuario.

## Datos y métricas — trampas conocidas

- **Métricas limpias en Supabase:** filtrar `orden_id is not null` Y excluir filas `CERRADA_MANUAL`. La wallet real tiene trades manuales (incl. activos fuera del universo del bot, ej. HYPE) que contaminan las métricas.
- **NO usar el campo `monto` de Supabase para análisis de PnL** — subreporta el tamaño de posición ~60% (bug pendiente de unificación `TAMANO_APUESTA` / `TAMANO_APUESTA_PCT`).
- `TAMANO_APUESTA_PCT` en `bot/hyperliquid.py:60` controla el sizing real. `TAMANO_APUESTA` en `main.py:79` es solo logging.
- `tamano_usd` = exposición nominal, NO margen. Con 5x: margen = tamano_usd / 5.
- `LEVERAGE=1` en `main.py:80` es código muerto. El apalancamiento se configura en la UI de Hyperliquid (5x aislado BTC/ETH). Nunca inferir leverage del código.
- Fechas en formato D/M/Y. Dos bugs aparentes recientes fueron datos mal leídos (ventana de logs equivocada, D/M/Y leído como M/D/Y). Verificar datos crudos antes de concluir que hay un bug de código.

## Hallazgos del análisis de código (09/06/2026) — diagnóstico confirmado

1. **`horizonte_min` es código muerto (CRÍTICO).** El backtest v3 (Sharpe 1.03, PF 1.19) cierra por timeout a las 6 velas (TREND) / 2 velas (ARB) — ver `scripts/backtest_v3.py:240-245`. Producción NO tiene salida por tiempo: las posiciones quedan abiertas hasta TP/SL indefinidamente. El bot live ejecuta una estrategia distinta a la validada.
2. **PnL ciego a costos.** Se calcula con precio mark y `tamano_usd` nominal. No incluye fees taker (~0.045% por lado ≈ 0.09% round trip, ~9% del SL de 1%), ni funding, ni slippage. `slippage=0.01` en `market_open` (`bot/hyperliquid.py:459`) permite hasta 1% — igual al SL completo de TREND. El PF 1.86 de Supabase está inflado; ground truth = equity de la wallet.
3. **ARBITRAJE roto por cadencia:** detecta movimientos de 5 min pero el loop corre cada 30 min — entra hasta 30 min tarde a momentum agotado.
4. **Entradas tardías sistemáticas (queja principal del usuario):** EMAs 15m lentas + ADX≥28 (umbral tardío) + loop de 30 min + el prompt del validador (`agent/analyst.py:335`) instruye a Claude que RSI extremo en dirección de la señal "confirma momentum" → valida persecución cerca del reversal.
5. **Validador no determinista:** la llamada a Claude no fija `temperature=0`; misma señal puede pasar o no el umbral 0.58. El prompt además dice "Validar operación en Azuro" (resto del bot deportivo).
6. **Clasificación de cierres frágil:** throttle 120s + tolerancia ±0.1% sobre TP/SL puede registrar un TP como CERRADA_MANUAL o invertido. Solución: usar `info.user_fills()` (closedPnl real con fees).
7. **Sizing estático:** `CAPITAL_REAL` (env) × `TAMANO_APUESTA_PCT`, nunca lee `get_saldo_usdc()`. No compone ni se reduce en drawdown.

## Plan aprobado — 3 fases (NO saltear el orden)

**Fase 1 — Medición (no toca producción):** script standalone que registre/reconcilie PnL real desde `info.user_fills()` (fees incluidos). Arregla el instrumento antes de tocar la estrategia.
**Fase 2 — Re-backtest offline (no toca producción):** agregar a `backtest_v3.py`: (a) fees + slippage realistas, (b) muestreo 5 min vs 30, (c) guard anti-persecución (no entrar si precio > X ATRs de EMA21 o cambio_15m ya extendido), (d) variante ADX cruzando 22-25 en subida vs nivel 28, (e) SL adaptativo por ATR (ej. 1.5×ATR) + sizing por riesgo fijo en USD (tamaño = riesgo_usd / distancia_SL) en vez de SL 1% fijo y nominal estático, (f) filtro de tendencia en timeframe superior (solo señales 15m alineadas con estructura EMA de 1h/4h). Comparar variantes sobre 6 meses con costos.
**Fase 3 — Deploy de lo validado, UN cambio por vez con premortem:** 1º salida por tiempo, 2º guard de extensión, 3º cadencia, 4º vol_ratio, 5º decidir ARB (apagar o loop rápido). Triviales en paralelo: `temperature=0`, sacar/invertir la instrucción de RSI del prompt, limpiar "Azuro", bajar slippage a ~0.002.

## Bugs conocidos / pendientes

- **`vol_ratio` (Hallazgo 1 de sesión 05-06):** el cambio de umbral se aplicó solo al string del log (`min 0.60`), no a la condición `volumen_ok` (gate real: 0.9). Confirmado en `agent/analyst.py:217,225`. Fix programado en Fase 3.
- **Supabase RLS:** alerta sin resolver. Falta confirmar tablas flageadas y tipo de key (service_role vs anon).
- **Cambio C** (señales MEDIA con sizing reducido): diferido hasta que el capital real supere ~$143.

## Conceptos que se confunden

- **Prioridad de sesión vs probabilidad de señal:** el filtro en `main.py:164` solo ejecuta con prioridad de sesión ALTA (LONDON_KILL_ZONE, NY_KILL_ZONE, OVERLAP_LONDON_NY). MEDIA (POST_OVERLAP) registra pero no ejecuta — es diseño, no bug. Los tags "[ALTA]/[MEDIA]/[BAJA]" en logs de señales son probabilidad de la señal, NO prioridad de sesión. Mismo vocabulario, conceptos distintos.

## Infraestructura

- **Repo:** `agustin-costa-dpv/trading-bot` (público — nunca commitear secretos, verificar `.gitignore` incluye `.env`)
- **Archivos clave:** `main.py`, `agent/analyst.py`, `bot/hyperliquid.py`, `agent/sessions.py`
- **Deploy:** Render Background Worker (Oregon). Logs: Render → Worker → pestaña Logs. Keywords útiles: "BTC", "señal", "orden", "ABIERTA", "WARNING", "[REAL]", "Order ejecutada".
- **Env vars críticas (en Render):**
  - `HYPERLIQUID_API_PRIVATE_KEY`: clave privada de la API Wallet (64–66 chars hex). Bug recurrente: si se pone la dirección pública (42 chars), las órdenes reales fallan en silencio mientras demo funciona normal.
  - `HYPERLIQUID_MAIN_ADDRESS`: dirección pública de la wallet master MetaMask (42 chars).
  - `MODE`: dual / real / demo.
- **SDK:** `hyperliquid-python-sdk==0.23.0`, API `api.hyperliquid.xyz`
- **Supabase:** tablas `posiciones_activas`, `apuestas`/`senales`

## Diagnóstico: "real no abrió una posición que demo sí"

Orden de chequeo:
1. Env var `MODE` en Render
2. Rama real implementada en `main.py` (se rompió varias veces en commits de ~4–5/5/2026)
3. `HYPERLIQUID_API_PRIVATE_KEY` correcta (ver arriba)
4. Capital USDC en cuenta real
5. Sizing vs piso de $12 USD
6. Buscar "Order ejecutada" o "[REAL]" en logs de Render
7. Parsing de `statuses[0]` de la respuesta del SDK

## Ideas evaluadas y descartadas (no proponer)

- **Expandir a índices, oro, petróleo u otros activos:** descartado hasta cerrar la validación cripto. Es el error de v1/v2 (complejidad prematura).
- **MEAN_REVERSION en cripto:** ya probada y descartada por backtest (PF 0.84).

## Proyecto separado

El bot de apuestas deportivas Azuro vive en OTRO repo, diferido hasta cerrar la validación de v3. No mezclar nada de Azuro en este repo.
