# Registro de decisiones técnicas

Formato: contexto → decisión → alternativas → consecuencias. Una decisión aceptada solo se cambia
con una nueva entrada que la sustituya (no se reescribe la historia).

| ID | Decisión | Estado |
|---|---|---|
| 001 | PostgreSQL como base de datos | Aceptada (Fase 1) |
| 002 | Dataset sintético por escenarios | Aceptada (Fase 1) |
| 003 | Etiquetas con fecha de reporte | Aceptada (Fase 1) |
| 004 | Features point-in-time en SQL compartidas offline/online | Aceptada (Fase 1) |
| 005 | Validación temporal | Aceptada (Fase 1) |
| 006 | PR-AUC como métrica de selección | Aceptada (Fase 1) |
| 007 | Política de decisión optimizada por coste dependiente del importe | Aceptada (Fase 1) |
| 008 | Sin SMOTE por defecto | Aceptada (Fase 1) |
| 009 | Stack mínimo | Aceptada (Fase 1) |
| 010 | ¿Gradient Boosting como modelo final? | Aceptada (Fase 7): sí, con regresión logística como alternativa interpretable |
| 011 | Data Quality: rechazar, reparar o avisar | Aceptada (Fase 3) |
| 012 | Clientes sin país: rechazo en cascada | Aceptada (Fase 3) |
| 013 | Simulador v1.0.1: corrección de edad mínima | Aceptada (Fase 3) |
| 014 | Artefacto conocido: transacciones en los bordes del periodo | Aceptada (Fase 4) |
| 015 | EDA solo sobre el periodo de entrenamiento | Aceptada (Fase 4) |
| 016 | Warm-up de 90 días excluido del entrenamiento | Aceptada (Fase 5) |
| 017 | Features offline con window functions, online con lookups + test de paridad | Aceptada (Fase 5) |
| 018 | Sin pesos de clase ni remuestreo | Aceptada (Fase 6) |
| 019 | Modelos guardados con skops y lista explícita de tipos de confianza | Aceptada (Fase 6) |
| 020 | Modelo seleccionado por CV temporal; validación solo confirma | Aceptada (Fase 7) |
| 021 | Política por coste esperado por transacción (P2) | Aceptada (Fase 8) |
| 022 | El test se evalúa una sola vez y queda registrado en el repositorio | Aceptada (Fase 8) |
| 023 | Explicaciones con SHAP PermutationExplainer y comprobación de aditividad | Aceptada (Fase 9) |

---

## 001 — PostgreSQL

**Contexto.** Necesitamos SQL avanzado (window functions sobre ventanas de tiempo), integridad
referencial y una base que la API pueda consultar online.

**Decisión.** PostgreSQL 16 en Docker.

**Alternativas.**
- *SQLite:* sin `RANGE BETWEEN INTERVAL` en ventanas ni concurrencia para la API.
- *DuckDB:* excelente para analítica offline, pero no es una base transaccional para servir la API.
- *Parquet + pandas:* sin constraints ni SQL real; no cumple el objetivo de demostrar SQL.

**Consecuencias.** Hace falta Docker. `COUNT(DISTINCT)` no está soportado en window functions →
algunas features requieren `LATERAL` (y medir su rendimiento).

---

## 002 — Dataset sintético por escenarios

**Contexto.** Ningún dataset público combina customers, cards, merchants, devices y timestamps
reales (ver `phase1_design.md` §3.1).

**Decisión.** Simulador propio donde el fraude se genera por escenarios de comportamiento (card
testing, ATO, clonado, fraud rings, merchant comprometido, stealth), con ruido en comportamiento y
etiquetas. El generador se congela y versiona antes de modelar.

**Alternativas.** IEEE-CIS (rico pero anónimo y no redistribuible), Sparkov/TabFormer (sin devices).

**Consecuencias.** Riesgo de circularidad: el modelo puede aprender las reglas del generador. Se
declara abiertamente; el proyecto demuestra metodología, no hallazgos sobre fraude real.
Ventaja: conocemos el ground truth de escenarios y de defectos de calidad, lo que permite medir el
recall de las reglas DQ y hacer error analysis por escenario.

---

## 003 — Etiquetas con fecha de reporte

**Contexto.** En fraude con tarjeta, la etiqueta llega días o semanas después (chargeback).

**Decisión.** Tabla `core.fraud_labels(transaction_id, is_fraud, reported_at, label_source)` en
lugar de una columna `transactions.is_fraud`.

**Alternativas.** Columna booleana en `transactions` (más simple, pero hace invisible el problema
de disponibilidad de la etiqueta).

**Consecuencias.** Toda feature derivada de etiquetas debe filtrar `reported_at < ts`. El
monitoring de rendimiento tiene que esperar a que las etiquetas maduren.

---

## 004 — Features point-in-time en SQL compartidas offline/online

**Contexto.** Las features históricas son la principal fuente de leakage y de training/serving skew.

**Decisión.** Cada feature se define una vez en SQL usando solo historia con `ts` estrictamente
anterior. La transacción actual no cuenta en su propia velocity. Offline se materializa para toda
la tabla; online se calcula para una sola transacción. Un test de paridad compara ambas.

**Alternativas.** Feature store (Feast): resuelve lo mismo con más infraestructura de la que
necesitamos. Features en pandas: más flexible, pero pierde el objetivo de SQL y duplica lógica con
la API.

**Consecuencias.** Algunas features pueden ser lentas en SQL; si ocurre, se mide y se documenta la
alternativa elegida.

---

## 005 — Validación temporal

**Contexto.** El fraude ocurre en campañas y las features dependen del tiempo. Un split aleatorio
mezcla una misma campaña y una misma tarjeta entre train y test.

**Decisión.** Train 12 meses (TimeSeriesSplit con gap para tuning) → validation 3 meses
(calibración y thresholds) → test 3 meses (una sola evaluación) → producción simulada 6 meses.

**Alternativas.** `train_test_split` aleatorio estratificado: lo usaremos solo para *medir* cuánto
sobreestima (hipótesis H7).

**Consecuencias.** Menos datos para entrenar que con CV aleatoria; resultados más pesimistas y más
creíbles.

---

## 006 — PR-AUC como métrica de selección

**Contexto.** Base rate esperada ~1%.

**Decisión.** Seleccionar modelos por PR-AUC (Average Precision); decidir la política por coste
esperado en €.

**Alternativas.** Accuracy (un modelo que aprueba todo tendría ~99%); ROC-AUC (se reporta, pero con
desbalance extremo puede ser alta con precisión pobre, porque los verdaderos negativos dominan la
tasa de falsos positivos).

**Consecuencias.** Hay que comunicar PR-AUC junto a la base rate para que sea interpretable.

---

## 007 — Política de decisión optimizada por coste dependiente del importe

**Contexto.** El objetivo es minimizar coste, no errores. Dejar pasar un fraude de 2 € y uno de
2.000 € no cuesta lo mismo, y la capacidad de revisión es limitada.

**Decisión.** Modelo de coste con supuestos explícitos en `configs/costs.yaml`. Comparar thresholds
globales (P1) con decisión por transacción basada en coste esperado (P2), ambas con restricción de
capacidad, elegidas en validation.

**Alternativas.** Costes planos por tipo de error (más simple, menos realista); threshold fijo 0,5
(sin justificación con base rate del 1%).

**Consecuencias.** P2 exige probabilidades calibradas. Los resultados dependen de supuestos →
análisis de sensibilidad obligatorio.

---

## 008 — Sin SMOTE por defecto

**Contexto.** Clase minoritaria ~1%.

**Decisión.** Usar `class_weight` y threshold tuning. SMOTE solo si se demuestra necesario.

**Alternativas.** SMOTE / undersampling.

**Consecuencias.** Evitamos generar transacciones sintéticas incoherentes (interpolar dos
historias de tarjeta distintas) y distorsionar las probabilidades que necesitamos calibradas.

---

## 009 — Stack mínimo

**Decisión.** Python, PostgreSQL, scikit-learn, FastAPI, MLflow, Docker, pytest, ruff, GitHub
Actions. Sin Kafka, Airflow, Spark, Kubernetes, feature store ni DVC.

**Motivo.** Cada una de esas herramientas resuelve un problema de escala u operación que este
proyecto no tiene; añadirlas sumaría complejidad sin sumar aprendizaje sobre los problemas
centrales (leakage, coste, calibración, drift).

---

## 010 — ¿Gradient Boosting como modelo final?

**Estado.** Aceptada en la Fase 7, con evidencia:

- Fase 6 (validación, comparación pareada por días): gradient boosting supera a regresión
  logística en +0,024 de PR-AUC (IC +0,015 a +0,033) y a random forest en +0,026.
- Fase 7 (CV temporal): 0,803 frente a 0,786 de la regresión logística ajustada.

La ventaja es real pero modesta. Regla de la decisión original: si la diferencia en coste
(Fase 8) resultara irrelevante, se preferiría la regresión logística por simplicidad. Se
mantiene como alternativa interpretable.

---

## 011 — Data Quality: rechazar, reparar o avisar

**Contexto.** El extracto `raw` trae duplicados, valores inválidos, referencias rotas y
valores extremos. Borrar todo lo sospechoso es fácil, pero elimina datos reales (y fraude).

**Decisión.** Cada regla tiene una acción explícita en `dq.rules`:
- `reject` cuando no se puede saber cuál es el valor correcto (importe negativo, tarjeta de
  otro cliente, mismo `transaction_id` con dos contenidos distintos).
- `repair` solo cuando la corrección es **determinista**: `customer_id` nulo → dueño de la
  tarjeta; `eur` → `EUR`; país de un pago presencial → país del comercio; device
  "unknown" → `NULL`.
- `warn` para valores raros pero posibles: importes extremos y devices compartidos.

**Alternativas.** Rechazar todo lo anómalo (pierde pagos reales); imputar valores probables
como `EURO` → `EUR` (adivinar no es una regla).

**Consecuencias.** Las transacciones online pueden tener `device_id = NULL` (device
desconocido): se relajó el CHECK de `core.transactions`. Los outliers se conservan: en la
ejecución completa tienen un 7,49% de fraude real frente al 0,99% del resto.

---

## 012 — Clientes sin país: rechazo en cascada

**Contexto.** 30 clientes llegan sin país de residencia. El país es la base de todas las
features geográficas (cross-border, cambio de país).

**Decisión.** Rechazar el cliente y, en cascada, sus tarjetas, transacciones y etiquetas.

**Alternativas.** Permitir país nulo en `core.customers` y que las features lo traten como
desconocido; o inferirlo de sus transacciones (usaría información posterior al alta: leakage).

**Consecuencias.** Se pierden 3.094 transacciones válidas (0,11%). La tasa de fraude de lo
excluido (0,953%) es similar a la de lo conservado (0,995%), así que la limpieza no sesga el
dataset. Si en producción el problema fuera frecuente, la alternativa de país nulo sería mejor.

---

## 013 — Simulador v1.0.1: corrección de edad mínima

**Contexto.** La regla `CUS_UNDERAGE_ACCOUNT` encontró 448 clientes que abrían la cuenta hasta
12 horas antes de cumplir 18 años. No era un defecto inyectado: era un bug del simulador, que
calculaba 18 años como 18 × 365,25 días en lugar de 18 años de calendario.

**Decisión.** Corregir el generador, subir la versión a 1.0.1 y añadir un test unitario.

**Consecuencias.** Cambia el hash del dataset (`a16b1bf8…` → `350d74bd…`). Los informes de
ambas versiones se conservan en `reports/simulation/`. Es un buen ejemplo de por qué la
calidad de datos se valida con reglas de negocio y no solo con los defectos esperados.

---

## 014 — Artefacto conocido: transacciones en los bordes del periodo

**Contexto.** El generador recorta (`clip`) los timestamps al periodo simulado. Las
transacciones legítimas cuya hora local cae fuera del periodo quedan apiladas en el primer y
el último segundo: 39 en `2024-07-01 00:00:00` y 54 en `2026-06-30 23:59:59` (0,003%).

**Decisión.** Documentarlo y no corregirlo ahora. Corregirlo exige una nueva versión del
simulador, regenerar y volver a pasar la calidad de datos, y cambiaría números ya publicados
en los PRs de las Fases 2 y 3 por un efecto despreciable.

**Consecuencias.** Si más adelante hace falta otra versión del simulador por otro motivo, se
corregirá a la vez (remuestrear en lugar de recortar). Las features de velocity de comercio en
esos dos segundos pueden ver un pico artificial.

---

## 015 — EDA solo sobre el periodo de entrenamiento

**Contexto.** El EDA decide qué features construir y cómo. Si mira datos de validación, test o
producción, esas decisiones quedan informadas por los datos con los que luego se evalúa el
modelo (*data snooping*): el resultado del test deja de ser una estimación honesta.

**Decisión.** El análisis exploratorio usa solo `train` (2024-07-01 → 2025-06-30, definido en
`configs/splits.yaml`). De los demás periodos solo se muestran recuentos de filas.

**Alternativas.** EDA sobre todo el histórico (más datos, pero contamina la evaluación).

**Consecuencias.** El drift diseñado (2026) no se ha mirado en el EDA: lo tendrá que
descubrir el monitoring de la Fase 12, como en un despliegue real.

---

## 016 — Warm-up de 90 días excluido del entrenamiento

**Contexto.** El extracto empieza el 2024-07-01 para todos los clientes, aunque muchos llevan
años operando. En los primeros meses todo parece "nuevo": en julio de 2024 el 76,8% de las
transacciones tienen `merchant_new_for_customer = true`, frente al 24,5% en junio de 2025.
Es un sesgo de historia truncada, no un cambio de comportamiento.

**Decisión.** Los modelos se entrenan con transacciones desde el 2024-10-01
(`warmup_days: 90` en `configs/features.yaml`). Las features se siguen calculando desde el
inicio para que la historia exista cuando empieza el entrenamiento.

**Alternativas.** Ventanas acotadas (p. ej. "comercio no visto en 180 días"): estabilizan la
feature, pero exigen 180 días de warm-up y pierden información.

**Consecuencias.** Se pierden 3 meses de entrenamiento. `merchant_new_for_customer` sigue
derivando a los 12 meses: se vigilará en monitoring (Fase 12).

---

## 017 — Features offline con window functions, online con lookups + test de paridad

**Contexto.** El diseño pedía una sola definición por feature. En PostgreSQL la forma eficiente
de calcular 2,85M filas (window functions) no sirve para una sola transacción en la API, y la
forma natural para una transacción (lookups LATERAL) es lenta para todo el histórico.

**Decisión.** Offline: window functions con `RANGE ... EXCLUDE GROUP` donde se pueda y LATERAL
solo para conteos distintos, medianas y etiquetas. Online (Fase 10): lookups por transacción.
La equivalencia se garantiza con tests: escenario construido a mano con valores esperados,
test de viaje en el tiempo (borrar el futuro no cambia el pasado) y, en la Fase 10, un test de
paridad offline/online.

**Alternativas.** Feature store; o lookups LATERAL también offline (una sola implementación,
pero un build mucho más lento).

**Consecuencias.** Dos implementaciones que pueden divergir: el test de paridad es obligatorio
en la Fase 10.

---

## 018 — Sin pesos de clase ni remuestreo

**Contexto.** Con un 0,73–0,85% de fraude, la respuesta habitual es `class_weight="balanced"`
o remuestreo (SMOTE, undersampling).

**Evidencia (validación, Fase 6).** Con pesos balanceados, el PR-AUC no mejora (gradient
boosting −0,002, no significativo) o empeora (regresión logística −0,046, IC pareado −0,058 a
−0,035), y las probabilidades medias pasan de 0,0072 (observado 0,0073) a 0,068 y 0,123.

**Decisión.** Entrenar sin pesos ni remuestreo. El desbalance se gestiona en la decisión
(umbral por coste y capacidad, Fase 8), que necesita probabilidades calibradas.

**Alternativas.** SMOTE: además de distorsionar las probabilidades igual que los pesos, crea
transacciones sintéticas interpolando entre tarjetas y clientes sin relación.

**Consecuencias.** Las probabilidades del modelo salen ya casi calibradas de media; la Fase 8
comprueba la calibración por tramos.

---

## 019 — Modelos guardados con skops y lista explícita de tipos de confianza

**Contexto.** MLflow 3 guarda los modelos de scikit-learn con skops. Un pickle puede ejecutar
código arbitrario al cargarse; skops solo reconstruye tipos que se declaran de confianza.

**Decisión.** Mantener skops y declarar los tipos necesarios en
`pipelines.SKOPS_TRUSTED_TYPES` (nuestra función `signed_log1p` y tipos internos de
scikit-learn). Un test comprueba que cada pipeline se guarda, se recarga con esa lista y
predice exactamente lo mismo.

**Alternativas.** `serialization_format="cloudpickle"`: funciona sin lista, pero cargar un
modelo pasa a ser ejecutar código de un fichero.

**Consecuencias.** Cualquier transformador nuevo obliga a ampliar la lista; el test lo detecta.

---

## 020 — Modelo seleccionado por CV temporal; validación solo confirma

**Contexto.** La búsqueda aleatoria (16 configuraciones × 4 folds temporales) encuentra un
gradient boosting con 0,803 de PR-AUC en CV frente a 0,797 de la configuración por defecto.
En validación la diferencia es +0,002 (IC pareado −0,004 a +0,008): no significativa.

**Decisión.** Seleccionar la configuración con mejor CV temporal (`selected_model` en
`configs/model.yaml`) sin elegir nada mirando validación. Validación no la contradice y no
cuesta más.

**Alternativas.** Volver a los valores por defecto (igual de defendible: la mejora está dentro
del ruido); elegir por validación (sería ajustar hiperparámetros con validación y convertirla
en otro conjunto de entrenamiento).

**Consecuencias.** La variación entre meses (0,759 a 0,825 en CV) es mayor que cualquier
ganancia de tuning: la incertidumbre por mes debe aparecer en los informes y en el monitoring.

---

## 021 — Política por coste esperado por transacción (P2)

**Contexto.** Dos formas de convertir la probabilidad en decisión: umbrales globales
(P1: revisar si p ≥ 4,4%, bloquear si p ≥ 25%) o elegir para cada transacción la acción de
menor coste esperado dado p y el importe (P2).

**Evidencia (validación).** P2 €62.925, P1 €63.742, bloquear si p ≥ 0,5 €95.162, reglas
€520.946, aprobar todo €946.616. En test, P2 €64.098 y P1 €66.962.

**Decisión.** Desplegar P2. Usa el importe (un pago de €1.000 merece revisión con p = 0,55%,
uno de €5 no hasta p = 26%) y se adapta solo si cambian los costes.

**Alternativas.** P1 es más fácil de explicar y solo un 1,3% más cara en validación; sería la
opción si negocio necesitara umbrales fijos y comunicables.

**Consecuencias.** P2 necesita probabilidades calibradas: la calibración pasa a ser un
requisito vigilado en monitoring, no un detalle.

---

## 022 — El test se evalúa una sola vez y queda registrado en el repositorio

**Contexto.** Si el test se mira, se ajusta algo y se vuelve a mirar, deja de ser una
estimación honesta del rendimiento futuro: se convierte en otro conjunto de validación.

**Decisión.** La política, el calibrador y los hashes del modelo se congelan y se suben al
repositorio antes de evaluar (`32ce00c`). La evaluación (`fraud policy evaluate-test`) escribe
`reports/policy/test_evaluation.json`, que también se sube (`3b95790`); si ese fichero existe,
el comando se niega a ejecutarse otra vez salvo con `--force`, que debe justificarse aquí.

**Resultado.** PR-AUC 0,752 en test frente a 0,792 en validación; coste de la política
€64.098 frente a €946.956 de aprobar todo. Se publica tal cual.

**Consecuencias.** El historial de git demuestra el orden: primero la política congelada,
después el test.

---

## 023 — Explicaciones con SHAP PermutationExplainer y comprobación de aditividad

**Contexto.** `shap.TreeExplainer` es el método rápido y exacto para modelos de árboles. Con
`HistGradientBoostingClassifier` y splits categóricos nativos devuelve valores erróneos sin
avisar: en este modelo las contribuciones no suman la salida del modelo (error de hasta 4,6
log-odds en todas las filas). Trata los splits categóricos como umbrales numéricos. Con un
modelo sin splits categóricos el error es 1e-14.

**Decisión.** Usar `shap.PermutationExplainer` (agnóstico al modelo, exacto por construcción,
más lento) y comprobar en cada ejecución que base + contribuciones = log-odds del modelo
(error máximo obtenido: 4,1e-14). La importancia global se mide además con permutación, por
variable y por familia de variables.

**Alternativas.** Reentrenar sin splits categóricos nativos para poder usar TreeExplainer
(explicaría otro modelo, no el desplegado); usar solo importancia por permutación (no da
explicaciones individuales).

**Consecuencias.** Explicar 900 transacciones cuesta ~4 minutos. Para la API (Fase 10),
explicar cada predicción en tiempo real sería caro: se valorará devolver solo las 3 variables
principales o explicar bajo demanda.
