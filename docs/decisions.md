# Registro de decisiones técnicas

Formato: contexto → decisión → alternativas → consecuencias. Una decisión aceptada solo se cambia
con una nueva entrada que la sustituya (no se reescribe la historia).

| ID | Decisión | Estado |
|---|---|---|
| 001 | PostgreSQL como base de datos | Propuesta |
| 002 | Dataset sintético por escenarios | Propuesta |
| 003 | Etiquetas con fecha de reporte | Propuesta |
| 004 | Features point-in-time en SQL compartidas offline/online | Propuesta |
| 005 | Validación temporal | Propuesta |
| 006 | PR-AUC como métrica de selección | Propuesta |
| 007 | Política de decisión optimizada por coste dependiente del importe | Propuesta |
| 008 | Sin SMOTE por defecto | Propuesta |
| 009 | Stack mínimo | Propuesta |
| 010 | ¿Gradient Boosting como modelo final? | **Pendiente de evidencia (Fase 7)** |

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

**Estado.** Pendiente. Se decidirá con los resultados de la Fase 7, no antes. Si LR o RF resultan
equivalentes en coste, se preferirá el más simple y se documentará.
