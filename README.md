# 📈 Modelado de Volatilidad y Value-at-Risk (VaR) / Expected Shortfall (ES)

## Framework ARIMA–GARCH con Backtesting Estadístico Formal + ES Dinámico (97.5%)

---

## 🔎 Descripción General

Este repositorio implementa un **pipeline profesional y didáctico de riesgo de mercado** orientado al modelamiento de volatilidad condicional y a la **validación formal de modelos de Value-at-Risk (VaR)**, incorporando además **Expected Shortfall (ES)** bajo un enfoque dinámico consistente con volatilidad condicional.

El framework integra:

- Transformación de precios ajustados a **retornos logarítmicos**
- Modelado de la **media condicional** mediante ARIMA
- Modelos de volatilidad de la **familia GARCH**
- Estimación de VaR:
  - **Paramétrico condicional** (ARIMA + GARCH-family + cuantiles)
  - **Histórico** (cuantil empírico sobre ventana móvil)
  - **Monte Carlo** (simulación bajo el modelo condicional)
- Backtesting estadístico formal:
  - **Kupiec** (Unconditional Coverage)
  - **Christoffersen** (Independence / Conditional Coverage)
- Estimación de **Expected Shortfall (ES)**:
  - **Histórico**
  - **Dinámico (serie temporal)** bajo ARIMA + GJR-GARCH con innovaciones t-Student (ES 97.5%)

**Caso de estudio:** Banco de Chile (CHILE.SN), datos diarios desde 2015.

---

# 🧠 Hechos Estilizados de Series de Tiempo Financieras

Las series financieras presentan propiedades empíricas ampliamente documentadas:

- Precios no estacionarios
- Retornos aproximadamente estacionarios
- Clustering de volatilidad
- Heterocedasticidad condicional
- Colas pesadas (fat tails)
- Alta persistencia de volatilidad
- Asimetría ante shocks negativos (efecto leverage)

Este proyecto modela explícitamente estas características y evalúa el desempeño del VaR bajo criterios estadísticos formales, incorporando ES como medida complementaria de severidad en cola.

---

# ⚙️ Metodología

---

## 1️⃣ Datos y Construcción de Retornos

Se utilizan precios ajustados (Adjusted Close) para evitar distorsiones por dividendos y splits.

A partir de ellos se construyen **retornos logarítmicos**, que constituyen la base para el modelamiento posterior.

---

## 2️⃣ Modelado de la Media (ARIMA)

La media condicional se modela utilizando un proceso **ARIMA(p,d,q)**.

Diagnósticos aplicados:

- ACF / PACF
- Test de Ljung–Box sobre residuos

El objetivo es aislar las **innovaciones** (residuos) para modelar sobre ellas la dinámica de volatilidad condicional.

---

## 3️⃣ Modelos de Volatilidad Condicional (Familia GARCH)

Antes de estimar modelos, se verifica la presencia de heterocedasticidad mediante:

- **ARCH-LM test** (detección de efectos ARCH)

Modelos implementados:

### 🔹 GARCH(1,1)

Modelo base para capturar:

- Clustering de volatilidad  
- Persistencia de la varianza  

---

### 🔹 EGARCH(1,1)

Extensión que permite:

- Modelar **asimetría** (shocks negativos impactan distinto que positivos)  
- Evitar restricciones de positividad al modelar en escala logarítmica  

---

### 🔹 GJR-GARCH(1,1)

Modelo diseñado para:

- Capturar explícitamente el **leverage effect** mediante un término indicador para shocks negativos  

---

### 🔹 Supuestos Distribucionales

Las innovaciones estandarizadas se estiman bajo supuestos que permiten capturar colas pesadas:

- Normal
- Student-t

> Nota: En el notebook se enfatiza Student-t para reflejar colas pesadas en retornos financieros.

---

## 4️⃣ Forecast de Volatilidad

El framework produce pronósticos de volatilidad (condicional) que se utilizan como entrada para la estimación de VaR y ES condicionales (forward-looking).

---

# 📉 Value-at-Risk (VaR)

---

## 🔹 VaR Paramétrico Condicional

Estimación basada en:

- Media condicional (ARIMA)
- Volatilidad condicional pronosticada (GARCH-family)
- Cuantiles según la distribución asumida (Normal / Student-t)

---

## 🔹 VaR Histórico

Estimado mediante:

- Cuantil empírico sobre ventana móvil (por ejemplo 250 días)

---

## 🔹 VaR Monte Carlo

Estimación por simulación de escenarios:

- Generación de shocks simulados desde la distribución estimada
- Construcción de retornos simulados con media y volatilidad condicional
- Cálculo del cuantil empírico de las pérdidas simuladas

---

# 🧪 Backtesting Estadístico (VaR)

Se define una violación cuando el retorno observado es menor que el VaR estimado (cola izquierda) al nivel de significancia elegido.

Validación formal:

- **Kupiec (Unconditional Coverage):** evalúa si la tasa de violaciones coincide con la esperada
- **Christoffersen (Independence):** evalúa independencia temporal de violaciones
- **Conditional Coverage:** evaluación conjunta de cobertura e independencia

Diagnósticos:

- Visualización de violaciones
- Evaluación dinámica (hit-rate / métricas en ventana móvil)

---

# 📊 Expected Shortfall (ES)

El **Expected Shortfall (ES)** captura la **severidad esperada** de las pérdidas extremas.

En el proyecto se implementa:

## 🔹 ES Histórico

Estimado como el promedio de retornos en la cola izquierda (más allá del VaR) usando la distribución empírica.

## 🔹 ES Dinámico (Serie Temporal) — ARIMA + GJR-GARCH(t)

Bajo un modelo condicional, ES evoluciona en el tiempo al depender de \(\sigma_t\):

- \( r_t = \mu_t + \sigma_t z_t \)
- \( ES_t = \mu_t + \sigma_t c_\alpha \)

Se incluye una visualización **serie temporal** comparando:

- Retornos
- **VaR 97.5%**
- **ES 97.5%**

> Nota: VaR se backtestea mediante violaciones (hits). ES no se evalúa mediante “violaciones” directas, ya que ES es una media condicional de cola.

---

# 🔮 Extensiones Potenciales

- Backtesting conjunto VaR–ES (enfoques avanzados)
- Validación out-of-sample formal
- Modelos multivariados (correlación dinámica)
- Stress testing estructural / escenarios

---

# 📌 Enfoque Profesional

Este proyecto replica un workflow utilizado en análisis de **Riesgo de Mercado** para:

- Modelar volatilidad condicional de retornos
- Estimar pérdidas potenciales (VaR) bajo supuestos realistas (colas pesadas / asimetría)
- Validar estadísticamente VaR mediante backtesting formal
- Incorporar ES como medida complementaria de severidad en cola bajo un enfoque dinámico

Diseñado como framework **reproducible y extensible**.

---

# ⚠️ Disclaimer

Proyecto con fines académicos y de investigación.  
No constituye recomendación de inversión.
