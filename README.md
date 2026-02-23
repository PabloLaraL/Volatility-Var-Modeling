# 📈 Modelado de Volatilidad y Value-at-Risk (VaR)

## Framework ARIMA–GARCH con Backtesting Estadístico Formal

---

## 🔎 Descripción General

Este repositorio implementa un **pipeline profesional de riesgo de mercado** orientado al modelamiento de volatilidad condicional y a la **validación formal de modelos de Value-at-Risk (VaR)**.

El framework integra:

- Transformación de precios ajustados a **retornos logarítmicos**
- Modelado de la **media condicional** mediante ARIMA
- Modelos de volatilidad de la **familia GARCH**
- Estimación de VaR: **Paramétrico Condicional**, **Histórico** y **Monte Carlo**
- Backtesting formal: **Kupiec** (Unconditional Coverage) y **Christoffersen** (Independence / Conditional Coverage)
- Diagnóstico dinámico mediante **rolling hit-rate**

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

Este proyecto modela explícitamente estas características y evalúa el desempeño del VaR bajo criterios estadísticos formales.

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

Las innovaciones estandarizadas se estiman bajo distintos supuestos para capturar colas pesadas y asimetría:

- Normal
- Student-t
- Skew-t

---

## 4️⃣ Forecast de Volatilidad

El framework produce pronósticos de volatilidad a distintos horizontes, que se utilizan como entrada para la estimación de VaR condicional (forward-looking).

---

# 📉 Value-at-Risk (VaR)

---

## 🔹 VaR Paramétrico Condicional

Estimación basada en:

- Media condicional (ARIMA)
- Volatilidad condicional pronosticada (GARCH-family)
- Cuantiles según la distribución asumida (Normal / Student-t / Skew-t)

---

## 🔹 VaR Histórico

Estimado mediante:

- Cuantil empírico sobre ventana móvil (por ejemplo 250 días)

---

## 🔹 VaR Monte Carlo

Estimación por simulación de escenarios:

- Generación de shocks simulados desde la distribución estimada
- Construcción de retornos simulados con media y volatilidad condicional
- Extensión multivariada disponible mediante descomposición de Cholesky para correlaciones

---

# 🧪 Backtesting Estadístico

Se define una violación cuando el retorno observado supera el VaR estimado al nivel de significancia elegido.

Validación formal:

- **Kupiec (Unconditional Coverage):** evalúa si la tasa de violaciones coincide con la esperada
- **Christoffersen (Independence):** evalúa independencia temporal de violaciones
- **Conditional Coverage:** evaluación conjunta de cobertura e independencia

Diagnósticos:

- Visualización de violaciones
- **Rolling hit-rate** (ej. ventana móvil 250 días)

---

# 🔮 Extensiones Potenciales

- Expected Shortfall (Basilea III)
- DCC-GARCH (multivariado dinámico)
- Modelos de cambio de régimen
- Stress testing estructural
- Validación out-of-sample formal

---

# 📌 Enfoque Profesional

Este proyecto replica workflows utilizados en equipos de **Riesgo de Mercado** para:

- Modelar volatilidad condicional de retornos
- Estimar pérdidas potenciales (VaR) bajo supuestos realistas (colas pesadas / asimetría)
- Validar estadísticamente el modelo mediante backtesting formal

Diseñado como framework **reproducible y extensible**.

---

Disclaimer

Proyecto con fines académicos y de investigación.
No constituye recomendación de inversión.
