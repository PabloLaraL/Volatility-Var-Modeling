# 📈 Modelado de Volatilidad y Value-at-Risk (VaR) / Expected Shortfall (ES)

## Framework ARIMA–GARCH con Backtesting Formal y ES Dinámico

---

## 🔎 Descripción General

Este repositorio implementa un pipeline estructurado de modelamiento de volatilidad condicional y medición de riesgo de mercado.

El framework integra:

- Construcción de **retornos logarítmicos**
- Modelado de la **media condicional** mediante ARIMA
- Modelos de volatilidad de la **familia GARCH**
- Estimación de **Value-at-Risk (VaR)**:
  - Paramétrico condicional
  - Histórico
  - Monte Carlo
- Backtesting estadístico formal:
  - Kupiec
  - Christoffersen
- Estimación de **Expected Shortfall (ES)**:
  - Histórico
  - Paramétrico dinámico (Normal y t-Student)
  - Simulación Monte Carlo

**Caso de estudio:** Banco de Chile (CHILE.SN), datos diarios desde 2015.

El objetivo es demostrar de forma estructurada cómo conectar modelos ARIMA–GARCH con métricas modernas de riesgo.

---

# 🧠 Hechos Estilizados de Series Financieras

Las series financieras presentan:

- Retornos aproximadamente estacionarios  
- Clustering de volatilidad  
- Heterocedasticidad condicional  
- Colas pesadas  
- Asimetría ante shocks negativos  

El framework modela explícitamente estas características.

---

# ⚙️ Metodología

---

## 1️⃣ Construcción de Retornos

Se utilizan precios ajustados para evitar distorsiones por dividendos y splits.

A partir de ellos se calculan retornos logarítmicos.

---

## 2️⃣ Modelado de la Media — ARIMA

Se estima la media condicional mediante ARIMA.

Diagnósticos aplicados:

- ACF / PACF  
- Test de Ljung–Box  

---

## 3️⃣ Modelos de Volatilidad — Familia GARCH

Se verifica heterocedasticidad mediante el test ARCH-LM.

Modelos implementados:

- GARCH(1,1)
- EGARCH(1,1)
- GJR-GARCH(1,1)

Distribuciones consideradas:

- Normal
- Student-t
- Skew-t

---

# 📉 Value-at-Risk (VaR)

Se implementan tres enfoques:

### 🔹 VaR Paramétrico Condicional
Basado en media y volatilidad condicional.

### 🔹 VaR Histórico
Cuantil empírico sobre ventana móvil.

### 🔹 VaR Monte Carlo
Simulación de escenarios bajo el modelo estimado.

---

# 🧪 Backtesting del VaR

Validación formal mediante:

- Kupiec (cobertura incondicional)
- Christoffersen (independencia)
- Cobertura condicional conjunta

Incluye visualización de violaciones y evaluación dinámica.

---

# 📊 Expected Shortfall (ES)

Se implementan enfoques comparativos de ES:

- ES Histórico
- ES Paramétrico Dinámico (Normal y t-Student)
- ES por Simulación (Monte Carlo)

Además, se incluye visualización en serie temporal comparando:

- Retornos
- VaR
- ES (97.5%)

El ES complementa al VaR capturando la severidad esperada en escenarios extremos.

---

# 🎯 Enfoque del Proyecto

Proyecto con fines educativos y analíticos.

Busca ilustrar:

- Modelado de volatilidad condicional
- Estimación coherente de VaR
- Validación estadística formal
- Integración de ES bajo un enfoque dinámico

No pretende replicar un motor regulatorio completo, sino mostrar fundamentos técnicos de medición de riesgo de mercado.

---

# ⚠️ Disclaimer

Proyecto con fines académicos.  
No constituye recomendación de inversión.
