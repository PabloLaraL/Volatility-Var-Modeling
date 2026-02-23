# 📈 Modelado de Volatilidad y Value-at-Risk (VaR)

## Framework ARIMA–GARCH con Backtesting Estadístico Formal

---

## 🔎 Descripción General

Este repositorio implementa un **pipeline profesional de riesgo de mercado** basado en modelamiento de volatilidad condicional y validación estadística formal de Value-at-Risk (VaR).

El framework integra:

- Transformación de precios ajustados a retornos logarítmicos  
- Modelado de la media condicional mediante ARIMA  
- Modelos de volatilidad de la familia GARCH  
- Estimación de VaR (Paramétrico Condicional, Histórico y Monte Carlo)  
- Backtesting formal (Kupiec y Christoffersen)  
- Diagnóstico dinámico mediante rolling hit-rate  

**Caso de estudio:** Banco de Chile (CHILE.SN), datos diarios desde 2015.

---

# 🧠 Hechos Estilizados de Series de Tiempo Financieras

Las series de retornos financieros presentan propiedades empíricas bien documentadas:

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

## 1️⃣ Construcción de Retornos

Se utilizan precios ajustados (Adjusted Close) para evitar distorsiones por dividendos y splits.

Los retornos logarítmicos se calculan como:

\[
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
\]

donde \( P_t \) corresponde al precio ajustado en el tiempo \( t \).

---

## 2️⃣ Modelado de la Media: ARIMA

La dinámica de la media condicional se modela mediante un proceso ARIMA(p,d,q):

\[
\Phi(L)(1-L)^d r_t = \Theta(L)\varepsilon_t
\]

donde:

- \( \Phi(L) \) y \( \Theta(L) \) son polinomios en el operador rezago  
- \( \varepsilon_t \) son innovaciones  

Diagnósticos aplicados:

- ACF / PACF  
- Test de Ljung–Box sobre residuos  

El objetivo es aislar innovaciones \( \varepsilon_t \) para modelar su varianza condicional.

---

## 3️⃣ Modelos de Volatilidad Condicional

Se evalúa la presencia de efectos ARCH mediante el test ARCH-LM antes de estimar modelos de volatilidad.

---

### 🔹 GARCH(1,1)

\[
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2
\]

Captura:

- Clustering de volatilidad  
- Persistencia  
- Dinámica autoregresiva de la varianza  

---

### 🔹 EGARCH(1,1)

\[
\ln(\sigma_t^2) = \omega + \beta \ln(\sigma_{t-1}^2)
+ \alpha \frac{\varepsilon_{t-1}}{\sigma_{t-1}}
+ \gamma \left( \left| \frac{\varepsilon_{t-1}}{\sigma_{t-1}} \right| - E|z| \right)
\]

Permite modelar asimetría sin imponer restricciones de positividad.

---

### 🔹 GJR-GARCH(1,1)

\[
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2
+ \gamma I_{\{\varepsilon_{t-1}<0\}}\varepsilon_{t-1}^2
+ \beta \sigma_{t-1}^2
\]

Modela explícitamente el efecto leverage.

---

### 🔹 Supuestos Distribucionales

Las innovaciones estandarizadas se estiman bajo:

- Normal  
- Student-t  
- Skew-t  

---

## 4️⃣ Forecast de Volatilidad

El modelo genera pronósticos multi-step de volatilidad condicional:

\[
\hat{\sigma}_{t+h}^2
\]

Estos se utilizan para estimar riesgo futuro bajo distintos horizontes.

---

# 📉 Value-at-Risk (VaR)

---

## 🔹 VaR Paramétrico Condicional

\[
VaR_{t+1}^{\alpha} = \mu_{t+1} + \sigma_{t+1} q_{\alpha}
\]

donde:

- \( \mu_{t+1} \) es la media condicional  
- \( \sigma_{t+1} \) es la volatilidad forecast  
- \( q_{\alpha} \) es el cuantil de la distribución asumida  

---

## 🔹 VaR Histórico

\[
VaR_t^{\alpha} = \text{Cuantil empírico}_{\alpha}
\]

calculado sobre una ventana móvil (ej. 250 días).

---

## 🔹 VaR Monte Carlo

Simulación de escenarios:

\[
r_{t+1}^{(i)} = \mu_{t+1} + \sigma_{t+1} z^{(i)}
\]

donde \( z^{(i)} \) son shocks simulados.

Extensión multivariada mediante descomposición de Cholesky para correlaciones.

---

# 🧪 Backtesting Estadístico

Se define una violación cuando:

\[
r_t < VaR_t^{\alpha}
\]

Se aplican los siguientes tests:

- **Kupiec (Unconditional Coverage)**  
- **Christoffersen (Independence Test)**  
- **Conditional Coverage Test**

Diagnósticos adicionales:

- Visualización de violaciones  
- Rolling hit-rate (ventana móvil de 250 días)

---

# 🗂️ Estructura del Repositorio

```
notebook.ipynb        → Notebook ejecutado con pipeline completo
src/run_pipeline.py   → Pipeline reproducible vía CLI
requirements.txt      → Dependencias
```

---

# 🚀 Cómo Ejecutar

## Notebook (Recomendado)

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Pipeline CLI

```bash
pip install -r requirements.txt
python src/run_pipeline.py
```

---

# 🔮 Extensiones Futuras

- Expected Shortfall (Basilea III)  
- DCC-GARCH  
- Modelos de cambio de régimen  
- Stress testing estructural  
- Validación out-of-sample  

---

# 📌 Enfoque Profesional

Este proyecto replica el workflow utilizado en áreas de **Riesgo de Mercado** para:

- Modelar volatilidad condicional  
- Estimar pérdidas potenciales  
- Validar modelos bajo estándares estadísticos formales  

Diseñado como framework reproducible y extensible para aplicaciones institucionales.
