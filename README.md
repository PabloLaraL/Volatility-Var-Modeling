Modelado de Volatilidad y Value-at-Risk (VaR)
Framework ARIMA–GARCH con Backtesting Estadístico (Kupiec & Christoffersen)
Resumen

Este repositorio implementa un pipeline completo de riesgo de mercado basado en volatilidad condicional, integrando:

Transformación de precios a retornos logarítmicos

Diagnóstico de estacionariedad (ADF) y estructura temporal (ACF/PACF)

Filtrado de la media mediante ARIMA

Detección de heterocedasticidad condicional (ARCH-LM)

Estimación de volatilidad con GARCH / EGARCH / GJR-GARCH

Comparación de distribuciones: Normal / Student-t / Skew-t

Forecast de volatilidad (horizonte H)

Cálculo de VaR:

Paramétrico condicional

Histórico (rolling window)

Monte Carlo (univariado y multivariado)

Validación formal mediante Backtesting:

Kupiec (Unconditional Coverage)

Christoffersen (Independence)

Conditional Coverage

Caso de estudio: Banco de Chile (CHILE.SN) con datos diarios desde 2015.

1. Motivación (Hechos Estilizados)

Las series financieras típicamente presentan:

Precios no estacionarios

Retornos estacionarios en media

Volatility clustering

Heterocedasticidad condicional

Colas pesadas

Alta persistencia de volatilidad

Posible asimetría ante shocks negativos (leverage effect)

Este proyecto modela explícitamente estas propiedades y valida estadísticamente la calidad del VaR estimado.

2. Metodología
2.1 Datos y retornos

Se parte de precios diarios P_t y se construyen retornos logarítmicos:

r_t = ln(P_t / P_{t-1})

Buenas prácticas: se utiliza Adj Close para evitar saltos artificiales producto de dividendos y splits.

2.2 Filtrado de media (ARIMA)

Se modela la media condicional para remover dependencia lineal:

r_t = mu_t + epsilon_t

donde epsilon_t son innovaciones.

Diagnóstico aplicado:

ACF/PACF

Test de Ljung–Box sobre residuos

2.3 Volatilidad condicional (GARCH-family)

Se evalúa evidencia ARCH mediante ARCH-LM y se estiman modelos de volatilidad.

GARCH(1,1)

sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

Captura persistencia y clustering de volatilidad.

EGARCH(1,1)

log(sigma_t^2) = omega + beta * log(sigma_{t-1}^2) + alpha * |z_{t-1}| + gamma * z_{t-1}

Permite modelar asimetría sin restricciones de positividad.

GJR-GARCH(1,1)

sigma_t^2 = omega
+ alpha * epsilon_{t-1}^2
+ gamma * epsilon_{t-1}^2 * I(epsilon_{t-1} < 0)
+ beta * sigma_{t-1}^2

Modela efecto leverage (mayor impacto de shocks negativos).

Distribuciones consideradas para z_t:

Normal

Student-t

Skew-t

2.4 Forecast de volatilidad

Se estima sigma_hat_{t+h} para un horizonte H, observando:

Convergencia a volatilidad de largo plazo

Persistencia de shocks

Diferencias entre especificaciones

3. Value-at-Risk (VaR)
3.1 VaR paramétrico condicional (ARIMA + GARCH)

VaR_{t+1}(alpha) = mu_{t+1} + q_alpha * sigma_{t+1}

donde q_alpha es el cuantil de la distribución asumida.

3.2 VaR histórico (rolling window)

Cuantil empírico sobre ventana móvil (ej. 250 días).

3.3 VaR Monte Carlo

Simulación de retornos:

r_{t+1}^{sim} = mu_{t+1} + sigma_{t+1} * z^{sim}

y cálculo del cuantil empírico de la distribución simulada.

Incluye extensión multivariada mediante descomposición de Cholesky para shocks correlacionados.

4. Backtesting

Se define violación cuando:

r_t < VaR_t

Se aplican:

Kupiec (Unconditional Coverage)

Christoffersen (Independence)

Conditional Coverage

Además se incluyen:

Visualización de violaciones

Rolling hit rate (ventana de 250 días)

5. Cómo ejecutar
Opción A — Notebook (recomendado)

pip install -r requirements.txt
jupyter notebook notebook.ipynb

El notebook se encuentra ejecutado e incluye outputs y visualizaciones.

Opción B — Script reproducible

python src/run_pipeline.py

Permite correr el pipeline completo en modo batch.

6. Estructura del repositorio

notebook.ipynb → Notebook ejecutado con outputs y visualizaciones

src/run_pipeline.py → Pipeline reproducible en modo CLI

requirements.txt → Dependencias

7. Limitaciones y extensiones

Limitaciones:

Modelo univariado

Correlación estática en bloque multivariado

No incluye Expected Shortfall

Extensiones naturales:

Expected Shortfall (Basel III)

DCC-GARCH

Regime-switching models

Stress testing

Validación fuera de muestra

Disclaimer

Proyecto con fines académicos y de investigación.
No constituye recomendación de inversión.
