📈 Modelado de Volatilidad y Medición de Riesgo de Mercado
Framework ARIMA–GARCH con VaR, ES y Backtesting Formal
🔎 Descripción General

Este repositorio presenta un framework estructurado para el modelado de volatilidad condicional y la estimación de medidas modernas de riesgo de mercado.

El pipeline integra:

Construcción de retornos logarítmicos

Modelado de la media condicional mediante ARIMA

Modelos de volatilidad de la familia GARCH

Estimación de Value-at-Risk (VaR):

Paramétrico condicional

Histórico

Monte Carlo

Backtesting formal:

Kupiec (Cobertura Incondicional)

Christoffersen (Independencia y Cobertura Condicional)

Estimación dinámica de Expected Shortfall (ES) 97.5%

Caso de estudio: Banco de Chile (CHILE.SN), datos diarios desde 2015.

El objetivo es demostrar, de forma didáctica y reproducible, cómo conectar modelos econométricos de volatilidad con métricas modernas de riesgo utilizadas en la práctica.

🧠 Hechos Estilizados de Retornos Financieros

Las series financieras suelen presentar:

Retornos aproximadamente estacionarios

Clustering de volatilidad

Heterocedasticidad condicional

Colas pesadas

Asimetría ante shocks negativos (leverage effect)

El framework modela explícitamente estas características.

⚙️ Metodología
1️⃣ Construcción de Retornos

Se utilizan precios ajustados para evitar distorsiones por dividendos y splits.
A partir de ellos se calculan retornos logarítmicos.

2️⃣ Modelado de la Media — ARIMA

La media condicional se modela mediante un proceso ARIMA.

Diagnósticos aplicados:

ACF / PACF

Test de Ljung–Box

El objetivo es aislar las innovaciones para modelar la varianza condicional.

3️⃣ Modelos de Volatilidad — Familia GARCH

Previo a la estimación se verifica heterocedasticidad mediante el test ARCH-LM.

Modelos implementados:

GARCH(1,1)

EGARCH(1,1)

GJR-GARCH(1,1)

Se consideran distribuciones Normal y t-Student para capturar colas pesadas.

📉 Value-at-Risk (VaR)

Se implementan tres enfoques:

🔹 VaR Paramétrico Condicional

Basado en:

Media condicional (ARIMA)

Volatilidad condicional (GARCH)

Cuantiles según la distribución asumida

🔹 VaR Histórico

Cuantil empírico sobre ventana móvil.

🔹 VaR Monte Carlo

Simulación de escenarios bajo el modelo condicional estimado.

🧪 Backtesting del VaR

Se evalúa el desempeño del modelo mediante:

Kupiec: consistencia en la frecuencia de violaciones

Christoffersen: independencia temporal de violaciones

Cobertura condicional conjunta

El análisis incluye visualización de violaciones y evaluación dinámica.

📊 Expected Shortfall (ES)

Se estima:

ES Histórico

ES Dinámico 97.5% bajo el modelo condicional GJR-GARCH(t)

El ES complementa al VaR capturando la severidad esperada en la cola izquierda de la distribución.

🎯 Enfoque del Proyecto

Este trabajo tiene fines educativos y analíticos.

Busca demostrar cómo:

Modelos ARIMA–GARCH pueden capturar hechos estilizados

El VaR puede validarse formalmente mediante backtesting

El ES puede integrarse de manera coherente en un entorno dinámico

No pretende replicar un motor regulatorio bancario completo, sino ilustrar de forma técnica y estructurada los fundamentos de medición de riesgo de mercado.

⚠️ Disclaimer

Proyecto con fines académicos.
No constituye recomendación de inversión.
