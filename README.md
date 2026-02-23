Modelado de Volatilidad y Value-at-Risk (VaR)
Framework ARIMA–GARCH con Backtesting Estadístico (Kupiec & Christoffersen)
Resumen

Este repositorio implementa un pipeline completo de riesgo de mercado basado en volatilidad condicional, integrando:

Transformación de precios a retornos logarítmicos

Diagnóstico de estacionariedad (ADF) y estructura temporal (ACF/PACF)

Filtrado de la media mediante ARIMA (innovaciones / residuos)

Detección de heterocedasticidad condicional (ARCH-LM)

Estimación de volatilidad con GARCH / EGARCH / GJR-GARCH

Comparación de distribuciones: Normal / Student-t / Skew-t

Forecast de volatilidad (horizonte 
𝐻
H)

Cálculo de VaR:

Paramétrico condicional

Histórico

Monte Carlo (univariado y multivariado)

Validación formal mediante Backtesting:

Kupiec (Unconditional Coverage)

Christoffersen (Independence)

Conditional Coverage (test conjunto)

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

Se parte de precios diarios 
𝑃
𝑡
P
t
	​

 y se construyen retornos logarítmicos:

𝑟
𝑡
=
ln
⁡
(
𝑃
𝑡
𝑃
𝑡
−
1
)
r
t
	​

=ln(
P
t−1
	​

P
t
	​

	​

)

Buenas prácticas: se utiliza Adj Close para evitar saltos artificiales producto de dividendos y splits.

2.2 Filtrado de media (ARIMA)

Se modela la media condicional para remover dependencia lineal:

𝑟
𝑡
=
𝜇
𝑡
+
𝜀
𝑡
r
t
	​

=μ
t
	​

+ε
t
	​


donde 
𝜀
𝑡
ε
t
	​

 son innovaciones.

Diagnóstico:

ACF/PACF sobre retornos

Test de Ljung–Box sobre residuos

2.3 Volatilidad condicional (GARCH-family)

Se evalúa evidencia ARCH mediante ARCH-LM y se estiman modelos de volatilidad.

GARCH(1,1)
𝜎
𝑡
2
=
𝜔
+
𝛼
𝜀
𝑡
−
1
2
+
𝛽
𝜎
𝑡
−
1
2
σ
t
2
	​

=ω+αε
t−1
2
	​

+βσ
t−1
2
	​

EGARCH(1,1)
log
⁡
(
𝜎
𝑡
2
)
=
𝜔
+
𝛽
log
⁡
(
𝜎
𝑡
−
1
2
)
+
𝛼
∣
𝑧
𝑡
−
1
∣
+
𝛾
𝑧
𝑡
−
1
log(σ
t
2
	​

)=ω+βlog(σ
t−1
2
	​

)+α∣z
t−1
	​

∣+γz
t−1
	​

GJR-GARCH(1,1)
𝜎
𝑡
2
=
𝜔
+
𝛼
𝜀
𝑡
−
1
2
+
𝛾
𝜀
𝑡
−
1
2
1
{
𝜀
𝑡
−
1
<
0
}
+
𝛽
𝜎
𝑡
−
1
2
σ
t
2
	​

=ω+αε
t−1
2
	​

+γε
t−1
2
	​

1
{ε
t−1
	​

<0}
	​

+βσ
t−1
2
	​


Se comparan distribuciones para 
𝑧
𝑡
z
t
	​

:

Normal

Student-t

Skew-t

2.4 Forecast de volatilidad

Se calcula:

𝜎
^
𝑡
+
ℎ
σ
^
t+h
	​


para un horizonte 
𝐻
H, observando:

Convergencia a volatilidad de largo plazo

Diferencias entre especificaciones

Persistencia de shocks

3. Value-at-Risk (VaR)
3.1 VaR paramétrico condicional (ARIMA + GARCH)
𝑉
𝑎
𝑅
𝑡
+
1
(
𝛼
)
=
𝜇
𝑡
+
1
+
𝑞
𝛼
𝜎
𝑡
+
1
VaR
t+1
(α)
	​

=μ
t+1
	​

+q
α
	​

σ
t+1
	​


donde 
𝑞
𝛼
q
α
	​

 es el cuantil de la distribución asumida.

3.2 VaR histórico (rolling window)

Cuantil empírico sobre ventana móvil (ej. 250 días).

3.3 VaR Monte Carlo

Simulación:

𝑟
𝑡
+
1
(
𝑠
𝑖
𝑚
)
=
𝜇
𝑡
+
1
+
𝜎
𝑡
+
1
𝑧
(
𝑠
𝑖
𝑚
)
r
t+1
(sim)
	​

=μ
t+1
	​

+σ
t+1
	​

z
(sim)

y cálculo del cuantil empírico de la distribución simulada.

Incluye extensión multivariada con descomposición de Cholesky.

4. Backtesting

Se define violación cuando:

𝑟
𝑡
<
𝑉
𝑎
𝑅
𝑡
r
t
	​

<VaR
t
	​


Se aplican:

Kupiec (Unconditional Coverage)

Christoffersen (Independence)

Conditional Coverage (test conjunto)

Además:

Visualización de violaciones

Rolling hit rate (250 días)

5. Cómo ejecutar
Opción A — Notebook (recomendado)
pip install -r requirements.txt
jupyter notebook notebook.ipynb

El notebook se encuentra ejecutado e incluye outputs y visualizaciones.

Opción B — Script reproducible
python src/run_pipeline.py
6. Estructura del repositorio

notebook.ipynb → Notebook ejecutado con outputs y visualizaciones.

src/run_pipeline.py → Pipeline reproducible en modo batch.

requirements.txt → Dependencias.

7. Limitaciones y extensiones

Limitaciones:

Modelo univariado

Correlación estática en versión multivariada

No incluye Expected Shortfall

Extensiones naturales:

Expected Shortfall (Basel III)

DCC-GARCH

Regime-switching

Stress testing

Validación fuera de muestra

Disclaimer

Proyecto con fines académicos y de investigación.
No constituye recomendación de inversión.
