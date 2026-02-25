#!/usr/bin/env python
"""
run_pipeline.py
---------------
Pipeline reproducible (CLI) coherente con framework ARIMA–GARCH y backtesting formal,
incluyendo Expected Shortfall (ES) en enfoques comparativos.

1) Precios -> retornos log
2) Diagnóstico ADF (retornos)
3) ARIMA (media condicional) -> innovaciones (residuos)
4) Diagnósticos sobre innovaciones: Ljung–Box + ARCH-LM
5) GARCH-family sobre innovaciones: GARCH / EGARCH / GJR (dist Normal o t)
6) Serie VaR rolling (1 día) para:
   - Paramétrico condicional (ARIMA + GARCH)
   - Histórico rolling
   - Monte Carlo (aprox: cuantil simulado una vez)
7) Backtesting VaR: Kupiec (UC) + Christoffersen (IND) + Conditional Coverage (CC)
8) ES (comparativo):
   - ES Histórico (cola empírica)
   - ES Paramétrico Dinámico (Normal / t; coherente con sigma condicional)
   - ES Monte Carlo (cola sobre escenarios simulados)

Uso:
  python -m src.run_pipeline --ticker CHILE.SN --start 2015-01-01 --conf 0.975 --method all --dist t

Ejemplos:
  # VaR/ES 97.5% (FRTB-style) con GJR y t
  python -m src.run_pipeline --ticker CHILE.SN --start 2015-01-01 --conf 0.975 --method all --dist t --vol-model GJR

  # VaR histórico 99% (alpha=1%) con ventana 250
  python -m src.run_pipeline --ticker CHILE.SN --start 2015-01-01 --alpha 0.01 --method hist --window 250
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, t as student_t, chi2

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

import pmdarima as pm
from arch import arch_model


# =========================
# Data
# =========================
def download_prices(ticker: str, start: str, end: Optional[str], use_adj_close: bool) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna()

    col = "Adj Close" if (use_adj_close and "Adj Close" in df.columns) else "Close"
    s = df[col].copy()
    s.name = "price"
    return s


def log_returns(price: pd.Series) -> pd.Series:
    r = np.log(price).diff().dropna()
    r.name = "r"
    return r


def adf_pvalue(x: pd.Series) -> float:
    _, pval, *_ = adfuller(x.dropna())
    return float(pval)


# =========================
# ARIMA (mean)
# =========================
def fit_auto_arima(r: pd.Series) -> pm.ARIMA:
    return pm.auto_arima(
        r,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        d=None,
        seasonal=False,
        stepwise=True,
        trace=False,
        information_criterion="aic",
        suppress_warnings=True,
        error_action="ignore",
    )


def arima_one_step_mean(arima: pm.ARIMA, r_index: pd.DatetimeIndex) -> pd.Series:
    """
    Aproximación μ̂_t (predicción 1-step in-sample).
    pmdarima entrega predict_in_sample() alineado al vector usado en fit.
    """
    mu = np.asarray(arima.predict_in_sample())
    mu_s = pd.Series(mu, index=r_index[-len(mu):], name="mu_hat")
    return mu_s


# =========================
# Diagnostics
# =========================
def ljung_box_pvalues(x: pd.Series, lags=(10, 20, 30)) -> pd.Series:
    out = acorr_ljungbox(x.dropna(), lags=list(lags), return_df=True)
    return out["lb_pvalue"]


def arch_lm_pvalue(x: pd.Series, nlags: int = 12) -> float:
    _, p, *_ = het_arch(x.dropna(), nlags=nlags)
    return float(p)


# =========================
# GARCH family (volatility)
# =========================
def fit_garch_family(eps: pd.Series, dist: str = "t") -> Dict[str, object]:
    """
    Ajusta modelos sobre innovaciones eps (mean='Zero'):
      - GARCH(1,1)
      - EGARCH(1,1)
      - GJR-GARCH(1,1)
    """
    models: Dict[str, object] = {}

    garch = arch_model(eps, mean="Zero", vol="GARCH", p=1, q=1, dist=dist)
    models["GARCH"] = garch.fit(disp="off")

    egarch = arch_model(eps, mean="Zero", vol="EGARCH", p=1, q=1, dist=dist)
    models["EGARCH"] = egarch.fit(disp="off")

    gjr = arch_model(eps, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist=dist)
    models["GJR"] = gjr.fit(disp="off")

    return models


def _t_scale_var1(nu: float) -> float:
    # escala para que Var(z)=1 cuando z ~ t(nu) escalada
    return float(np.sqrt((nu - 2.0) / nu))


def quantile_alpha(alpha: float, dist: str, nu: Optional[float]) -> float:
    """
    Cuantil z_alpha para shocks estandarizados con Var=1.
    """
    if dist == "normal":
        return float(norm.ppf(alpha))
    if dist == "t":
        if nu is None or np.isnan(nu):
            raise ValueError("nu requerido para distribución t.")
        q = float(student_t.ppf(alpha, df=nu))
        return _t_scale_var1(nu) * q
    raise ValueError("dist debe ser 'normal' o 't'.")


def es_coef_alpha(alpha: float, dist: str, nu: Optional[float]) -> float:
    """
    Coeficiente c_alpha para ES en shocks estandarizados (Var=1).
    Retorna valor NEGATIVO (cola izquierda) para usar: ES_t = mu_t + sigma_t * c_alpha
    """
    if dist == "normal":
        z = float(norm.ppf(alpha))
        return float(-norm.pdf(z) / alpha)  # negativo
    if dist == "t":
        if nu is None or np.isnan(nu):
            raise ValueError("nu requerido para distribución t.")
        q = float(student_t.ppf(alpha, df=nu))          # cuantil t
        f = float(student_t.pdf(q, df=nu))              # densidad en q
        # ES de t estándar (cola izquierda), en magnitud positiva:
        es_pos = ((nu + q*q) / (nu - 1.0)) * (f / alpha)
        # Ajuste a Var=1 y signo negativo para cola izquierda:
        return -_t_scale_var1(nu) * float(es_pos)
    raise ValueError("dist debe ser 'normal' o 't'.")


# =========================
# VaR series
# =========================
def var_series_parametric(mu_hat: pd.Series, sigma_hat: pd.Series, alpha: float, dist: str, nu: Optional[float]) -> pd.Series:
    q = quantile_alpha(alpha, dist, nu)
    v = mu_hat + sigma_hat * q
    v.name = "VaR_param"
    return v


def var_series_historical(r: pd.Series, alpha: float, window: int) -> pd.Series:
    v = r.rolling(window).quantile(alpha).shift(1)  # 1-step (info hasta t-1)
    v.name = "VaR_hist"
    return v


def var_series_monte_carlo(mu_hat: pd.Series, sigma_hat: pd.Series, alpha: float, dist: str, nu: Optional[float], n_sims: int, seed: int) -> pd.Series:
    """
    Monte Carlo rolling (aprox eficiente):
    - En vez de simular cada día (carísimo), simulamos UNA vez el cuantil z_alpha
      bajo la distribución (shocks estandarizados) y lo usamos como shock cuantil.
    """
    rng = np.random.default_rng(seed)

    if dist == "normal":
        z = rng.normal(size=n_sims)
    elif dist == "t":
        if nu is None or np.isnan(nu):
            raise ValueError("nu requerido para distribución t.")
        z = student_t.rvs(df=nu, size=n_sims, random_state=rng)
        z = _t_scale_var1(nu) * z  # Var=1
    else:
        raise ValueError("dist debe ser 'normal' o 't'.")

    z_alpha = float(np.quantile(z, alpha))
    v = mu_hat + sigma_hat * z_alpha
    v.name = "VaR_mc"
    return v


# =========================
# ES series
# =========================
def es_series_parametric(mu_hat: pd.Series, sigma_hat: pd.Series, alpha: float, dist: str, nu: Optional[float]) -> pd.Series:
    c = es_coef_alpha(alpha, dist, nu)
    es = mu_hat + sigma_hat * c
    es.name = "ES_param"
    return es


def es_series_historical(r: pd.Series, alpha: float, window: int) -> pd.Series:
    """
    ES histórico rolling (1-step):
    - Calcula VaR rolling y luego promedio de retornos <= VaR dentro de la ventana.
    - shift(1) para usar info hasta t-1.
    """
    def tail_mean(x: np.ndarray) -> float:
        if x.size == 0:
            return np.nan
        v = np.quantile(x, alpha)
        tail = x[x <= v]
        return float(np.mean(tail)) if tail.size else np.nan

    es = r.rolling(window).apply(lambda x: tail_mean(np.asarray(x)), raw=False).shift(1)
    es.name = "ES_hist"
    return es


def es_series_monte_carlo(mu_hat: pd.Series, sigma_hat: pd.Series, alpha: float, dist: str, nu: Optional[float], n_sims: int, seed: int) -> pd.Series:
    """
    ES Monte Carlo rolling (aprox eficiente):
    - Simulamos una vez shocks z (Var=1), luego:
      ES_z = mean(z | z <= q_alpha)
      ES_t = mu_t + sigma_t * ES_z
    """
    rng = np.random.default_rng(seed)

    if dist == "normal":
        z = rng.normal(size=n_sims)
    elif dist == "t":
        if nu is None or np.isnan(nu):
            raise ValueError("nu requerido para distribución t.")
        z = student_t.rvs(df=nu, size=n_sims, random_state=rng)
        z = _t_scale_var1(nu) * z  # Var=1
    else:
        raise ValueError("dist debe ser 'normal' o 't'.")

    q_alpha = float(np.quantile(z, alpha))
    tail = z[z <= q_alpha]
    es_z = float(np.mean(tail)) if tail.size else np.nan  # negativo
    es = mu_hat + sigma_hat * es_z
    es.name = "ES_mc"
    return es


# =========================
# Backtesting (VaR)
# =========================
@dataclass
class BacktestResult:
    alpha: float
    T: int
    violations: int
    hit_rate: float
    kupiec_p: float
    indep_p: float
    cc_p: float


def make_hits(r: pd.Series, var: pd.Series) -> np.ndarray:
    tmp = pd.concat([r, var], axis=1).dropna()
    hits = (tmp.iloc[:, 0].values < tmp.iloc[:, 1].values).astype(int)
    return hits


def kupiec_pvalue(hits: np.ndarray, alpha: float) -> float:
    x = int(hits.sum())
    T = int(hits.size)
    if T <= 0:
        return float("nan")

    phat = x / T
    eps = 1e-12
    phat = float(np.clip(phat, eps, 1 - eps))
    alpha = float(np.clip(alpha, eps, 1 - eps))

    ll0 = x * np.log(alpha) + (T - x) * np.log(1 - alpha)
    ll1 = x * np.log(phat) + (T - x) * np.log(1 - phat)
    LRuc = -2 * (ll0 - ll1)
    return float(1 - chi2.cdf(LRuc, df=1))


def christoffersen_indep_pvalue(hits: np.ndarray) -> float:
    h = hits.astype(int)
    if h.size < 2:
        return float("nan")

    n00 = np.sum((h[:-1] == 0) & (h[1:] == 0))
    n01 = np.sum((h[:-1] == 0) & (h[1:] == 1))
    n10 = np.sum((h[:-1] == 1) & (h[1:] == 0))
    n11 = np.sum((h[:-1] == 1) & (h[1:] == 1))

    def sdiv(a, b):
        return a / b if b > 0 else 0.0

    pi01 = sdiv(n01, n00 + n01)
    pi11 = sdiv(n11, n10 + n11)
    pi1  = sdiv(n01 + n11, n00 + n01 + n10 + n11)

    eps = 1e-12
    pi01 = float(np.clip(pi01, eps, 1 - eps))
    pi11 = float(np.clip(pi11, eps, 1 - eps))
    pi1  = float(np.clip(pi1,  eps, 1 - eps))

    ll_ind = (
        n00 * np.log(1 - pi01) + n01 * np.log(pi01) +
        n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    )
    ll_null = (n00 + n10) * np.log(1 - pi1) + (n01 + n11) * np.log(pi1)

    LRind = -2 * (ll_null - ll_ind)
    return float(1 - chi2.cdf(LRind, df=1))


def conditional_coverage_pvalue(hits: np.ndarray, alpha: float) -> float:
    # CC = UC + IND (df=2)
    x = int(hits.sum())
    T = int(hits.size)
    if T <= 1:
        return float("nan")

    # LRuc
    phat = x / T
    eps = 1e-12
    phat = float(np.clip(phat, eps, 1 - eps))
    alpha = float(np.clip(alpha, eps, 1 - eps))

    ll0 = x * np.log(alpha) + (T - x) * np.log(1 - alpha)
    ll1 = x * np.log(phat) + (T - x) * np.log(1 - phat)
    LRuc = -2 * (ll0 - ll1)

    # LRind
    h = hits.astype(int)
    n00 = np.sum((h[:-1] == 0) & (h[1:] == 0))
    n01 = np.sum((h[:-1] == 0) & (h[1:] == 1))
    n10 = np.sum((h[:-1] == 1) & (h[1:] == 0))
    n11 = np.sum((h[:-1] == 1) & (h[1:] == 1))

    def sdiv(a, b):
        return a / b if b > 0 else 0.0

    pi01 = sdiv(n01, n00 + n01)
    pi11 = sdiv(n11, n10 + n11)
    pi1  = sdiv(n01 + n11, n00 + n01 + n10 + n11)

    pi01 = float(np.clip(pi01, eps, 1 - eps))
    pi11 = float(np.clip(pi11, eps, 1 - eps))
    pi1  = float(np.clip(pi1,  eps, 1 - eps))

    ll_ind = (
        n00 * np.log(1 - pi01) + n01 * np.log(pi01) +
        n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    )
    ll_null = (n00 + n10) * np.log(1 - pi1) + (n01 + n11) * np.log(pi1)

    LRind = -2 * (ll_null - ll_ind)

    LRcc = LRuc + LRind
    return float(1 - chi2.cdf(LRcc, df=2))


def backtest(r: pd.Series, var: pd.Series, alpha: float) -> BacktestResult:
    hits = make_hits(r, var)
    T = int(hits.size)
    x = int(hits.sum())
    kup = kupiec_pvalue(hits, alpha)
    ind = christoffersen_indep_pvalue(hits)
    cc  = conditional_coverage_pvalue(hits, alpha)
    return BacktestResult(alpha=alpha, T=T, violations=x, hit_rate=(x / T if T else np.nan),
                          kupiec_p=kup, indep_p=ind, cc_p=cc)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default="CHILE.SN")
    ap.add_argument("--start", type=str, default="2015-01-01")
    ap.add_argument("--end", type=str, default=None)

    # Nivel: puedes pasar conf (ej 0.975) o alpha directamente
    ap.add_argument("--conf", type=float, default=None, help="Nivel de confianza (ej 0.975). Si se indica, alpha=1-conf.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Nivel de cola izquierda (ej 0.05). Ignorado si --conf está seteado.")

    ap.add_argument("--use-adj-close", type=int, default=1)
    ap.add_argument("--dist", type=str, default="t", choices=["normal", "t"])
    ap.add_argument("--method", type=str, default="all", choices=["param", "hist", "mc", "all"])
    ap.add_argument("--window", type=int, default=250, help="Ventana para métodos históricos rolling")
    ap.add_argument("--n-sims", type=int, default=200_000, help="Simulaciones para MC (una vez)")
    ap.add_argument("--seed", type=int, default=42, help="Seed para MC reproducible")
    ap.add_argument("--vol-model", type=str, default="GARCH", choices=["GARCH", "EGARCH", "GJR"])
    ap.add_argument("--with-es", type=int, default=1, help="1: calcula ES (hist/param/mc) además de VaR. 0: solo VaR/backtesting.")
    args = ap.parse_args()

    alpha = (1.0 - float(args.conf)) if (args.conf is not None) else float(args.alpha)

    # 1) Data
    price = download_prices(args.ticker, args.start, args.end, bool(args.use_adj_close))
    r = log_returns(price)

    print(f"\nTicker: {args.ticker}")
    print(f"Observaciones (retornos): {len(r)}")
    print(f"ADF p-value (retornos): {adf_pvalue(r):.6f}")
    print(f"Nivel: conf={(1-alpha):.3%} | alpha={alpha:.3%}")

    # 2) ARIMA mean
    arima = fit_auto_arima(r)
    mu_hat = arima_one_step_mean(arima, r.index)

    # Alineamos retornos al mu_hat (por recorte de ARIMA)
    r_al = r.loc[mu_hat.index]
    eps = (r_al - mu_hat).dropna()
    eps.name = "eps"

    print(f"\nARIMA order (auto): {arima.order}")
    print("Ljung–Box p-values (innovaciones):")
    print(ljung_box_pvalues(eps))

    # 3) ARCH evidence on innovations
    arch_p = arch_lm_pvalue(eps, nlags=12)
    print(f"\nARCH-LM p-value (innovaciones, 12 lags): {arch_p:.6g}")

    # 4) GARCH-family on innovations
    dist_arch = "t" if args.dist == "t" else "normal"
    fits = fit_garch_family(eps, dist=dist_arch)
    fit = fits[args.vol_model]

    nu = None
    if args.dist == "t":
        nu = float(fit.params.get("nu", np.nan))

    # sigma_t (in-sample) y 1-step: usamos shift(1) para info t-1
    sigma_t = pd.Series(fit.conditional_volatility, index=eps.index, name="sigma")
    sigma_1step = sigma_t.shift(1)

    # mu_t|t-1: aproximación 1-step con shift(1)
    mu_1step = mu_hat.shift(1).loc[sigma_1step.index]

    # 5) VaR series (rolling 1-step)
    vars_out = {}
    es_out = {}

    if args.method in ("param", "all"):
        v_param = var_series_parametric(mu_1step, sigma_1step, alpha, args.dist, nu)
        vars_out["Paramétrico (condicional)"] = v_param
        if args.with_es:
            es_out["ES Paramétrico (dinámico)"] = es_series_parametric(mu_1step, sigma_1step, alpha, args.dist, nu)

    if args.method in ("hist", "all"):
        v_hist = var_series_historical(r_al, alpha, args.window)
        vars_out["Histórico (rolling)"] = v_hist
        if args.with_es:
            es_out["ES Histórico (rolling)"] = es_series_historical(r_al, alpha, args.window)

    if args.method in ("mc", "all"):
        v_mc = var_series_monte_carlo(mu_1step, sigma_1step, alpha, args.dist, nu, n_sims=args.n_sims, seed=args.seed)
        vars_out["Monte Carlo (aprox)"] = v_mc
        if args.with_es:
            es_out["ES Monte Carlo (aprox)"] = es_series_monte_carlo(mu_1step, sigma_1step, alpha, args.dist, nu, n_sims=args.n_sims, seed=args.seed)

    # 6) Backtesting VaR
    print("\n==================== Backtesting VaR ====================")
    for name, v in vars_out.items():
        bt = backtest(r_al, v, alpha)
        print(f"\n[{name}]  alpha={bt.alpha:.2%}")
        print(f"  T={bt.T} | violaciones={bt.violations} | hit_rate={bt.hit_rate:.4%}")
        print(f"  Kupiec p-value (UC):        {bt.kupiec_p:.6g}")
        print(f"  Christoffersen p-value IND: {bt.indep_p:.6g}")
        print(f"  Conditional Coverage p-val: {bt.cc_p:.6g}")

    # 7) Resumen ES (sin “violaciones”)
    if args.with_es and es_out:
        print("\n==================== Expected Shortfall (ES) ====================")
        # mostramos último valor disponible como referencia
        for name, es in es_out.items():
            last = float(es.dropna().iloc[-1]) if es.dropna().shape[0] else float("nan")
            print(f"[{name}]  ES (último disponible): {last:.6f}")

    print("\nListo ✅  (ARIMA→innovaciones→GARCH-family→VaR→Backtesting + ES)")


if __name__ == "__main__":
    main()
