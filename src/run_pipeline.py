#!/usr/bin/env python
"""
run_pipeline.py
---------------
Ejecución reproducible (sin notebook) del pipeline de riesgo:

Precios -> retornos log -> ADF -> (auto) ARIMA -> residuos -> ARCH-LM ->
GARCH/EGARCH/GJR (t) -> forecast sigma -> VaR (param/hist/MC) -> backtesting.

Uso:
  python -m src.run_pipeline --ticker CHILE.SN --start 2015-01-01 --alpha 0.05 --use-adj-close 1
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, t as student_t

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import pmdarima as pm

from arch import arch_model


@dataclass
class BacktestResult:
    alpha: float
    T: int
    violations: int
    hit_rate: float
    kupiec_p: float
    indep_p: float
    cc_p: float


def download_prices(ticker: str, start: str, end: str | None, use_adj_close: bool) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna()

    price_col = "Adj Close" if (use_adj_close and "Adj Close" in df.columns) else "Close"
    price = df[price_col].copy()
    price.name = "price"
    return price


def log_returns(price: pd.Series) -> pd.Series:
    r = np.log(price).diff().dropna()
    r.name = "log_ret"
    return r


def adf_pvalue(x: pd.Series) -> float:
    stat, pval, *_ = adfuller(x.dropna())
    return float(pval)


def fit_auto_arima(r: pd.Series) -> pm.ARIMA:
    model = pm.auto_arima(
        r,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        d=None,
        seasonal=False,
        stepwise=True,
        trace=False,
        information_criterion="aic",
        suppress_warnings=True
    )
    return model


def ljung_box_pvalues(resid: pd.Series, lags=(10, 20, 30)) -> pd.Series:
    out = acorr_ljungbox(resid, lags=list(lags), return_df=True)
    return out["lb_pvalue"]


def arch_lm_pvalue(resid: pd.Series, nlags: int = 12) -> float:
    lm_stat, lm_p, *_ = het_arch(resid.dropna(), nlags=nlags)
    return float(lm_p)


def fit_garch_family(r: pd.Series, dist: str = "t") -> Dict[str, object]:
    """
    Ajusta modelos:
      - GARCH(1,1)
      - EGARCH(1,1)
      - GJR-GARCH(1,1)
    """
    models = {}

    garch = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist=dist)
    models["GARCH"] = garch.fit(disp="off")

    egarch = arch_model(r, mean="Constant", vol="EGARCH", p=1, q=1, dist=dist)
    models["EGARCH"] = egarch.fit(disp="off")

    gjr = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist=dist)
    models["GJR"] = gjr.fit(disp="off")

    return models


def forecast_sigma_1d(fit) -> float:
    fc = fit.forecast(horizon=1)
    var1 = float(fc.variance.values[-1, 0])
    return float(np.sqrt(var1))


def var_parametric(mu: float, sigma: float, alpha: float, dist: str, nu: float | None = None) -> float:
    if dist == "normal":
        q = norm.ppf(alpha)
    elif dist == "t":
        if nu is None:
            raise ValueError("nu requerido para distribución t.")
        q = student_t.ppf(alpha, df=nu)
    else:
        raise ValueError("dist debe ser 'normal' o 't'.")
    return float(mu + sigma * q)


def var_historical(r: pd.Series, alpha: float) -> float:
    return float(np.quantile(r.dropna().values, alpha))


def var_monte_carlo(mu: float, sigma: float, alpha: float, dist: str, nu: float | None, n_sims: int = 200_000) -> float:
    if dist == "normal":
        z = np.random.normal(size=n_sims)
    elif dist == "t":
        if nu is None:
            raise ValueError("nu requerido para distribución t.")
        z = student_t.rvs(df=nu, size=n_sims)
    else:
        raise ValueError("dist debe ser 'normal' o 't'.")
    sims = mu + sigma * z
    return float(np.quantile(sims, alpha))


def kupiec_test(violations: int, T: int, alpha: float) -> float:
    """
    Kupiec (Unconditional Coverage) LR test p-value.
    """
    from scipy.stats import chi2

    if T <= 0:
        return np.nan

    x = violations
    pi_hat = x / T
    pi0 = alpha

    # Evitar log(0)
    eps = 1e-12
    pi_hat = min(max(pi_hat, eps), 1 - eps)
    pi0 = min(max(pi0, eps), 1 - eps)

    ll0 = (T - x) * np.log(1 - pi0) + x * np.log(pi0)
    ll1 = (T - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)

    LR = -2 * (ll0 - ll1)
    p = 1 - chi2.cdf(LR, df=1)
    return float(p)


def christoffersen_independence(hits: np.ndarray) -> float:
    """
    Christoffersen Independence test p-value (1 df).
    hits: array binario de violaciones (1=violación, 0=no).
    """
    from scipy.stats import chi2

    h = hits.astype(int)
    if h.size < 2:
        return np.nan

    n00 = np.sum((h[:-1] == 0) & (h[1:] == 0))
    n01 = np.sum((h[:-1] == 0) & (h[1:] == 1))
    n10 = np.sum((h[:-1] == 1) & (h[1:] == 0))
    n11 = np.sum((h[:-1] == 1) & (h[1:] == 1))

    # Probabilidades transicionales
    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    pi01 = safe_div(n01, n00 + n01)
    pi11 = safe_div(n11, n10 + n11)
    pi1  = safe_div(n01 + n11, n00 + n01 + n10 + n11)

    eps = 1e-12
    pi01 = min(max(pi01, eps), 1 - eps)
    pi11 = min(max(pi11, eps), 1 - eps)
    pi1  = min(max(pi1,  eps), 1 - eps)

    ll_ind = (
        n00 * np.log(1 - pi01) + n01 * np.log(pi01) +
        n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    )
    ll_null = (n00 + n10) * np.log(1 - pi1) + (n01 + n11) * np.log(pi1)

    LR = -2 * (ll_null - ll_ind)
    p = 1 - chi2.cdf(LR, df=1)
    return float(p)


def conditional_coverage_p(kupiec_p: float, indep_p: float, violations: int, T: int, alpha: float, hits: np.ndarray) -> float:
    """
    Conditional Coverage p-value (2 df) = UC + IND.
    Para evitar recomputar LR, usamos aproximación reconstruyendo LR desde p-values.
    Mejor: calcular LR_uc y LR_ind explícitos. Aquí implementamos explícito.
    """
    from scipy.stats import chi2

    # Recompute LR_uc
    x = violations
    pi_hat = x / T
    pi0 = alpha
    eps = 1e-12
    pi_hat = min(max(pi_hat, eps), 1 - eps)
    pi0 = min(max(pi0, eps), 1 - eps)

    ll0 = (T - x) * np.log(1 - pi0) + x * np.log(pi0)
    ll1 = (T - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
    LR_uc = -2 * (ll0 - ll1)

    # LR_ind
    h = hits.astype(int)
    n00 = np.sum((h[:-1] == 0) & (h[1:] == 0))
    n01 = np.sum((h[:-1] == 0) & (h[1:] == 1))
    n10 = np.sum((h[:-1] == 1) & (h[1:] == 0))
    n11 = np.sum((h[:-1] == 1) & (h[1:] == 1))

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    pi01 = safe_div(n01, n00 + n01)
    pi11 = safe_div(n11, n10 + n11)
    pi1  = safe_div(n01 + n11, n00 + n01 + n10 + n11)

    pi01 = min(max(pi01, eps), 1 - eps)
    pi11 = min(max(pi11, eps), 1 - eps)
    pi1  = min(max(pi1,  eps), 1 - eps)

    ll_ind = (
        n00 * np.log(1 - pi01) + n01 * np.log(pi01) +
        n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    )
    ll_null = (n00 + n10) * np.log(1 - pi1) + (n01 + n11) * np.log(pi1)

    LR_ind = -2 * (ll_null - ll_ind)

    LR_cc = LR_uc + LR_ind
    p = 1 - chi2.cdf(LR_cc, df=2)
    return float(p)


def backtest_var(r: pd.Series, var_series: pd.Series, alpha: float) -> BacktestResult:
    aligned = pd.concat([r, var_series], axis=1).dropna()
    r_al = aligned.iloc[:, 0].values
    v_al = aligned.iloc[:, 1].values

    hits = (r_al < v_al).astype(int)
    T = hits.size
    x = int(hits.sum())

    kup = kupiec_test(x, T, alpha)
    ind = christoffersen_independence(hits)
    cc = conditional_coverage_p(kup, ind, x, T, alpha, hits)

    return BacktestResult(alpha=alpha, T=T, violations=x, hit_rate=x / T, kupiec_p=kup, indep_p=ind, cc_p=cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default="CHILE.SN")
    ap.add_argument("--start", type=str, default="2015-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--use-adj-close", type=int, default=1)
    ap.add_argument("--dist", type=str, default="t", choices=["normal", "t"])
    args = ap.parse_args()

    price = download_prices(args.ticker, args.start, args.end, bool(args.use_adj_close))
    r = log_returns(price)

    print(f"\nTicker: {args.ticker}")
    print(f"Obs retornos: {len(r)}")
    print(f"ADF p-value (retornos): {adf_pvalue(r):.6f}")

    # Mean model (auto_arima)
    arima = fit_auto_arima(r)
    resid = pd.Series(arima.resid(), index=r.index[-len(arima.resid()):]).dropna()
    print(f"\nARIMA order (auto): {arima.order}")
    print("Ljung-Box p-values (residuos):")
    print(ljung_box_pvalues(resid))

    # ARCH evidence
    print(f"\nARCH-LM p-value (residuos, 12 lags): {arch_lm_pvalue(resid, 12):.6g}")

    # Volatility models
    fits = fit_garch_family(r, dist=("t" if args.dist == "t" else "normal"))
    base = fits["GARCH"]
    mu = float(base.params.get("mu", 0.0))
    nu = float(base.params.get("nu", np.nan)) if args.dist == "t" else None
    sigma1 = forecast_sigma_1d(base)

    print("\nGARCH(1,1) fitted params:")
    for k in ["omega", "alpha[1]", "beta[1]"]:
        if k in base.params:
            print(f"  {k}: {base.params[k]:.6g}")
    if args.dist == "t":
        print(f"  nu: {nu:.4g}")
    print(f"Forecast sigma (1d): {sigma1:.4%}")

    # VaR estimates (1-step, con mu~constante)
    var_p = var_parametric(mu, sigma1, args.alpha, args.dist, nu=nu)
    var_h = var_historical(r, args.alpha)
    var_mc = var_monte_carlo(mu, sigma1, args.alpha, args.dist, nu=nu)

    print(f"\nVaR ({args.alpha:.2%}) Paramétrico condicional: {var_p:.4%}")
    print(f"VaR ({args.alpha:.2%}) Histórico:             {var_h:.4%}")
    print(f"VaR ({args.alpha:.2%}) Monte Carlo:          {var_mc:.4%}")

    print("\nListo. Para backtesting completo, usar notebook (series VaR dinámicas).")


if __name__ == "__main__":
    main()
