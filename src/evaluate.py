"""
So how do we judge a claim-frequency model?

Accuracy and the R^2 are poor fits here: ~95% of policies have zero claims, so a model that always predicts a 0 
looks "accurate" but it's useless for pricing. Insurers mainly ask 3 questions. 

1. Fit - Poisson deviance: how likely are the observed claim
   counts under the model's predictions? (lower = better)
   
2. Ranking - Gini coefficient: does the model put the risky
   policies at the top? (higher = better)
                   
3. Calibration - do predicted total claims match actual total claims,
   overall and within each risk band?
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance


def poisson_deviance(y_freq, pred_freq, exposure) -> float:
    """Exposure-weighted mean Poisson deviance (lower is better)."""
    pred_freq = np.clip(pred_freq, 1e-9, None)  # deviance needs pred > 0
    return mean_poisson_deviance(y_freq, pred_freq, sample_weight=exposure)


def lorenz_curve(y_freq, pred_freq, exposure):
    """Sort policies from SAFEST to RISKIEST (by prediction), then track
    what share of all claims has been accumulated vs share of exposure."""
    order = np.argsort(pred_freq)
    claims = (y_freq * exposure)[order]          # actual claim counts
    cum_claims = np.cumsum(claims) / claims.sum()
    cum_exposure = np.cumsum(exposure[order]) / exposure.sum()
    return (np.concatenate([[0], cum_exposure]),
            np.concatenate([[0], cum_claims]))


def gini(y_freq, pred_freq, exposure) -> float:
    """Gini = 1 - 2 * (area under the Lorenz curve).
    0 = model ranks policies no better than random.
    The better the model separates risky from safe, the higher the Gini."""
    x, y = lorenz_curve(y_freq, np.asarray(pred_freq), np.asarray(exposure))
    return 1 - 2 * np.trapezoid(y, x)


def calibration_table(y_freq, pred_freq, exposure, n_bins: int = 10) -> pd.DataFrame:
    """Group policies into bins of equal exposure by predicted risk, then
    compare average predicted vs observed frequency in each bin."""
    df = pd.DataFrame({"y": y_freq, "pred": pred_freq, "w": exposure})
    df = df.sort_values("pred")
    df["bin"] = pd.cut(df["w"].cumsum() / df["w"].sum(),
                       bins=np.linspace(0, 1, n_bins + 1),
                       labels=False, include_lowest=True)
    return (df.groupby("bin")
              .apply(lambda g: pd.Series({
                  "predicted": np.average(g["pred"], weights=g["w"]),
                  "observed": np.average(g["y"], weights=g["w"]),
                  "exposure": g["w"].sum()}), include_groups=False)
              .reset_index())


def summarise(y_freq, pred_freq, exposure) -> dict:
    y_freq, pred_freq, exposure = map(np.asarray, (y_freq, pred_freq, exposure))
    actual = float((y_freq * exposure).sum())
    predicted = float((pred_freq * exposure).sum())
    return {
        "poisson_deviance": round(float(poisson_deviance(y_freq, pred_freq, exposure)), 5),
        "gini": round(float(gini(y_freq, pred_freq, exposure)), 4),
        "actual_claims": round(actual),
        "predicted_claims": round(predicted),
        "predicted_over_actual": round(predicted / actual, 4),
    }
