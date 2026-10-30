"""
Train, evaluate and explain the three claim-frequency models.

Usage (from the repo root):
    python -m src.train               # real data (downloads ~680k policies once)
    python -m src.train --synthetic   # fake data, offline smoke test only
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from .data import CATEGORICAL, FEATURES, NUMERIC, clean, load_raw
from .evaluate import summarise
from .models import make_baseline, make_gbm, make_glm
from .plots import plot_calibration, plot_lorenz

ROOT = Path(__file__).resolve().parents[1]
RESULTS, MODELS = ROOT / "results", ROOT / "models"

# Small-integer features (e.g. VehPower) give duplicate quantile bin edges;
# sklearn merges them automatically, so this warning is harmless.
warnings.filterwarnings("ignore", message="Bins whose width are too small")


def explain_gbm(gbm, X_sample, path):
    """SHAP: how much did each feature push each prediction up or down?"""
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        print("  (shap not installed - skipping explanation plot)")
        return
    Xt = gbm.named_steps["preprocess"].transform(X_sample)
    names = gbm.named_steps["preprocess"].get_feature_names_out()
    explainer = shap.TreeExplainer(gbm.named_steps["model"])
    values = explainer.shap_values(Xt)  # in log-frequency units
    shap.summary_plot(values, Xt, feature_names=names, show=False,
                      plot_size=(7, 5))
    plt.title("What drives predicted claim frequency (gradient boosting)",
              fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main(synthetic: bool = False, seed: int = 0):
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    print("Loading data...")
    df = clean(load_raw(synthetic=synthetic))
    print(f"  {len(df):,} policies, {int(df['ClaimNb'].sum()):,} claims, "
          f"{df['Exposure'].sum():,.0f} policy-years")
    print(f"  {100 * (df['ClaimNb'] == 0).mean():.1f}% of policies had no claims")

    train, test = train_test_split(df, test_size=0.2, random_state=seed)
    X_tr, X_te = train[FEATURES], test[FEATURES]
    y_tr, y_te = train["Frequency"], test["Frequency"]
    w_tr, w_te = train["Exposure"], test["Exposure"]

    models = {"Baseline": make_baseline(),
              "Poisson GLM": make_glm(),
              "Gradient boosting": make_gbm(seed)}
    fit_kwargs = {"Baseline": {"sample_weight": w_tr},
                  "Poisson GLM": {"model__sample_weight": w_tr},
                  "Gradient boosting": {"model__sample_weight": w_tr}}

    metrics, preds = {}, {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_tr, y_tr, **fit_kwargs[name])
        pred = model.predict(X_te)
        metrics[name] = summarise(y_te, pred, w_te)
        preds[name] = (pred, metrics[name]["gini"])
        print(f"  {metrics[name]}")

    # Deviance explained vs baseline - the headline number for the README
    base = metrics["Baseline"]["poisson_deviance"]
    for name in ("Poisson GLM", "Gradient boosting"):
        d = metrics[name]["poisson_deviance"]
        metrics[name]["deviance_improvement_vs_baseline_pct"] = \
            round(100 * (base - d) / base, 2)

    print("Saving charts...")
    plot_lorenz(y_te.values, preds, w_te.values, RESULTS / "lorenz.png")
    plot_calibration(y_te.values, preds, w_te.values, RESULTS / "calibration.png")
    explain_gbm(models["Gradient boosting"],
                X_te.sample(min(2000, len(X_te)), random_state=seed),
                RESULTS / "shap_summary.png")

    metrics["_data"] = {"synthetic": synthetic, "n_train": len(train),
                        "n_test": len(test)}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save models + what the demo app needs to build its input form
    meta = {
        "categories": {c: sorted(df[c].astype(str).unique().tolist())
                       for c in CATEGORICAL},
        "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in NUMERIC},
        "portfolio_frequency": float(df["ClaimNb"].sum() / df["Exposure"].sum()),
        "synthetic": synthetic,
    }
    joblib.dump({"glm": models["Poisson GLM"],
                 "gbm": models["Gradient boosting"], "meta": meta},
                MODELS / "models.joblib")
    print(f"Done. Results in {RESULTS}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true",
                   help="use fake data (offline testing only)")
    main(**vars(p.parse_args()))
