"""
The three models we compare.

    1. Baseline      - predicts the same average frequency for everyone.
    2. Poisson GLM   - the industry-standard actuarial pricing model.
    3. Gradient boosting with a Poisson loss - a modern ML model.

All three are trained the same way:
    target       y = ClaimNb / Exposure    (claims per policy-year)
    sample_weight  = Exposure

Weighting by exposure tells the model that a policy observed for a full
year carries more information than one observed for a week. For a Poisson
model this is mathematically equivalent to using log(Exposure) as an
"offset" - see the README for the derivation.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

from .data import CATEGORICAL

# A GLM is linear in its inputs, so it can't learn a U-shaped effect
# (young AND very old drivers being risky) from a raw number. Splitting
# these into bins lets each age band get its own coefficient.
BINNED = ["VehAge", "DrivAge", "VehPower"]
SCALED = ["BonusMalus", "LogDensity"]


def make_baseline() -> DummyRegressor:
    return DummyRegressor(strategy="mean")


def make_glm(alpha: float = 1e-4) -> Pipeline:
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("bins", KBinsDiscretizer(n_bins=10, encode="onehot",
                                  strategy="quantile",
                                  quantile_method="averaged_inverted_cdf"),
         BINNED),
        ("num", StandardScaler(), SCALED),
    ])
    # PoissonRegressor uses a LOG link: log(frequency) = w . x + b
    # so every coefficient acts as a multiplier on the risk: exp(w).
    return Pipeline([
        ("preprocess", preprocess),
        ("model", PoissonRegressor(alpha=alpha, solver="newton-cholesky",
                                   max_iter=300)),
    ])


def make_gbm(random_state: int = 0) -> Pipeline:
    preprocess = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=np.nan,
                                encoded_missing_value=np.nan), CATEGORICAL)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    # Trees can find non-linear effects and interactions on their own,
    # so no binning is needed. loss="poisson" makes it optimise the same
    # Poisson deviance as the GLM - a fair comparison.
    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        categorical_features=list(range(len(CATEGORICAL))),
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])
