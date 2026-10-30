"""Quick checks that the cleaning and metrics behave as expected.
Run with:  pytest -q
"""

import numpy as np

from src.data import clean, make_synthetic
from src.evaluate import gini, poisson_deviance


def test_clean_caps_and_derives_columns():
    df = make_synthetic(1_000)
    df.loc[0, "ClaimNb"] = 10      # absurd claim count
    df.loc[1, "Exposure"] = 2.0    # exposure longer than a year
    out = clean(df)
    assert out["ClaimNb"].max() <= 4
    assert out["Exposure"].max() <= 1.0
    assert np.allclose(out["Frequency"], out["ClaimNb"] / out["Exposure"])
    assert "LogDensity" in out


def test_perfect_ranking_beats_random_ranking():
    rng = np.random.default_rng(0)
    true_rate = rng.uniform(0.01, 1.0, 5_000)
    exposure = np.ones_like(true_rate)
    y = rng.poisson(true_rate) / exposure
    assert gini(y, true_rate, exposure) > 0.2
    assert abs(gini(y, rng.uniform(size=5_000), exposure)) < 0.05


def test_deviance_is_zero_for_exact_predictions():
    y = np.array([0.5, 1.0, 2.0])
    w = np.ones(3)
    assert poisson_deviance(y, y, w) < 1e-9
    assert poisson_deviance(y, y * 2, w) > 0
