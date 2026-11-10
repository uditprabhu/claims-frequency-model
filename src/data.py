"""
Loading and cleaning the French motor third-party liability dataset (freMTPL2freq).

Each row is one insurance policy:
    ClaimNb   - number of claims the policyholder made (the TARGET)
    Exposure  - fraction of a year the policy was active (0.5 = six months)
    plus driver / vehicle / location features.

We model claim FREQUENCY = ClaimNb / Exposure  (claims per policy-year).
"""

import numpy as np
import pandas as pd

OPENML_ID = 41214  # freMTPL2freq on openml.org

CATEGORICAL = ["Area", "VehBrand", "VehGas", "Region"]
NUMERIC = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]
FEATURES = CATEGORICAL + NUMERIC


def load_raw(synthetic: bool = False, n_synthetic: int = 50_000) -> pd.DataFrame:
    """Download the real dataset from OpenML (cached after the first run).

    synthetic=True generates fake data with the same columns. It is ONLY for
    testing that the code runs without internet - never report its results.
    """
    if synthetic:
        return make_synthetic(n_synthetic)

    from sklearn.datasets import fetch_openml

    df = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto").frame
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the standard actuarial clean-up used for this dataset."""
    df = df.copy()

    # Some OpenML versions wrap category values in quotes: "'A'" -> "A"
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip("'").astype("category")

    numeric_cols = ["ClaimNb", "Exposure", "VehPower", "VehAge",
                    "DrivAge", "BonusMalus", "Density"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # 1. A handful of policies have >4 claims in a year - almost certainly
    #    data errors. Capping is the standard treatment.
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)

    # 2. Exposure is a fraction of a YEAR, so anything above 1 is an error.
    df["Exposure"] = df["Exposure"].clip(upper=1.0)

    # 3. Cap extreme values so a few outliers don't dominate the model.
    df["VehAge"] = df["VehAge"].clip(upper=20)
    df["DrivAge"] = df["DrivAge"].clip(upper=90)
    df["BonusMalus"] = df["BonusMalus"].clip(upper=150)

    # 4. Population density is extremely skewed (1 to ~27,000 people/km^2).
    #    Taking the log makes it far easier for a linear model to use.
    df["LogDensity"] = np.log(df["Density"])

    # 5. The quantity we actually model: claims per policy-year.
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]

    return df



