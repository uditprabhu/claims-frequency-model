// def make_synthetic(n: int, seed: int = 0) -> pd.DataFrame:
    """Fake data with the same schema as freMTPL2freq (for offline testing)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "IDpol": np.arange(n),
        "Exposure": rng.uniform(0.05, 1.0, n).round(2),
        "Area": rng.choice(list("ABCDEF"), n),
        "VehPower": rng.integers(4, 16, n),
        "VehAge": rng.integers(0, 25, n),
        "DrivAge": rng.integers(18, 95, n),
        "BonusMalus": rng.choice([50] * 6 + list(range(51, 200)), n),
        "VehBrand": rng.choice([f"B{i}" for i in range(1, 15)], n),
        "VehGas": rng.choice(["Diesel", "Regular"], n),
        "Density": rng.lognormal(5, 2, n).clip(1, 27000).round(),
        "Region": rng.choice([f"R{i}" for i in range(11, 95, 4)], n),
    })
    # Build in a sensible "true" risk so the models have something to learn
    log_rate = (-2.6
                + 0.012 * (df["BonusMalus"] - 50)
                + 0.08 * np.log(df["Density"])
                + 0.6 * (df["DrivAge"] < 25)
                - 0.02 * df["VehAge"])
    rate = np.exp(log_rate) * df["Exposure"]
    df["ClaimNb"] = rng.poisson(rate)
    return df
//
