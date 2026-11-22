"""Forecast definition shared by the notebook and the live worker."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

TZ = "America/Argentina/Buenos_Aires"
DAY = pd.Timedelta(days=1)
WINDOW = pd.Timedelta(hours=6)
HORIZONS = {name: pd.DateOffset(**offset) for name, offset in [
    ("1D", {"days": 1}), ("7D", {"days": 7}), ("14D", {"days": 14}),
    ("1M", {"months": 1}), ("3M", {"months": 3}), ("6M", {"months": 6}),
    ("1Y", {"years": 1})]}
CONFIG = Path(__file__).with_name("tuned_config.json")


def make_pairs(prices):
    times, values = prices.index, prices.to_numpy(dtype=float)
    blocks = []
    for name, offset in [*HORIZONS.items(), ("adjacent", None)]:
        due = times + offset if offset else times
        end = times.searchsorted(due) if offset else np.arange(len(times)) + 1
        i = np.flatnonzero(end < len(times))
        j = end[i]
        if offset:
            keep = times[j] - due[i] <= WINDOW
            i, j = i[keep], j[keep]
        blocks.append(pd.DataFrame({"start": times[i], "end": times[j], "h": name,
            "elapsed": (times[j]-times[i])/DAY, "y": np.log(values[j]/values[i])}))
    return pd.concat(blocks, ignore_index=True).drop_duplicates(["start", "end"])


def historical_window(pairs, cutoff):
    return pairs.loc[(pairs.start >= cutoff-pd.DateOffset(years=4)) & (pairs.end <= cutoff)]


def fit_model(training, params):
    counts = training.h.value_counts()
    weights = 1 / training.h.map(counts)
    model = XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42,
                         n_jobs=1, **params)
    model.fit(training[["elapsed"]], training.y, sample_weight=weights/weights.mean())
    return model


def forecast(model, base, quote_at, target, weight):
    x = pd.DataFrame({"elapsed": [(target-quote_at)/DAY]})
    predicted = float(base * np.exp(float(model.predict(x)[0])))
    return predicted, base + weight*(predicted-base)


def match(prices, target):
    j = prices.index.searchsorted(target)
    if j < len(prices) and prices.index[j] <= target+WINDOW:
        return prices.index[j], float(prices.iloc[j])
    return pd.NaT, np.nan


def load_config():
    if not CONFIG.exists():
        raise SystemExit("Run USD_ARS_Final.ipynb to generate tuned_config.json first.")
    config = json.loads(CONFIG.read_text())
    specs = config["horizons"]
    if config["schema"] != 1 or set(specs) != set(HORIZONS):
        raise ValueError("Invalid horizon configuration")
    for spec in specs.values():
        if not 0 <= spec["xgb_weight"] <= 1:
            raise ValueError("Invalid blend weight")
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    return specs, key
