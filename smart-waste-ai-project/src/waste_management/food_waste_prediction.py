"""Food Waste Prediction module (Practical 1).

Predicts estimated food-waste quantity (kg) from inputs such as number of
people, meals prepared, day type, and historical waste quantity.

Uses a small synthetic dataset and a regression model with a train/test split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

SEED = 42


def make_synthetic_data(n: int = 200, seed: int = SEED) -> pd.DataFrame:
    """Create a clearly-labelled synthetic food-waste dataset.

    The data is generated from a simple interpretable rule plus noise so the
    example is reproducible.  It is NOT real-world data.
    """
    rng = np.random.default_rng(seed)
    people = rng.integers(20, 401, size=n)
    meals = rng.integers(1, 5, size=n)
    day_type = rng.integers(0, 2, size=n)          # 0 = weekday, 1 = weekend
    hist_waste = rng.uniform(2.0, 60.0, size=n)

    # Illustrative rule: more people and meals -> more waste; weekends a bit more.
    base = 0.08 * people + 1.2 * meals + 3.0 * day_type + 0.25 * hist_waste
    noise = rng.normal(0, 3.0, size=n)
    waste = np.clip(base + noise, 0.0, None)

    df = pd.DataFrame({
        "people": people,
        "meals": meals,
        "day_type": day_type,
        "hist_waste": np.round(hist_waste, 2),
        "food_waste_kg": np.round(waste, 2),
    })
    return df


def train_model(df: pd.DataFrame, model_type: str = "linear"):
    """Train a regression model and return (model, X_test, y_test, metrics)."""
    X = df[["people", "meals", "day_type", "hist_waste"]].values
    y = df["food_waste_kg"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED
    )
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=SEED)
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "MAE": round(mean_absolute_error(y_test, preds), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 3),
        "R2": round(r2_score(y_test, preds), 3),
    }
    return model, X_test, y_test, preds, metrics


def predict(model, people: int, meals: int, day_type: int, hist_waste: float) -> float:
    """Return predicted food-waste in kg for one sample."""
    X = np.array([[people, meals, day_type, hist_waste]])
    return round(float(model.predict(X)[0]), 2)


def save_sample_csv(path: str, n: int = 200) -> None:
    make_synthetic_data(n).to_csv(path, index=False)


if __name__ == "__main__":
    df = make_synthetic_data()
    print(df.head())
    model, X_test, y_test, preds, metrics = train_model(df)
    print("Metrics:", metrics)
    print("Sample prediction:", predict(model, 150, 3, 1, 25.0), "kg")
