"""Practical 9 — Supervised and Unsupervised Neural Learning.

Part A: MLP regressor to predict parking occupancy percentage.
Part B: Self-Organizing Map (SOM) to group parking areas by daily occupancy
        patterns.

Uses clearly-labelled synthetic data.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

SEED = 42


def make_parking_data(n=500, seed=SEED):
    """Synthetic hourly parking occupancy data."""
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    weekend = rng.integers(0, 2, n)
    area_type = rng.integers(0, 3, n)  # 0=office, 1=residential, 2=mall
    base = 30 + 25 * np.sin((hour - 6) * np.pi / 12) ** 2
    base += 15 * (area_type == 2) * (hour >= 10) * (hour <= 20)
    base -= 20 * (area_type == 0) * (hour >= 18)
    base += 10 * weekend * (area_type == 1)
    occupancy = np.clip(base + rng.normal(0, 8, n), 0, 100)
    X = np.column_stack([hour, weekend, area_type])
    return X, occupancy


def make_daily_profiles(n_areas=12, seed=SEED):
    """Synthetic daily occupancy profiles (24 values per area)."""
    rng = np.random.default_rng(seed)
    profiles = []
    for _ in range(n_areas):
        peak_hour = rng.choice([9, 12, 18])
        profile = 20 + 60 * np.exp(-0.5 * ((np.arange(24) - peak_hour) / 3) ** 2)
        profile += rng.normal(0, 5, 24)
        profiles.append(np.clip(profile, 0, 100))
    return np.array(profiles)


class SOM:
    """Minimal 1-D Self-Organizing Map for grouping occupancy profiles."""

    def __init__(self, n_nodes=4, seed=SEED):
        self.n_nodes = n_nodes
        self.rng = np.random.default_rng(seed)
        self.weights = None

    def fit(self, X, epochs=100):
        n_features = X.shape[1]
        self.weights = self.rng.uniform(X.min(), X.max(), (self.n_nodes, n_features))
        for epoch in range(epochs):
            lr = 0.1 * (1 - epoch / epochs)
            sigma = max(self.n_nodes / 2 * (1 - epoch / epochs), 0.5)
            for x in X:
                dists = np.linalg.norm(self.weights - x, axis=1)
                bmu = int(np.argmin(dists))
                for i in range(self.n_nodes):
                    influence = np.exp(-((i - bmu) ** 2) / (2 * sigma ** 2))
                    self.weights[i] += lr * influence * (x - self.weights[i])
        return self

    def predict(self, X):
        return np.array([int(np.argmin(np.linalg.norm(self.weights - x, axis=1))) for x in X])


def run_part_a():
    X, y = make_parking_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)
    mlp = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=500, random_state=SEED)
    mlp.fit(X_train, y_train)
    preds = mlp.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print("Part A — Supervised MLP parking-occupancy prediction")
    print(f"  MAE: {mae:.2f}")
    print(f"  R2 : {r2:.3f}")
    print("  Sample predictions:")
    for i in range(5):
        print(f"    hour={X_test[i,0]:.0f}, weekend={X_test[i,1]:.0f}, "
              f"type={X_test[i,2]:.0f} -> pred={preds[i]:.1f}%, actual={y_test[i]:.1f}%")
    return mae, r2


def run_part_b():
    profiles = make_daily_profiles()
    som = SOM(n_nodes=3, seed=SEED).fit(profiles)
    labels = som.predict(profiles)
    print("\nPart B — SOM grouping of parking areas")
    for i, label in enumerate(labels):
        print(f"  Area {i:>2} -> Group {label}")
    plt.figure(figsize=(10, 4))
    for i, prof in enumerate(profiles):
        plt.plot(prof, alpha=0.5, label=f"Area {i}" if i < 4 else None)
    plt.xlabel("Hour of day")
    plt.ylabel("Occupancy (%)")
    plt.title("Daily occupancy profiles (synthetic)")
    plt.tight_layout()
    plt.savefig("practical_09_profiles.png", dpi=120)
    plt.close()
    plt.figure(figsize=(8, 4))
    for g in range(3):
        members = np.where(labels == g)[0]
        plt.bar([f"Area {m}" for m in members], [profiles[m].mean() for m in members],
                label=f"Group {g}")
    plt.ylabel("Mean occupancy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("practical_09_groups.png", dpi=120)
    plt.close()
    print("  Saved practical_09_profiles.png and practical_09_groups.png")
    return labels


def run():
    run_part_a()
    run_part_b()


if __name__ == "__main__":
    run()
