"""Practical 8 — Hybrid Neuro-Fuzzy System (ANFIS).

Predicts student performance (marks 0-100) from study hours per day and
attendance percentage.

Implements a Sugeno-style ANFIS with:
  - Gaussian membership functions for premise parameters,
  - rule firing strengths and normalized weights,
  - linear consequent functions,
  - hybrid learning: least-squares for consequents + gradient descent for
    premise parameters.

Uses synthetic illustrative data.  Results do not establish real-world accuracy.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

SEED = 42


def make_data(n=120, seed=SEED):
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0.5, 8.0, n)
    attendance = rng.uniform(40, 100, n)
    marks = 5 * hours + 0.5 * attendance + rng.normal(0, 4, n)
    marks = np.clip(marks, 0, 100)
    X = np.column_stack([hours, attendance])
    return X, marks


def gauss(x, c, s):
    return np.exp(-((x - c) ** 2) / (2 * s ** 2))


class ANFIS:
    """2-input, 2-membership-per-input Sugeno ANFIS (4 rules)."""

    def __init__(self, X):
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-9
        Xn = (X - self.x_mean) / self.x_std
        self.c = np.array([[Xn[:, 0].min(), Xn[:, 0].max()],
                           [Xn[:, 1].min(), Xn[:, 1].max()]])
        self.s = np.array([[0.8, 0.8], [0.8, 0.8]])
        self.conseq = np.zeros((4, 3))

    def _normalize(self, X):
        return (X - self.x_mean) / self.x_std

    def _firing(self, Xn):
        mf0 = gauss(Xn[:, 0], self.c[0, 0], self.s[0, 0])
        mf1 = gauss(Xn[:, 0], self.c[0, 1], self.s[0, 1])
        mf2 = gauss(Xn[:, 1], self.c[1, 0], self.s[1, 0])
        mf3 = gauss(Xn[:, 1], self.c[1, 1], self.s[1, 1])
        return np.column_stack([mf0 * mf2, mf0 * mf3, mf1 * mf2, mf1 * mf3])

    def _rule_outputs(self, Xn):
        n = Xn.shape[0]
        out = np.zeros((n, 4))
        for k in range(4):
            out[:, k] = (self.conseq[k, 0] * Xn[:, 0] +
                         self.conseq[k, 1] * Xn[:, 1] +
                         self.conseq[k, 2])
        return out

    def predict(self, X):
        Xn = self._normalize(X)
        w = self._firing(Xn)
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum[w_sum == 0] = 1e-9
        norm = w / w_sum
        y = self._rule_outputs(Xn)
        return np.clip((norm * y).sum(axis=1), 0, 100)

    def fit(self, X, y_target, epochs=100, lr=0.05):
        Xn = self._normalize(X)
        n = len(Xn)
        # Step 1: regularized least-squares for consequent parameters
        w = self._firing(Xn)
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum[w_sum == 0] = 1e-9
        norm = w / w_sum
        I = np.eye(3) * 1.0
        for k in range(4):
            A_k = norm[:, k][:, None] * np.column_stack([Xn, np.ones(n)])
            A_T_A = A_k.T @ A_k + I
            self.conseq[k] = np.linalg.solve(A_T_A, A_k.T @ y_target)
        # Step 2: gradient descent on premise parameters
        eps = 1e-9
        for _ in range(epochs):
            w = self._firing(Xn)
            w_sum = w.sum(axis=1, keepdims=True)
            w_sum[w_sum < eps] = eps
            norm = w / w_sum
            rule_out = self._rule_outputs(Xn)
            pred = np.clip((norm * rule_out).sum(axis=1), 0, 100)
            err = pred - y_target
            for i in range(2):
                for m in range(2):
                    k = i * 2 + m
                    d = Xn[:, i] - self.c[i, m]
                    common = err * (rule_out[:, k] - pred) / w_sum[:, 0] * w[:, k]
                    grad_c = (common * d / (self.s[i, m] ** 2 + eps)).sum()
                    grad_s = (common * (d ** 2) / (self.s[i, m] ** 3 + eps)).sum()
                    grad_c = np.clip(grad_c, -1e3, 1e3)
                    grad_s = np.clip(grad_s, -1e3, 1e3)
                    self.c[i, m] -= lr * grad_c * 0.001
                    self.s[i, m] -= lr * grad_s * 0.001
                    self.s[i, m] = max(self.s[i, m], 0.1)
        return self


def run():
    X, y = make_data()
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = ANFIS(X_train).fit(X_train, y_train, epochs=100, lr=0.05)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    mae_train = np.mean(np.abs(pred_train - y_train))
    mae_test = np.mean(np.abs(pred_test - y_test))
    ss_res = np.sum((y_test - pred_test) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print("ANFIS student-performance prediction (synthetic data)")
    print(f"Train MAE: {mae_train:.2f}")
    print(f"Test  MAE: {mae_test:.2f}")
    print(f"Test  R2 : {r2:.3f}")
    print("\nSample predictions:")
    for i in range(5):
        print(f"  hours={X_test[i,0]:.1f}, attendance={X_test[i,1]:.0f} "
              f"-> pred={pred_test[i]:.1f}, actual={y_test[i]:.1f}")

    plt.figure(figsize=(8, 4))
    plt.scatter(y_test, pred_test, alpha=0.6)
    plt.plot([0, 100], [0, 100], "r--")
    plt.xlabel("Actual marks")
    plt.ylabel("Predicted marks")
    plt.tight_layout()
    plt.savefig("practical_08_anfis.png", dpi=120)
    plt.close()
    print("\nSaved practical_08_anfis.png")


if __name__ == "__main__":
    run()
