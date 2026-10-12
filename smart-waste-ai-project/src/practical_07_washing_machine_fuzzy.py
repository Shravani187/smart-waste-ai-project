"""Practical 7 — Mamdani and Sugeno Washing-Machine Fuzzy Controllers.

Inputs: clothes load size (kg) and dirt level (0-100).
Output: required water level (litres).

Implements both Mamdani (output aggregation + defuzzification) and Sugeno
(crisp/linear consequents + weighted averaging) approaches.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def trimf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    return (x - a) / (b - a) if x < b else (c - x) / (c - b)


def trapmf(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    return (x - a) / (b - a) if x < b else (d - x) / (d - c)


def load_mf(load):
    return {
        "small": trapmf(load, 0, 0, 2, 4),
        "medium": trimf(load, 2, 5, 8),
        "large": trapmf(load, 6, 8, 12, 12),
    }


def dirt_mf(dirt):
    return {
        "low": trapmf(dirt, 0, 0, 20, 40),
        "medium": trimf(dirt, 20, 50, 80),
        "high": trapmf(dirt, 60, 80, 100, 100),
    }


def water_mf(w):
    return {
        "low": trimf(w, 0, 15, 30),
        "medium": trimf(w, 20, 40, 60),
        "high": trimf(w, 50, 70, 90),
    }


RULES = [
    (("small", "low"), "low"),
    (("small", "medium"), "low"),
    (("small", "high"), "medium"),
    (("medium", "low"), "low"),
    (("medium", "medium"), "medium"),
    (("medium", "high"), "high"),
    (("large", "low"), "medium"),
    (("large", "medium"), "high"),
    (("large", "high"), "high"),
]

SUGENO_CONSEQUENTS = {
    "low": 15.0,
    "medium": 40.0,
    "high": 70.0,
}


def mamdani(load, dirt):
    lm, dm = load_mf(load), dirt_mf(dirt)
    w = np.linspace(0, 90, 451)
    aggregated = np.zeros_like(w)
    for (lt, dt), out in RULES:
        strength = min(lm[lt], dm[dt])
        if strength > 0:
            aggregated = np.maximum(aggregated, strength * np.array([water_mf(x)[out] for x in w]))
    total = aggregated.sum()
    if total == 0:
        return 0.0
    return float(np.sum(w * aggregated) / total)


def sugeno(load, dirt):
    lm, dm = load_mf(load), dirt_mf(dirt)
    num, den = 0.0, 0.0
    for (lt, dt), out in RULES:
        strength = min(lm[lt], dm[dt])
        num += strength * SUGENO_CONSEQUENTS[out]
        den += strength
    return num / den if den > 0 else 0.0


def run():
    test_cases = [(2, 20), (5, 50), (8, 80), (3, 70), (10, 30)]
    print("Washing-machine water level (litres)")
    print(f"{'Load(kg)':>8} {'Dirt':>5} {'Mamdani':>9} {'Sugeno':>8}")
    for load, dirt in test_cases:
        m = mamdani(load, dirt)
        s = sugeno(load, dirt)
        print(f"{load:>8} {dirt:>5} {m:>9.2f} {s:>8.2f}")

    loads = np.linspace(0, 12, 60)
    dirt_fixed = 50
    m_vals = [mamdani(l, dirt_fixed) for l in loads]
    s_vals = [sugeno(l, dirt_fixed) for l in loads]
    plt.figure(figsize=(8, 4))
    plt.plot(loads, m_vals, label="Mamdani")
    plt.plot(loads, s_vals, label="Sugeno")
    plt.xlabel("Load size (kg)")
    plt.ylabel("Water level (L)")
    plt.title(f"Water level vs load (dirt={dirt_fixed})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("practical_07_comparison.png", dpi=120)
    plt.close()
    print("\nSaved practical_07_comparison.png")


if __name__ == "__main__":
    run()
