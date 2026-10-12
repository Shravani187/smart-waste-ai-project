"""Practical 6 — Fuzzy Logic Room Temperature Controller.

Inputs: temperature error e = setpoint - measured (negative = too hot,
positive = too cold) and change in error.
Output: controller action (negative = cool, positive = heat).

This is a standalone software simulation.  It is NOT connected to any real
heating or cooling equipment.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def trimf(x, a, b, c):
    """Triangular membership function."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def memberships_error(e):
    """Linguistic sets for temperature error (degrees C)."""
    return {
        "cold": trapmf(e, -10, -10, -5, -2),
        "cool": trimf(e, -4, -1.5, 1),
        "ok": trimf(e, -1, 0, 1),
        "warm": trimf(e, -1, 1.5, 4),
        "hot": trapmf(e, 2, 5, 10, 10),
    }


def memberships_delta(d):
    """Linguistic sets for change in error."""
    return {
        "falling": trimf(d, -3, -1.5, 0),
        "steady": trimf(d, -1, 0, 1),
        "rising": trimf(d, 0, 1.5, 3),
    }


def _trimf_vec(x, a, b, c):
    """Vectorized triangular membership function."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    left = (x > a) & (x < b)
    right = (x >= b) & (x < c)
    y[left] = (x[left] - a) / (b - a)
    y[right] = (c - x[right]) / (c - b)
    y[x == b] = 1.0
    return y


def memberships_output(u):
    """Linguistic sets for controller output."""
    u = np.asarray(u)
    return {
        "heat": _trimf_vec(u, -1, -0.5, 0),
        "hold": _trimf_vec(u, -0.4, 0, 0.4),
        "cool": _trimf_vec(u, 0, 0.5, 1),
    }


def fuzzify(e, d):
    return memberships_error(e), memberships_delta(d)


def infer(e, d):
    """Apply fuzzy rules and return aggregated output membership function.

    Rules (error, delta) -> output:
      cold/rising -> heat,  cold/steady -> heat,  cold/falling -> hold,
      cool/rising -> hold,  cool/steady -> hold,  cool/falling -> cool,
      ok/rising   -> hold,  ok/steady   -> hold,  ok/falling   -> cool,
      warm/rising -> cool,  warm/steady -> cool,  warm/falling -> cool,
      hot/rising  -> cool,  hot/steady  -> cool,  hot/falling  -> cool
    """
    em, dm = fuzzify(e, d)
    rules = [
        (("cold", "rising"), "heat"),
        (("cold", "steady"), "heat"),
        (("cold", "falling"), "hold"),
        (("cool", "rising"), "hold"),
        (("cool", "steady"), "hold"),
        (("cool", "falling"), "cool"),
        (("ok", "rising"), "hold"),
        (("ok", "steady"), "hold"),
        (("ok", "falling"), "cool"),
        (("warm", "rising"), "cool"),
        (("warm", "steady"), "cool"),
        (("warm", "falling"), "cool"),
        (("hot", "rising"), "cool"),
        (("hot", "steady"), "cool"),
        (("hot", "falling"), "cool"),
    ]
    u = np.linspace(-1, 1, 401)
    aggregated = np.zeros_like(u)
    for (et, dt), out in rules:
        strength = min(em[et], dm[dt])
        if strength > 0:
            aggregated = np.maximum(aggregated, strength * memberships_output(u)[out])
    return u, aggregated


def defuzzify(u, aggregated):
    """Centre-of-gravity defuzzification."""
    total = aggregated.sum()
    if total == 0:
        return 0.0
    return float(np.sum(u * aggregated) / total)


def controller_output(e, d):
    u, agg = infer(e, d)
    return defuzzify(u, agg)


def simulate(setpoint=22.0, initial=28.0, steps=30):
    """Simulate room temperature response over time using the controller."""
    temp = initial
    temps = [temp]
    for _ in range(steps):
        e = setpoint - temp
        u = controller_output(e, 0.0)
        # Simple plant model: positive u heats, negative u cools.
        temp += 0.3 * u + np.random.normal(0, 0.05)
        temps.append(temp)
    return np.array(temps)


def plot_memberships():
    e = np.linspace(-10, 10, 401)
    d = np.linspace(-3, 3, 201)
    u = np.linspace(-1, 1, 401)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, fn in memberships_error(0).items():
        pass
    for label, params in [("cold", (-10, -10, -5, -2)), ("cool", (-4, -1.5, 1)),
                          ("ok", (-1, 0, 1)), ("warm", (-1, 1.5, 4)),
                          ("hot", (2, 5, 10, 10))]:
        if len(params) == 4:
            y = [trapmf(x, *params) for x in e]
        else:
            y = [trimf(x, *params) for x in e]
        axes[0].plot(e, y, label=label)
    axes[0].set_title("Error membership")
    for label, params in [("falling", (-3, -1.5, 0)), ("steady", (-1, 0, 1)),
                          ("rising", (0, 1.5, 3))]:
        y = [trimf(x, *params) for x in d]
        axes[1].plot(d, y, label=label)
    axes[1].set_title("Delta-error membership")
    for label, params in [("heat", (-1, -0.5, 0)), ("hold", (-0.4, 0, 0.4)),
                          ("cool", (0, 0.5, 1))]:
        y = [trimf(x, *params) for x in u]
        axes[2].plot(u, y, label=label)
    axes[2].set_title("Output membership")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig("practical_06_memberships.png", dpi=120)
    plt.close()


def plot_response(temps, setpoint=22.0):
    plt.figure(figsize=(8, 4))
    plt.plot(temps, label="Room temperature")
    plt.axhline(setpoint, color="r", linestyle="--", label="Setpoint")
    plt.xlabel("Time step")
    plt.ylabel("Temperature (deg C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("practical_06_response.png", dpi=120)
    plt.close()


def run():
    print("Fuzzy Room Temperature Controller (standalone simulation)")
    print("Sample inputs/outputs:")
    for e, d in [(5, 0), (2, -1), (0, 0), (-2, 1), (-5, 0)]:
        print(f"  error={e:>3}, delta={d:>3} -> output={controller_output(e, d):+.3f}")
    np.random.seed(42)
    temps = simulate()
    print("\nSimulated temperatures (first 10):", np.round(temps[:10], 2))
    plot_memberships()
    plot_response(temps)
    print("Saved practical_06_memberships.png and practical_06_response.png")


if __name__ == "__main__":
    run()
