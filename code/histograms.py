"""
Created on Tues Jan 28 8:34 pm 2025

@author: kuckreja

@purpose: Compare stochastic vs. deterministic population distributions using Euler-Maruyama scheme
"""

import numpy as np 
import matplotlib.pyplot as plt

def euler_maruyama(c_factor, alpha_1, alpha_2, beta_1, beta_2, dt=0.001, T=1.0, num_trials=10000):
    """Run Euler-Maruyama simulation and return population distributions at selected times."""
    dtc = dt * c_factor  # Coarse time step.
    n = int(T / dt)  # Number of time steps.

    sqrtdt = np.sqrt(dt)

    # Model constants
    be = 1.0
    de = 1.4

    # Storage for population values at selected times
    check_steps = [int(n * 1/8), int(n * 1/2), int(n * 0.9)]  # (1/8)th, halfway, and 90%
    pop_values = {step: [] for step in check_steps}  

    for _ in range(num_trials):  # Run the same dtc many times to average
        # Array initialization
        w1 = np.zeros(n)
        w1c = np.zeros(int(n / c_factor))
        w2 = np.zeros(n)
        w2c = np.zeros(int(n / c_factor))
        w3 = np.zeros(n)
        w3c = np.zeros(int(n / c_factor))
        y = np.zeros(n)
        yc = np.zeros(int(n / c_factor))
        b = np.zeros(n)
        bc = np.zeros(int(n / c_factor))
        d = np.zeros(n)
        dc = np.zeros(int(n / c_factor))

        # Initial conditions
        w1[0] = 0
        w2[0] = 0
        w3[0] = 0
        y[0] = 30
        yc[0] = 30
        b[0] = be
        bc[0] = be
        d[0] = de
        dc[0] = de

        # Run Euler-Maruyama for fine dt
        for i in range(n - 1):
            w1[i + 1] = w1[i] + sqrtdt * np.random.randn()
            w2[i + 1] = w2[i] + sqrtdt * np.random.randn()
            w3[i + 1] = w3[i] + sqrtdt * np.random.randn()
            if (i % c_factor == 0):
                w1c[i // c_factor] = w1[i]
                w2c[i // c_factor] = w2[i]
                w3c[i // c_factor] = w3[i]  # Construct coarse Wiener from same fine ones
            y[i + 1] = y[i] + (b[i] * y[i] - d[i] * y[i]) * dt + np.sqrt(max(0, b[i] * y[i] + d[i] * y[i])) * (w1[i + 1] - w1[i])
            b[i + 1] = b[i] + beta_1 * (be - b[i]) * dt + alpha_1 * (w2[i + 1] - w2[i])
            d[i + 1] = d[i] + beta_2 * (de - d[i]) * dt + alpha_2 * (w3[i + 1] - w3[i])

            # Store population values at selected steps
            if i in check_steps:
                pop_values[i].append(y[i + 1])

    return pop_values  # Return dictionary of population distributions

# === RUN SIMULATION USING YOUR CODE FOR BOTH CASES ===
c_factor = 10  # Chosen time-step ratio

# Case 1: Stochastic (α = 0.5, β = 1.0)
pop_stochastic = euler_maruyama(c_factor, 0.5, 0.5, 1.0, 1.0)

# Case 2: Deterministic (α = 0, β = 0)
pop_deterministic = euler_maruyama(c_factor, 0.0, 0.0, 0.0, 0.0)

# === PLOT HISTOGRAM COMPARISON FOR BOTH CASES ===
bins = np.linspace(10, 60, 100)  # Population value range
fig, ax = plt.subplots(figsize=(10, 6))
check_labels = ["(1/8)th Simulation", "Halfway", "90% Completed"]
colors = ["blue", "green", "red"]  # Different colors for different times

for i, step in enumerate(pop_stochastic.keys()):
    # Stochastic case (solid lines with markers)
    hist_stoch, bin_edges = np.histogram(pop_stochastic[step], bins=bins, density=True)
    ax.plot((bin_edges[1:] + bin_edges[:-1]) / 2, hist_stoch, linestyle="-", color=colors[i], label=f"Stochastic {check_labels[i]}")

    # Deterministic case (dashed lines with markers)
    hist_det, bin_edges = np.histogram(pop_deterministic[step], bins=bins, density=True)
    ax.plot((bin_edges[1:] + bin_edges[:-1]) / 2, hist_det, linestyle="--", color=colors[i], label=f"Deterministic {check_labels[i]}")

ax.set_xlabel("Population Size")
ax.set_ylabel("Frequency")
ax.set_title("Comparison of Population Distributions (Stochastic vs Deterministic)")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)

plt.show()
