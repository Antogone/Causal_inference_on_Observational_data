import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/lalonde.csv")

# ── 1. ATE comparison forest plot ─────────────────────────────────────────────
estimates = {
    "Naive":  {"ate": 1124, "ci_low": None, "ci_high": None},
    "PSM":    {"ate": 1296, "ci_low": None, "ci_high": None},
    "IPW":    {"ate": 1180, "ci_low": None, "ci_high": None},
    "DiD":    {"ate": 1134, "ci_low": 447,  "ci_high": 1820},
}

names  = list(estimates.keys())
ates   = [estimates[n]["ate"] for n in names]
colors = ["#DC2626", "#0D9488", "#7C3AED", "#D97706"]

fig, ax = plt.subplots(figsize=(8, 5))

for i, (name, color) in enumerate(zip(names, colors)):
    ax.scatter(estimates[name]["ate"], i,
               color=color, s=100, zorder=3)
    if estimates[name]["ci_low"] is not None:
        ax.plot([estimates[name]["ci_low"],
                 estimates[name]["ci_high"]],
                [i, i], color=color, linewidth=2, alpha=0.7)

ax.axvline(1500, color="black", linestyle="--",
           linewidth=1.5, label="True ATE = $1,500")
ax.axvline(0, color="gray", linewidth=0.5)

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel("Estimated ATE ($)")
ax.set_title("ATE Estimates Across Methods\n(dashed = true causal effect)")

# Custom legend — colour + value for each method
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
           markersize=10, label=f"{name}: ${estimates[name]['ate']:,}")
    for name, color in zip(names, colors)
] + [
    Line2D([0], [0], color="black", linestyle="--",
           linewidth=1.5, label="True ATE: $1,500")
]
ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)
plt.tight_layout()
plt.savefig("outputs/ate_comparison.png", dpi=150)
plt.close()
print("Saved: outputs/ate_comparison.png")

# ── 2. Rosenbaum sensitivity analysis ─────────────────────────────────────────
# How strong would unmeasured confounding have to be to overturn our conclusion?
# Gamma = odds ratio of unmeasured confounder
# We test: at what Gamma does our PSM result become insignificant?

from scipy.stats import norm

def rosenbaum_bounds(outcomes_treated, outcomes_control, gamma_values):
    """
    Compute upper bound p-value under unmeasured confounding of strength gamma.
    Uses Wilcoxon signed-rank test bounds.
    """
    d = outcomes_treated - outcomes_control
    d = d[d != 0]
    n = len(d)

    # Ranks of absolute differences
    ranks = pd.Series(np.abs(d)).rank().values
    t_plus = ranks[d > 0].sum()  # observed test statistic

    results = []
    for gamma in gamma_values:
        # Upper bound on expected value and variance under gamma
        p_upper = gamma / (1 + gamma)
        mu_upper  = p_upper * n * (n + 1) / 2
        var_upper = p_upper * (1 - p_upper) * n * (n + 1) * (2*n + 1) / 6
        z_upper   = (t_plus - mu_upper) / np.sqrt(var_upper)
        p_upper_val = 1 - norm.cdf(z_upper)
        results.append({"gamma": gamma, "p_upper": p_upper_val})

    return pd.DataFrame(results)

treated_matched = df[df["treat"]==1]["re78"].values
# Use simple random sample of control as matched pairs for illustration
np.random.seed(42)
control_matched = df[df["treat"]==0]["re78"].sample(
    len(treated_matched), random_state=42
).values

gamma_values = np.arange(1.0, 3.1, 0.1)
bounds = rosenbaum_bounds(treated_matched, control_matched, gamma_values)

# Find critical gamma where p > 0.05
critical_gamma = bounds[bounds["p_upper"] > 0.05]["gamma"].min()

print("\n── Rosenbaum Sensitivity Analysis ──────────")
print(bounds.to_string(index=False))
print(f"\nCritical Gamma: {critical_gamma:.1f}")
print(f"Interpretation: Unmeasured confounding would need to create a "
      f"{critical_gamma:.1f}x odds ratio")
print(f"to overturn the conclusion at p=0.05")

# ── 3. Plot sensitivity curve ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(bounds["gamma"], bounds["p_upper"],
        color="#7C3AED", linewidth=2)
ax.axhline(0.05, color="red", linestyle="--",
           label="p = 0.05 threshold")
if not np.isnan(critical_gamma):
    ax.axvline(critical_gamma, color="gray", linestyle="--",
               label=f"Critical Γ = {critical_gamma:.1f}")
ax.set_xlabel("Gamma (strength of unmeasured confounding)")
ax.set_ylabel("Upper bound p-value")
ax.set_title("Rosenbaum Sensitivity Analysis\n"
             "How strong must unmeasured confounding be to overturn results?")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/sensitivity.png", dpi=150)
plt.close()
print("Saved: outputs/sensitivity.png")