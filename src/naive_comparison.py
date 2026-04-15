import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/lalonde.csv")

# ── 1. Naive estimate ─────────────────────────────────────────────────────────
treated  = df[df["treat"] == 1]["re78"]
control  = df[df["treat"] == 0]["re78"]

naive_ate = treated.mean() - control.mean()

print("── Naive Estimate ───────────────────────────")
print(f"Treated mean:  ${treated.mean():,.0f}")
print(f"Control mean:  ${control.mean():,.0f}")
print(f"Naive ATE:     ${naive_ate:,.0f}")
print(f"True ATE:      $1,500")
print(f"Bias:          ${naive_ate - 1500:,.0f}")

# ── 2. Covariate balance check ────────────────────────────────────────────────
# Standardised mean difference (SMD) for each covariate
# SMD > 0.1 indicates meaningful imbalance
covariates = ["age", "education", "black", "hispanic",
              "married", "nodegree", "re74", "re75"]

smds = []
for col in covariates:
    mean_t = df[df["treat"] == 1][col].mean()
    mean_c = df[df["treat"] == 0][col].mean()
    std_p  = np.sqrt(
        (df[df["treat"] == 1][col].var() + df[df["treat"] == 0][col].var()) / 2
    )
    smd = (mean_t - mean_c) / std_p
    smds.append({"covariate": col, "smd": smd,
                 "mean_treated": mean_t, "mean_control": mean_c})

balance_df = pd.DataFrame(smds)
print("\n── Covariate Balance (before matching) ──────")
print(balance_df.to_string(index=False))

# ── 3. Love plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

colors = ["#DC2626" if abs(s) > 0.1 else "#0D9488"
          for s in balance_df["smd"]]

ax.barh(balance_df["covariate"], balance_df["smd"],
        color=colors, alpha=0.8)
ax.axvline(0,    color="black", linewidth=0.8)
ax.axvline(0.1,  color="gray", linestyle="--",
           linewidth=0.8, label="|SMD| = 0.1 threshold")
ax.axvline(-0.1, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Standardised Mean Difference (SMD)")
ax.set_title("Covariate Balance Before Matching\n"
             "(red = |SMD| > 0.1, imbalanced)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/covariate_balance_before.png", dpi=150)
plt.close()
print("\nSaved: outputs/covariate_balance_before.png")