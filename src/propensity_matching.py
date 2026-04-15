import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/lalonde.csv")

covariates = ["age", "education", "black", "hispanic",
              "married", "nodegree", "re74", "re75"]

# ── 1. Estimate propensity scores ─────────────────────────────────────────────
# P(treated | covariates) via logistic regression
X = df[covariates]
y = df["treat"]

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X, y)

df["propensity_score"] = lr.predict_proba(X)[:, 1]

print("── Propensity Score Distribution ────────────")
print(df.groupby("treat")["propensity_score"].describe())

# ── 2. Check overlap (common support) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df[df["treat"]==0]["propensity_score"], bins=30,
        alpha=0.6, color="#7C3AED", label="Control")
ax.hist(df[df["treat"]==1]["propensity_score"], bins=30,
        alpha=0.6, color="#0D9488", label="Treated")
ax.set_xlabel("Propensity score")
ax.set_ylabel("Count")
ax.set_title("Propensity score distribution — overlap check")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/propensity_overlap.png", dpi=150)
plt.close()
print("Saved: outputs/propensity_overlap.png")

# ── 3. 1-to-1 nearest neighbour matching without replacement ──────────────────
treated_df = df[df["treat"] == 1].copy()
control_df = df[df["treat"] == 0].copy()

# Fit NN on control propensity scores
nn = NearestNeighbors(n_neighbors=1)
nn.fit(control_df[["propensity_score"]])

# Find nearest control for each treated unit
distances, indices = nn.kneighbors(treated_df[["propensity_score"]])

matched_control = control_df.iloc[indices.flatten()].copy()

# ── 4. Estimate ATT (average treatment effect on the treated) ─────────────────
att_psm = (treated_df["re78"].values - matched_control["re78"].values).mean()

print(f"\n── Propensity Score Matching ─────────────────")
print(f"Matched pairs:  {len(treated_df)}")
print(f"PSM ATT:        ${att_psm:,.0f}")
print(f"True ATE:       $1,500")
print(f"Bias remaining: ${att_psm - 1500:,.0f}")

# ── 5. Check covariate balance after matching ─────────────────────────────────
smds_after = []
for col in covariates:
    mean_t  = treated_df[col].mean()
    mean_c  = matched_control[col].mean()
    std_p   = np.sqrt(
        (treated_df[col].var() + matched_control[col].var()) / 2
    )
    smd = (mean_t - mean_c) / std_p
    smds_after.append({"covariate": col, "smd_after": smd})

balance_after = pd.DataFrame(smds_after)
print("\n── Covariate Balance After Matching ──────────")
print(balance_after.to_string(index=False))

# ── 6. Before vs after balance plot ───────────────────────────────────────────
smds_before = {
    "age": -0.094, "education": 0.026, "black": 0.112,
    "hispanic": -0.001, "married": -0.002, "nodegree": 0.137,
    "re74": -0.174, "re75": -0.007
}

fig, ax = plt.subplots(figsize=(8, 5))
y_pos = range(len(covariates))

ax.scatter([smds_before[c] for c in covariates], y_pos,
           color="#DC2626", label="Before matching", zorder=3, s=60)
ax.scatter(balance_after["smd_after"], y_pos,
           color="#0D9488", label="After matching", zorder=3, s=60)

for i, col in enumerate(covariates):
    ax.plot([smds_before[col], balance_after.iloc[i]["smd_after"]],
            [i, i], color="gray", linewidth=0.8, alpha=0.5)

ax.axvline(0,    color="black", linewidth=0.8)
ax.axvline(0.1,  color="gray", linestyle="--", linewidth=0.8)
ax.axvline(-0.1, color="gray", linestyle="--", linewidth=0.8)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(covariates)
ax.set_xlabel("Standardised Mean Difference")
ax.set_title("Covariate Balance Before and After PSM")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/covariate_balance_after_psm.png", dpi=150)
plt.close()
print("Saved: outputs/covariate_balance_after_psm.png")