import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/lalonde.csv")

covariates = ["age", "education", "black", "hispanic",
              "married", "nodegree", "re74", "re75"]

# ── 1. Estimate propensity scores ─────────────────────────────────────────────
X = df[covariates]
y = df["treat"]

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X, y)
df["ps"] = lr.predict_proba(X)[:, 1]

# ── 2. Compute IPW weights ────────────────────────────────────────────────────
# Treated:  weight = 1 / P(treated | X)
# Control:  weight = 1 / P(not treated | X)
df["weight"] = np.where(
    df["treat"] == 1,
    1 / df["ps"],
    1 / (1 - df["ps"])
)

# Stabilised weights — multiply by marginal treatment probability
# This reduces variance of the estimator
p_treat = df["treat"].mean()
df["weight_stable"] = np.where(
    df["treat"] == 1,
    p_treat / df["ps"],
    (1 - p_treat) / (1 - df["ps"])
)

print("── IPW Weight Distribution ──────────────────")
print(df.groupby("treat")["weight_stable"].describe())

# ── 3. Compute IPW ATE ────────────────────────────────────────────────────────
# Weighted means
treated_wmean = np.average(
    df[df["treat"]==1]["re78"],
    weights=df[df["treat"]==1]["weight_stable"]
)
control_wmean = np.average(
    df[df["treat"]==0]["re78"],
    weights=df[df["treat"]==0]["weight_stable"]
)

ipw_ate = treated_wmean - control_wmean

print(f"\n── IPW Estimate ─────────────────────────────")
print(f"Weighted treated mean:  ${treated_wmean:,.0f}")
print(f"Weighted control mean:  ${control_wmean:,.0f}")
print(f"IPW ATE:                ${ipw_ate:,.0f}")
print(f"True ATE:               $1,500")
print(f"Bias remaining:         ${ipw_ate - 1500:,.0f}")