import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/lalonde.csv")

# ── Difference in Differences ─────────────────────────────────────────────────
# We use re75 as pre-treatment outcome and re78 as post-treatment outcome
# DiD = (treated_after - treated_before) - (control_after - control_before)

# ── 1. Manual DiD calculation ─────────────────────────────────────────────────
treated_before = df[df["treat"]==1]["re75"].mean()
treated_after  = df[df["treat"]==1]["re78"].mean()
control_before = df[df["treat"]==0]["re75"].mean()
control_after  = df[df["treat"]==0]["re78"].mean()

did_manual = (treated_after - treated_before) - (control_after - control_before)

print("── Manual DiD ───────────────────────────────")
print(f"Treated before (re75): ${treated_before:,.0f}")
print(f"Treated after  (re78): ${treated_after:,.0f}")
print(f"Control before (re75): ${control_before:,.0f}")
print(f"Control after  (re78): ${control_after:,.0f}")
print(f"\nTreated difference:    ${treated_after - treated_before:,.0f}")
print(f"Control difference:    ${control_after - control_before:,.0f}")
print(f"DiD estimate:          ${did_manual:,.0f}")

# ── 2. DiD via OLS (panel format) ─────────────────────────────────────────────
# Reshape to long format: one row per person per period
pre  = df[["treat", "re75"]].copy()
pre["period"]  = 0
pre["outcome"] = pre["re75"]
pre = pre.drop(columns="re75")

post = df[["treat", "re78"]].copy()
post["period"]  = 1
post["outcome"] = post["re78"]
post = post.drop(columns="re78")

panel = pd.concat([pre, post], ignore_index=True)

# DiD regression: outcome = β0 + β1*treat + β2*period + β3*treat*period
# β3 is the DiD estimate — the interaction term
model  = smf.ols("outcome ~ treat + period + treat:period", data=panel).fit()

did_ols = model.params["treat:period"]
did_ci  = model.conf_int().loc["treat:period"]

print(f"\n── DiD via OLS ──────────────────────────────")
print(f"DiD coefficient:  ${did_ols:,.0f}")
print(f"95% CI:           [${did_ci[0]:,.0f}, ${did_ci[1]:,.0f}]")
print(f"P-value:          {model.pvalues['treat:period']:.4f}")
print(f"True ATE:         $1,500")
print(f"Bias remaining:   ${did_ols - 1500:,.0f}")
print(f"\n{model.summary()}")