import pandas as pd
import numpy as np
import os

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

np.random.seed(42)
n = 614  # original Lalonde sample size

# ── Simulate covariates matching Lalonde statistics ───────────────────────────
age        = np.random.normal(25, 7, n).clip(17, 55).astype(int)
education  = np.random.normal(10, 2, n).clip(0, 16).astype(int)
black      = np.random.binomial(1, 0.83, n)
hispanic   = np.random.binomial(1, 0.06, n)
married    = np.random.binomial(1, 0.17, n)
nodegree   = np.random.binomial(1, 0.71, n)
re74       = np.random.exponential(2000, n) * (np.random.binomial(1, 0.5, n))
re75       = np.random.exponential(1500, n) * (np.random.binomial(1, 0.5, n))

# ── Treatment assignment — not random, depends on covariates ──────────────────
# People with lower earnings and less education more likely to seek training
propensity = 1 / (1 + np.exp(-(
    -1.5
    - 0.02 * age
    + 0.05 * education
    - 0.0001 * re74
    - 0.0001 * re75
    + 0.3  * black
    + 0.2  * nodegree
)))
treat = np.random.binomial(1, propensity, n)

# ── Outcome — earnings in 1978 ────────────────────────────────────────────────
# True causal effect of training = +$1500
true_effect = 1500

re78 = (
    2000
    + 0.3  * re74
    + 0.3  * re75
    + 500  * education
    - 100  * nodegree
    + true_effect * treat          # causal effect
    + np.random.normal(0, 3000, n) # noise
).clip(0)

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame({
    "treat":     treat,
    "age":       age,
    "education": education,
    "black":     black,
    "hispanic":  hispanic,
    "married":   married,
    "nodegree":  nodegree,
    "re74":      re74,
    "re75":      re75,
    "re78":      re78,
})

df.to_csv("data/lalonde.csv", index=False)

print("Dataset generated and saved to data/lalonde.csv")
print("\nShape:", df.shape)
print("\nTreatment distribution:\n", df["treat"].value_counts())
print("\nOutcome stats by group:")
print(df.groupby("treat")["re78"].describe())
print(f"\nTrue causal effect: ${true_effect}")
print(f"Naive estimate:     ${df[df.treat==1].re78.mean() - df[df.treat==0].re78.mean():.0f}")