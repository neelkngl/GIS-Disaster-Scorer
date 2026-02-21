import pandas as pd
import numpy as np
from pathlib import Path
from project_paths import processed_dir as proj_processed_dir

# =====================
# Helpers
# =====================

def zfill5(x):
    return x.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)

def minmax(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.min()) / (s.max() - s.min())

def safe_div(a, b):
    return np.where((b == 0) | pd.isna(b), np.nan, a / b)

# =====================
# Load data
# =====================

nri_base = Path(proj_processed_dir())
nri = pd.read_csv(nri_base / "nri_county_total.csv")
cross = pd.read_csv(nri_base / "county_zip_conversion_processed.csv")
metro = pd.read_csv(nri_base / "metro_snapshot_zhvi_markettemp.csv")

# =====================
# Clean keys
# =====================

nri["GEOID"] = zfill5(nri["GEOID"])
cross["COUNTY"] = zfill5(cross["COUNTY"])
cross["ZIP"] = zfill5(cross["ZIP"])

cross["RES_RATIO"] = pd.to_numeric(
    cross.get("RES_RATIO", cross.get("TOT_RATIO")),
    errors="coerce"
).fillna(0)

cross = cross[cross["RES_RATIO"] > 0]

# =====================
# COUNTY → ZIP RISK
# =====================

df = cross.merge(
    nri,
    left_on="COUNTY",
    right_on="GEOID",
    how="left"
)

df["RISK_SCORE"] = pd.to_numeric(df["RISK_SCORE"], errors="coerce")
df["EAL_VALT"] = pd.to_numeric(df["EAL_VALT"], errors="coerce")
df["BUILDVALUE"] = pd.to_numeric(df["BUILDVALUE"], errors="coerce")

# weighted components
df["risk_w"] = df["RISK_SCORE"] * df["RES_RATIO"]
df["eal_w"] = df["EAL_VALT"] * df["RES_RATIO"]
df["build_w"] = df["BUILDVALUE"] * df["RES_RATIO"]

zip_risk = (
    df.groupby("ZIP", as_index=False)
      .agg({
          "risk_w": "sum",
          "RES_RATIO": "sum",
          "eal_w": "sum",
          "build_w": "sum"
      })
)

zip_risk["risk_score"] = safe_div(
    zip_risk["risk_w"],
    zip_risk["RES_RATIO"]
)

zip_risk["risk_ratio"] = safe_div(
    zip_risk["eal_w"],
    zip_risk["build_w"]
)

# yearly expected loss per $100k property
zip_risk["risk_tax_100k"] = zip_risk["risk_ratio"] * 100_000

# =====================
# METRO HOTNESS BASELINE
# =====================

metro["market_temp_latest"] = pd.to_numeric(
    metro["market_temp_latest"], errors="coerce"
)

# national housing demand baseline
national_hotness = metro["market_temp_latest"].mean()

zip_risk["hotness_raw"] = national_hotness

# =====================
# NORMALIZATION
# =====================

zip_risk["hotness_norm"] = 1.0  # constant baseline
zip_risk["risk_norm"] = minmax(zip_risk["risk_score"])
zip_risk["cost_norm"] = minmax(zip_risk["risk_tax_100k"])

# =====================
# SCORING MODEL
# =====================

# Balanced buyer profile
HOTNESS_W = 0.9
RISK_W = 0.8
COST_W = 0.7

zip_risk["score_raw"] = (
    HOTNESS_W * zip_risk["hotness_norm"]
    - RISK_W * zip_risk["risk_norm"]
    - COST_W * zip_risk["cost_norm"]
)

zip_risk["score_0_100"] = 100 * minmax(zip_risk["score_raw"])

# =====================
# Save output
# =====================

out_base = Path(proj_processed_dir())
out_base.mkdir(parents=True, exist_ok=True)
zip_risk.to_csv(out_base / "zip_scores_balanced.csv", index=False)

print(zip_risk.head())