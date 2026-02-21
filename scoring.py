import argparse
import json
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

def load_inputs(processed_root: Path):
    nri = pd.read_csv(processed_root / "nri_county_total.csv")
    cross = pd.read_csv(processed_root / "county_zip_conversion_processed.csv")
    metro = pd.read_csv(processed_root / "metro_snapshot_zhvi_markettemp.csv")
    # hazard long contains per-county × Prefix rows
    haz_long = pd.read_csv(processed_root / "nri_county_hazard_long.csv", dtype={"GEOID": str})
    return nri, cross, metro, haz_long

def prepare_and_score(weights: dict, processed_root: Path):
    nri, cross, metro, haz_long = load_inputs(processed_root)

    # Clean keys
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

    # -------------------------
    # Hazard component: build per-county weighted hazard score
    # -------------------------
    # prefer RISKS -> RISKV -> RISKR
    metric_candidates = ["RISKS", "RISKV", "RISKR"]
    metric = None
    for m in metric_candidates:
        if m in haz_long.columns:
            metric = m
            break

    if metric is None:
        # no hazard metric present; set zero hazard component
        df["hazard_score"] = 0.0
    else:
        # ensure numeric and pivot to wide per GEOID
        haz_long_num = haz_long[["GEOID", "Prefix", metric]].copy()
        haz_long_num[metric] = pd.to_numeric(haz_long_num[metric], errors="coerce")
        haz_wide = haz_long_num.pivot(index="GEOID", columns="Prefix", values=metric)

        # ensure GEOID zfill
        haz_wide.index = zfill5(haz_wide.index.to_series())

        # normalize each hazard column (min-max) to 0-1
        haz_norm = haz_wide.apply(lambda s: (s - s.min()) / (s.max() - s.min()) if s.notna().any() else s)

        # build weighted sum using weights dict (weights are 0-100 from sliders)
        # convert weights to 0-1 multiplier
        weights_frac = {k: float(v) / 100.0 for k, v in (weights or {}).items()}

        # ensure all prefixes present as columns
        for p in list(weights_frac.keys()):
            if p not in haz_norm.columns:
                haz_norm[p] = 0.0

        # compute per-GEOID hazard_score
        if len(weights_frac) > 0:
            # multiply normalized hazard columns by respective weight and sum
            weighted_cols = [haz_norm[p].fillna(0.0) * weights_frac.get(p, 0.0) for p in weights_frac]
            hazard_score_series = pd.concat(weighted_cols, axis=1).sum(axis=1)
        else:
            hazard_score_series = pd.Series(0.0, index=haz_norm.index)

        hazard_score_series.name = "hazard_score"

        # merge hazard_score back into df by GEOID
        hazard_df = hazard_score_series.rename_axis("GEOID").reset_index()
        df = df.merge(hazard_df, left_on="GEOID", right_on="GEOID", how="left")
        df["hazard_score"] = df["hazard_score"].fillna(0.0)

    # if hazard_score wasn't added above (no metric), ensure present
    if "hazard_score" not in df.columns:
        df["hazard_score"] = 0.0

    # include hazard weighted by RES_RATIO when aggregating to ZIP
    df["hazard_w"] = df["hazard_score"] * df["RES_RATIO"]

    zip_risk = (
        df.groupby("ZIP", as_index=False)
          .agg({
              "risk_w": "sum",
              "RES_RATIO": "sum",
              "eal_w": "sum",
              "build_w": "sum",
              "hazard_w": "sum",
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

    # hazard component aggregated to ZIP (weighted average)
    zip_risk["hazard_component"] = safe_div(zip_risk["hazard_w"], zip_risk["RES_RATIO"])

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
    # incorporate hazard_component into risk normalization
    # normalize components first
    zip_risk["risk_norm_raw"] = minmax(zip_risk["risk_score"])
    zip_risk["hazard_norm_raw"] = minmax(zip_risk["hazard_component"])

    # combined raw risk (simple additive) then renormalize
    zip_risk["risk_norm"] = minmax(zip_risk["risk_norm_raw"].fillna(0.0) + zip_risk["hazard_norm_raw"].fillna(0.0))
    zip_risk["cost_norm"] = minmax(zip_risk["risk_tax_100k"])

    # =====================
    # SCORING MODEL
    # =====================

    # Balanced buyer profile (these can be tuned independently from hazard sliders)
    HOTNESS_W = 0.9
    RISK_W = 0.8
    COST_W = 0.7

    zip_risk["score_raw"] = (
        HOTNESS_W * zip_risk["hotness_norm"]
        - RISK_W * zip_risk["risk_norm"]
        - COST_W * zip_risk["cost_norm"]
    )

    zip_risk["score_0_100"] = 100 * minmax(zip_risk["score_raw"])

    return zip_risk

def parse_args():
    parser = argparse.ArgumentParser(description="Compute ZIP scores with optional hazard sliders (0-100 avoid).")
    # hazard sliders
    for code in ["CFLD", "ERQK", "HRCN", "TRND", "WFIR", "CWAV", "HWAV", "DRGT"]:
        parser.add_argument(f"--{code}", type=float, default=0.0, help=f"Avoidance slider for {code} (0-100)")
    parser.add_argument("--out-dir", type=str, default=None, help="Processed output directory (defaults to project processed)")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = {code: getattr(args, code) for code in ["CFLD", "ERQK", "HRCN", "TRND", "WFIR", "CWAV", "HWAV", "DRGT"]}
    processed_root = Path(proj_processed_dir()) if args.out_dir is None else Path(args.out_dir)
    processed_root = processed_root.resolve()
    zip_risk = prepare_and_score(weights, processed_root)

    out_base = processed_root
    out_base.mkdir(parents=True, exist_ok=True)
    out_path = out_base / "zip_scores_balanced.csv"
    zip_risk.to_csv(out_path, index=False)
    print(zip_risk.head())


if __name__ == "__main__":
    main()