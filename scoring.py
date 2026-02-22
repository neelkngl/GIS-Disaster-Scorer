import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from project_paths import processed_dir as proj_processed_dir

# ============================================================
# Goals
# - ZIP scores from COUNTY NRI + COUNTY→ZIP (RES_RATIO) crosswalk
# - Hazard sliders act as SMALL, personalized modifiers (no double counting)
# - Stable normalization: hazard norms are pre-normalized per hazard column
# - Produce BOTH zip_scores and county_scores to match county-geometry GeoJSON
# ============================================================

HAZARDS_DEFAULT = ["CFLD", "ERQK", "HRCN", "TRND", "WFIR", "CWAV", "HWAV", "DRGT"]

# -------------------------
# Helpers
# -------------------------
def zfill5(x: pd.Series) -> pd.Series:
    return x.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)

def safe_div(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return np.where((den == 0) | pd.isna(den), np.nan, num / den)

def pct_rank(s: pd.Series) -> pd.Series:
    """Stable normalization in [0,1] using percentile ranks (robust to outliers)."""
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    return s.rank(pct=True).fillna(0.0)

def weighted_avg(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy()
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).to_numpy()
    ws = w.sum()
    if ws <= 0:
        return 0.0
    return float((v * w).sum() / ws)

# -------------------------
# Load inputs
# -------------------------
def load_inputs(processed_root: Path):
    nri = pd.read_csv(processed_root / "nri_county_total.csv")
    cross = pd.read_csv(processed_root / "county_zip_conversion_processed.csv")
    metro = pd.read_csv(processed_root / "metro_snapshot_zhvi_markettemp.csv")
    haz_long = pd.read_csv(processed_root / "nri_county_hazard_long.csv", dtype={"GEOID": str})
    return nri, cross, metro, haz_long

# -------------------------
# County feature table (base + hazard norms)
# -------------------------
def build_county_features(nri: pd.DataFrame, haz_long: pd.DataFrame, hazards: list[str]) -> pd.DataFrame:
    # Base county metrics
    base = nri.copy()
    base["GEOID"] = zfill5(base["GEOID"])

    for c in ["RISK_SCORE", "EAL_VALT", "BUILDVALUE"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    # Risk ratio + "risk tax" per $100k
    base["risk_ratio"] = safe_div(base.get("EAL_VALT"), base.get("BUILDVALUE"))
    base["risk_tax_100k"] = base["risk_ratio"] * 100_000.0

    # Hazard table: choose best metric available
    metric_candidates = ["RISKS", "RISKV", "RISKR", "EALT", "EALS", "EALR"]
    metric = next((m for m in metric_candidates if m in haz_long.columns), None)

    haz_wide = None
    if metric is not None:
        h = haz_long[["GEOID", "Prefix", metric]].copy()
        h["GEOID"] = zfill5(h["GEOID"])
        h = h[h["Prefix"].isin(hazards)]
        h[metric] = pd.to_numeric(h[metric], errors="coerce")
        haz_wide = h.pivot(index="GEOID", columns="Prefix", values=metric)

        # Rename to explicit columns and compute percentile ranks per hazard
        haz_wide = haz_wide.reindex(columns=hazards)  # ensure all hazards present
        for hz in hazards:
            base[f"{hz}_raw"] = haz_wide[hz].values if hz in haz_wide.columns else np.nan
            base[f"{hz}_norm"] = pct_rank(base[f"{hz}_raw"])
    else:
        # No hazard metric: norms = 0
        for hz in hazards:
            base[f"{hz}_raw"] = np.nan
            base[f"{hz}_norm"] = 0.0

    # Stable base normalizations
    base["risk_norm"] = pct_rank(base["RISK_SCORE"])
    base["cost_norm"] = pct_rank(base["risk_tax_100k"])

    return base

# -------------------------
# COUNTY -> ZIP aggregation
# -------------------------
def county_to_zip(county_features: pd.DataFrame, cross: pd.DataFrame, hazards: list[str]) -> pd.DataFrame:
    c = cross.copy()
    c["COUNTY"] = zfill5(c["COUNTY"])
    c["ZIP"] = zfill5(c["ZIP"])
    wcol = "RES_RATIO" if "RES_RATIO" in c.columns else "TOT_RATIO"
    c[wcol] = pd.to_numeric(c[wcol], errors="coerce").fillna(0.0)
    c = c[c[wcol] > 0].copy()
    c = c.rename(columns={wcol: "w"})

    m = c.merge(county_features, left_on="COUNTY", right_on="GEOID", how="left")

    # Weighted averages for score-like features (risk_norm/cost_norm/hazard norms)
    avg_cols = ["risk_norm", "cost_norm", "RISK_SCORE", "risk_tax_100k"]
    avg_cols += [f"{hz}_norm" for hz in hazards]

    # Weighted sums for EAL & BUILD to recompute ratio correctly
    for col in ["EAL_VALT", "BUILDVALUE"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")
    m["eal_w"] = pd.to_numeric(m.get("EAL_VALT"), errors="coerce") * m["w"]
    m["build_w"] = pd.to_numeric(m.get("BUILDVALUE"), errors="coerce") * m["w"]

    # multiply avg cols
    for col in avg_cols:
        if col in m.columns:
            m[f"{col}_w"] = pd.to_numeric(m[col], errors="coerce") * m["w"]
        else:
            m[f"{col}_w"] = 0.0

    agg = {"w": "sum", "eal_w": "sum", "build_w": "sum"}
    for col in avg_cols:
        agg[f"{col}_w"] = "sum"

    z = m.groupby("ZIP", as_index=False).agg(agg).rename(columns={"w": "w_sum"})

    # Weighted averages
    for col in avg_cols:
        z[col] = safe_div(z[f"{col}_w"], z["w_sum"])

    # Recompute ratio/tax at ZIP level based on weighted sums
    z["risk_ratio"] = safe_div(z["eal_w"], z["build_w"])
    z["risk_tax_100k"] = z["risk_ratio"] * 100_000.0
    z["cost_norm"] = pct_rank(z["risk_tax_100k"])  # override cost_norm based on zip tax

    keep = ["ZIP", "RISK_SCORE", "risk_norm", "cost_norm", "risk_ratio", "risk_tax_100k"]
    keep += [f"{hz}_norm" for hz in hazards]
    return z[keep]

# -------------------------
# Scoring: base + small hazard modifier
# -------------------------
def compute_scores(df: pd.DataFrame, weights: dict, hazards: list[str], *,
                   risk_w: float = 0.65,
                   cost_w: float = 0.35,
                   hazard_w: float = 0.12) -> pd.DataFrame:
    """
    - risk_w + cost_w define the consistent baseline.
    - hazard_w is the "small personalized modifier" strength.
    """

    out = df.copy()

    # Normalize slider weights to sum=1 (so number of sliders touched doesn't scale magnitude)
    w = {hz: float(weights.get(hz, 0.0)) for hz in hazards}
    w_sum = sum(max(0.0, v) for v in w.values())

    if w_sum <= 0:
        out["hazard_penalty"] = 0.0
    else:
        # Weighted average of hazard norms
        hazard_terms = []
        for hz in hazards:
            col = f"{hz}_norm"
            if col not in out.columns:
                continue
            hazard_terms.append((max(0.0, w[hz]) / w_sum) * out[col].fillna(0.0))
        out["hazard_penalty"] = sum(hazard_terms) if hazard_terms else 0.0

    # Base penalties (no double-counting with hazard component)
    out["base_penalty"] = risk_w * out["risk_norm"].fillna(0.0) + cost_w * out["cost_norm"].fillna(0.0)

    # Final score: higher is better
    out["score_raw"] = 1.0 - out["base_penalty"] - hazard_w * out["hazard_penalty"]

    # Stable 0-100 score via percentile rank (not min-max)
    out["score_0_100"] = 100.0 * out["score_raw"].rank(pct=True).fillna(0.0)

    # Contributions for explainability
    out["contrib_risk"] = -risk_w * out["risk_norm"].fillna(0.0)
    out["contrib_cost"] = -cost_w * out["cost_norm"].fillna(0.0)
    out["contrib_hazard"] = -hazard_w * out["hazard_penalty"]

    return out

# -------------------------
# Main entry: produce ZIP + COUNTY score tables
# -------------------------
def prepare_and_score(weights: dict, processed_root: Path, hazards: list[str] = None):
    hazards = hazards or HAZARDS_DEFAULT

    nri, cross, metro, haz_long = load_inputs(processed_root)

    county_feat = build_county_features(nri, haz_long, hazards)

    # ZIP-level features
    zip_feat = county_to_zip(county_feat, cross, hazards)

    # County-level scoring (direct)
    county_scores = compute_scores(
        county_feat[["GEOID", "RISK_SCORE", "risk_norm", "cost_norm", "risk_ratio", "risk_tax_100k"] + [f"{hz}_norm" for hz in hazards]].copy(),
        weights=weights,
        hazards=hazards
    ).rename(columns={"GEOID": "GEOID"})

    # ZIP-level scoring
    zip_scores = compute_scores(
        zip_feat,
        weights=weights,
        hazards=hazards
    )

    # NOTE: Metro hotness is currently not joinable to ZIP with existing processed datasets.
    # We keep it out of the score for now (otherwise it's a constant and doesn't change ranks).

    return zip_scores, county_scores

def parse_args():
    parser = argparse.ArgumentParser(description="Compute ZIP + COUNTY scores with hazard sliders (0-100 avoid).")
    for code in HAZARDS_DEFAULT:
        parser.add_argument(f"--{code}", type=float, default=0.0, help=f"Avoidance slider for {code} (0-100)")
    parser.add_argument("--out-dir", type=str, default=None, help="Processed output directory (defaults to project processed)")
    return parser.parse_args()

def main():
    args = parse_args()
    weights = {code: getattr(args, code) for code in HAZARDS_DEFAULT}

    processed_root = Path(proj_processed_dir()) if args.out_dir is None else Path(args.out_dir)
    processed_root = processed_root.resolve()
    processed_root.mkdir(parents=True, exist_ok=True)

    zip_scores, county_scores = prepare_and_score(weights, processed_root)

    zip_out = processed_root / "zip_scores_balanced.csv"
    county_out = processed_root / "county_scores_balanced.csv"

    zip_scores.to_csv(zip_out, index=False)
    county_scores.to_csv(county_out, index=False)

    print(f"[OK] wrote {zip_out}")
    print(f"[OK] wrote {county_out}")
    print(zip_scores.head())

if __name__ == "__main__":
    main()
