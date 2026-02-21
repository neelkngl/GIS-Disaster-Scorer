"""
preprocess.py (CSV-only pipeline)

INPUTS (expected in same folder as this script unless --raw-dir is passed):
- Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
- Metro_market_temp_index_uc_sfrcondo_month.csv
- NRI_Table_Counties.csv
- NRI_HazardInfo.csv
- COUNTY_ZIP_CONVERSION.csv
- NRIDataDictionary.csv (optional)

OUTPUTS (written to ./data/processed unless --out-dir is passed):
1) metro_snapshot_zhvi_markettemp.csv
   - one row per metro with latest value + 5y % change + 10y slope for ZHVI and market temp.

2) county_zip_conversion_processed.csv
   - COUNTY and ZIP as zero-padded strings (5 digits), safe for joins.

3) nri_county_total.csv
   - one row per county (GEOID=STCOFIPS), core overall metrics only (good for base scoring / choropleth).

4) nri_county_hazard_long.csv
   - one row per county × hazard, with hazard Prefix + Hazard name preserved, and hazard-specific metrics.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------
# Helpers
# ---------------------------

DATE_COL_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")


def detect_date_columns(cols: List[str]) -> List[str]:
    return [c for c in cols if isinstance(c, str) and DATE_COL_RE.match(c)]


def melt_wide_to_long(df: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
    date_cols = detect_date_columns(df.columns.tolist())
    if not date_cols:
        raise ValueError("No YYYY-MM or YYYY-MM-DD date columns detected.")

    long_df = df.melt(id_vars=id_cols, value_vars=date_cols, var_name="date", value_name="value")
    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["date", "value"])
    return long_df


def compute_trend_metrics(long_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    For each group:
      - latest_date, latest_value
      - pct_change_5y: (latest / value_at_or_before(latest-5y) - 1)*100
      - slope_per_year_10y: least-squares slope using last 10 years of monthly points
    """
    out = []
    for key, g in long_df.groupby(group_col, sort=False):
        g = g.sort_values("date")
        if g.empty:
            continue

        latest = g.iloc[-1]
        latest_date = latest["date"]
        latest_value = float(latest["value"])

        # 5-year % change
        five_years_ago = latest_date - pd.DateOffset(years=5)
        g5 = g[g["date"] <= five_years_ago]
        pct_5y = np.nan
        if not g5.empty:
            base_val = float(g5.iloc[-1]["value"])
            if base_val != 0:
                pct_5y = (latest_value / base_val - 1.0) * 100.0

        # 10-year slope (per year)
        ten_years_ago = latest_date - pd.DateOffset(years=10)
        g10 = g[g["date"] >= ten_years_ago]
        slope = np.nan
        if len(g10) >= 24:  # at least ~2 years of monthly points
            x = (g10["date"] - g10["date"].iloc[0]).dt.days / 365.25
            y = g10["value"].astype(float)
            x_mean = x.mean()
            y_mean = y.mean()
            denom = ((x - x_mean) ** 2).sum()
            if denom != 0:
                slope = (((x - x_mean) * (y - y_mean)).sum()) / denom

        out.append({
            group_col: key,
            "latest_date": latest_date.date().isoformat(),
            "latest_value": latest_value,
            "pct_change_5y": None if pd.isna(pct_5y) else float(pct_5y),
            "slope_per_year_10y": None if pd.isna(slope) else float(slope),
        })

    return pd.DataFrame(out)


def zfill_series(s: pd.Series, width: int) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(width)


def must_exist(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file for {label}: {path}")


# ---------------------------
# Main processing steps
# ---------------------------

def process_metro(raw_dir: Path, out_dir: Path) -> Path:
    zhvi_path = raw_dir / "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    temp_path = raw_dir / "Metro_market_temp_index_uc_sfrcondo_month.csv"
    must_exist(zhvi_path, "ZHVI metro")
    must_exist(temp_path, "Market temp metro")

    zhvi = pd.read_csv(zhvi_path, low_memory=False)
    temp = pd.read_csv(temp_path, low_memory=False)

    # Determine ID columns (non-date columns)
    zhvi_date_cols = detect_date_columns(zhvi.columns.tolist())
    zhvi_id_cols = [c for c in zhvi.columns if c not in zhvi_date_cols]
    temp_date_cols = detect_date_columns(temp.columns.tolist())
    temp_id_cols = [c for c in temp.columns if c not in temp_date_cols]

    # Wide->long
    zhvi_long = melt_wide_to_long(zhvi, zhvi_id_cols)
    temp_long = melt_wide_to_long(temp, temp_id_cols)

    # Choose a stable grouping key
    group_key = "RegionID" if "RegionID" in zhvi_long.columns else "RegionName"
    if group_key not in temp_long.columns:
        # fallback: use RegionName if RegionID missing in temp
        group_key = "RegionName"

    zhvi_metrics = compute_trend_metrics(zhvi_long.rename(columns={"value": "value"}), group_key).rename(columns={
        "latest_date": "zhvi_latest_date",
        "latest_value": "zhvi_latest_value",
        "pct_change_5y": "zhvi_pct_change_5y",
        "slope_per_year_10y": "zhvi_slope_per_year_10y",
    })

    temp_metrics = compute_trend_metrics(temp_long.rename(columns={"value": "value"}), group_key).rename(columns={
        "latest_date": "temp_latest_date",
        "latest_value": "market_temp_latest",
        "pct_change_5y": "market_temp_pct_change_5y",
        "slope_per_year_10y": "market_temp_slope_per_year_10y",
    })

    snapshot = zhvi_metrics.merge(temp_metrics, on=group_key, how="outer")

    # Add human-readable columns from ZHVI file if present
    meta_cols = [c for c in ["RegionID", "RegionName", "StateName", "State", "Metro", "SizeRank"] if c in zhvi.columns]
    if meta_cols:
        meta = zhvi[meta_cols].drop_duplicates(subset=[c for c in ["RegionID", "RegionName"] if c in meta_cols][:1])
        # join on group_key if present, else just merge on whichever exists
        if group_key in meta.columns:
            snapshot = meta.merge(snapshot, on=group_key, how="right")

    # Fix types
    if "SizeRank" in snapshot.columns:
        snapshot["SizeRank"] = pd.to_numeric(snapshot["SizeRank"], errors="coerce").astype("Int64")

    out_path = out_dir / "metro_snapshot_zhvi_markettemp.csv"
    snapshot.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path}  rows={len(snapshot):,} cols={snapshot.shape[1]}")
    return out_path


def process_county_zip(raw_dir: Path, out_dir: Path) -> Path:
    cz_path = raw_dir / "COUNTY_ZIP_CONVERSION.csv"
    must_exist(cz_path, "county-zip conversion")
    cz = pd.read_csv(cz_path, low_memory=False)

    # Enforce zero-padded strings
    if "COUNTY" in cz.columns:
        cz["COUNTY"] = zfill_series(cz["COUNTY"], 5)
    if "ZIP" in cz.columns:
        cz["ZIP"] = zfill_series(cz["ZIP"], 5)

    out_path = out_dir / "county_zip_conversion_processed.csv"
    cz.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path}  rows={len(cz):,} cols={cz.shape[1]}")
    return out_path


def process_nri(raw_dir: Path, out_dir: Path) -> Dict[str, Path]:
    nri_path = raw_dir / "NRI_Table_Counties.csv"
    haz_path = raw_dir / "NRI_HazardInfo.csv"
    must_exist(nri_path, "NRI_Table_Counties")
    must_exist(haz_path, "NRI_HazardInfo")

    nri = pd.read_csv(nri_path, low_memory=False)
    haz = pd.read_csv(haz_path, low_memory=False)

    # Normalize column names
    nri.columns = [c.strip() if isinstance(c, str) else c for c in nri.columns]
    haz.columns = [c.strip() if isinstance(c, str) else c for c in haz.columns]

    # County GEOID
    # NRI has STCOFIPS (e.g., 1001 for Autauga AL) -> should be 5 digits.
    if "STCOFIPS" in nri.columns:
        nri["GEOID"] = zfill_series(nri["STCOFIPS"], 5)
    elif "NRI_ID" in nri.columns:
        # NRI_ID looks like "C01001" -> take last 5
        nri["GEOID"] = nri["NRI_ID"].astype(str).str.replace(r"[^0-9]", "", regex=True).str[-5:].str.zfill(5)
    else:
        raise ValueError("Could not find STCOFIPS or NRI_ID to build GEOID.")

    # -----------------------
    # A) County total table
    # -----------------------
    total_cols_wanted = [
        "GEOID",
        "STATE", "STATEABBRV", "STATEFIPS",
        "COUNTY", "COUNTYTYPE", "COUNTYFIPS", "STCOFIPS",
        "POPULATION", "BUILDVALUE", "AGRIVALUE", "AREA",
        "RISK_VALUE", "RISK_SCORE", "RISK_RATNG", "RISK_SPCTL",
        "EAL_SCORE", "EAL_RATNG", "EAL_SPCTL",
        "EAL_VALT", "EAL_VALB", "EAL_VALP", "EAL_VALPE", "EAL_VALA",
        "NRI_VER",
    ]
    total_cols = [c for c in total_cols_wanted if c in nri.columns]
    nri_total = nri[total_cols].copy()

    out_total = out_dir / "nri_county_total.csv"
    nri_total.to_csv(out_total, index=False)
    print(f"[OK] Wrote {out_total}  rows={len(nri_total):,} cols={nri_total.shape[1]}")

    # -----------------------
    # B) County × hazard long table
    # -----------------------
    # HazardInfo provides Prefix mapping (e.g., WNTW, CFLD, WFIR, etc.)
    if "Prefix" not in haz.columns or "Hazard" not in haz.columns:
        raise ValueError("NRI_HazardInfo.csv must contain Prefix and Hazard columns.")

    prefix_to_hazard = dict(zip(haz["Prefix"].astype(str), haz["Hazard"].astype(str)))

    # Keep only useful hazard-level metrics (present as PREFIX_SUFFIX)
    suffix_priority = [
        "EALS", "EALR", "EALB", "EALP", "EALA",
        "ALRB", "ALRP", "ALRA", "ALR_NPCTL",
        "RISKV", "RISKS", "RISKR",
        "EAL_NPCTL", "RISKS_NPCTL", "RISKV_NPCTL"
    ]

    hazard_rows = []
    for prefix, hazard_name in prefix_to_hazard.items():
        prefix = str(prefix).strip()
        if not prefix:
            continue

        # find all columns with this prefix_
        prefixed_cols = [c for c in nri.columns if isinstance(c, str) and c.startswith(prefix + "_")]
        if not prefixed_cols:
            continue

        # choose a stable subset of columns by suffix list, falling back to whatever exists
        chosen = []
        for suf in suffix_priority:
            col = f"{prefix}_{suf}"
            if col in nri.columns:
                chosen.append(col)

        # If none matched our known suffixes, keep a small slice rather than exploding the file
        if not chosen:
            chosen = prefixed_cols[:8]

        tmp = nri[["GEOID", "STATEABBRV", "COUNTY", "NRI_VER"] + chosen].copy()
        tmp.insert(1, "Prefix", prefix)
        tmp.insert(2, "Hazard", hazard_name)

        # strip prefix_ from hazard columns
        rename_map = {c: c.replace(prefix + "_", "") for c in chosen}
        tmp = tmp.rename(columns=rename_map)

        hazard_rows.append(tmp)

    if hazard_rows:
        nri_hazard_long = pd.concat(hazard_rows, ignore_index=True)

        # sanity: one row per GEOID×Prefix ideally
        # (duplicates can exist in rare cases; if so, we keep them but you can dedupe later)
        out_haz = out_dir / "nri_county_hazard_long.csv"
        nri_hazard_long.to_csv(out_haz, index=False)
        print(f"[OK] Wrote {out_haz}  rows={len(nri_hazard_long):,} cols={nri_hazard_long.shape[1]}")
    else:
        out_haz = out_dir / "nri_county_hazard_long.csv"
        pd.DataFrame().to_csv(out_haz, index=False)
        print(f"[WARN] No hazard-prefixed columns found; wrote empty {out_haz}")

    return {"total": out_total, "hazard_long": out_haz}


def process_dictionary(raw_dir: Path, out_dir: Path) -> Optional[Path]:
    dd_path = raw_dir / "NRIDataDictionary.csv"
    if not dd_path.exists():
        return None
    dd = pd.read_csv(dd_path, low_memory=False)
    out = out_dir / "nri_data_dictionary_preview.csv"
    dd.to_csv(out, index=False)
    print(f"[OK] Wrote {out}  rows={len(dd):,} cols={dd.shape[1]}")
    return out


# ---------------------------
# Entrypoint
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default=str(Path(__file__).resolve().parent),
                        help="Directory containing the raw input CSVs.")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent / "data" / "processed"),
                        help="Directory to write processed outputs.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"RAW_DIR: {raw_dir}")
    print(f"OUT_DIR: {out_dir}")

    outputs = {}

    # Metro
    outputs["metro_snapshot"] = process_metro(raw_dir, out_dir)

    # County/ZIP mapping
    outputs["county_zip"] = process_county_zip(raw_dir, out_dir)

    # NRI tables
    nri_out = process_nri(raw_dir, out_dir)
    outputs.update({f"nri_{k}": v for k, v in nri_out.items()})

    # Dictionary (optional)
    dd_out = process_dictionary(raw_dir, out_dir)
    if dd_out:
        outputs["nri_dictionary"] = dd_out

    # Write a small manifest for your backend
    manifest = out_dir / "preprocess_manifest.txt"
    with open(manifest, "w", encoding="utf-8") as f:
        for k, v in outputs.items():
            f.write(f"{k}: {v}\n")

    print(f"[OK] Wrote {manifest}")
    print("\nDone.")


if __name__ == "__main__":
    main()