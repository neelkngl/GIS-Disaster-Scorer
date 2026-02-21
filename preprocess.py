"""
preprocess.py

Prepares processed datasets from the uploaded CSVs:
 - Metro ZHVI (wide -> long; compute latest / 5y % / 10y slope)
 - Metro Market Temp (same)
 - Join metro metrics into metro_snapshot_zhvi_markettemp.csv

 - NRI:
    - If NRI_Table_Counties.csv exists, use it (no .gdb required).
    - Merge with NRI_HazardInfo.csv and NRIDataDictionary.csv when present.
    - Output nri_counties_processed.csv

 - COUNTY_ZIP_CONVERSION.csv (copied to processed with small cleaning)

Outputs written to ./data/processed/
"""

import os
import sys
import re
import math
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR  # assumes running where the CSVs reside
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Input filenames (expected)
INPUT_FILES = {
    "zhvi": "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "market_temp": "Metro_market_temp_index_uc_sfrcondo_month.csv",
    "nri_table": "NRI_Table_Counties.csv",
    "nri_hazard": "NRI_HazardInfo.csv",
    "nri_dict": "NRIDataDictionary.csv",
    "county_zip": "COUNTY_ZIP_CONVERSION.csv",
}

def file_path(key):
    return RAW_DIR / INPUT_FILES[key]

def exists(key):
    return file_path(key).exists()

def detect_date_columns(cols):
    """Return list of column names that look like YYYY-MM or YYYY-MM-DD"""
    date_cols = []
    for c in cols:
        if not isinstance(c, str):
            continue
        if re.match(r"^\d{4}-\d{2}(-\d{2})?$", c):
            date_cols.append(c)
    return date_cols

def melt_wide_to_long(df, id_cols):
    date_cols = detect_date_columns(df.columns)
    if not date_cols:
        raise ValueError("No date-like columns detected in dataframe")
    long = df.melt(id_vars=id_cols, value_vars=date_cols, var_name="date", value_name="value")
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long = long.dropna(subset=["value"]).sort_values(id_cols + ["date"])
    # coerce numeric values
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    return long

def compute_trend_metrics(long_df, group_cols, value_col="value", date_col="date"):
    """Compute latest value, 5-year pct change, 10-year slope per year for each group."""
    out_rows = []
    grouped = long_df.groupby(group_cols, sort=False)
    for key, g in grouped:
        g = g.sort_values(date_col).dropna(subset=[value_col])
        if g.empty:
            continue
        latest_row = g.iloc[-1]
        latest_date = pd.to_datetime(latest_row[date_col])
        latest_val = float(latest_row[value_col])

        # 5-year percent change (closest earlier value <= latest_date - 5 years)
        five_years_ago = latest_date - pd.DateOffset(years=5)
        g5 = g[g[date_col] <= five_years_ago]
        if not g5.empty:
            base_val = float(g5.iloc[-1][value_col])
            pct_5y = (latest_val / base_val - 1.0) * 100.0 if base_val != 0 else None
        else:
            pct_5y = None

        # 10-year slope (per year) using data from last 10 years (if available)
        ten_years_ago = latest_date - pd.DateOffset(years=10)
        g10 = g[g[date_col] >= ten_years_ago]
        slope_per_year = None
        if len(g10) >= 24:  # need at least ~2 years of monthly data to estimate slope
            x = (g10[date_col] - g10[date_col].iloc[0]).dt.days / 365.25
            y = g10[value_col].astype(float)
            x_mean, y_mean = x.mean(), y.mean()
            denom = ((x - x_mean) ** 2).sum()
            if denom != 0:
                slope = (((x - x_mean) * (y - y_mean)).sum()) / denom
                slope_per_year = float(slope)

        # Build output row
        row = {
            "latest_date": latest_date,
            "latest_value": latest_val,
            "pct_change_5y": pct_5y,
            "slope_per_year_10y": slope_per_year,
        }

        # if key is tuple of group columns, expand
        if isinstance(key, tuple):
            for c, v in zip(group_cols, key):
                row[c] = v
        else:
            row[group_cols[0]] = key

        out_rows.append(row)

    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows)

def process_metro_files():
    """Read ZHVI and Market Temp, compute metrics and produce a joined snapshot."""
    if not exists("zhvi") and not exists("market_temp"):
        print("No metro Zillow files found. Skipping metro processing.")
        return None

    zhvi_df = None
    temp_df = None

    if exists("zhvi"):
        print("Loading ZHVI:", file_path("zhvi"))
        zhvi_df = pd.read_csv(file_path("zhvi"), low_memory=False)
        try:
            date_cols = detect_date_columns(zhvi_df.columns)
            id_cols = [c for c in zhvi_df.columns if c not in date_cols]
            zhvi_long = melt_wide_to_long(zhvi_df, id_cols)
            # Choose grouping key heuristically
            group_cols = ["RegionID"] if "RegionID" in zhvi_long.columns else (["RegionName"] if "RegionName" in zhvi_long.columns else [id_cols[0]])
            zhvi_metrics = compute_trend_metrics(zhvi_long, group_cols=group_cols)
            # rename columns to make intent explicit
            zhvi_metrics = zhvi_metrics.rename(columns={
                "latest_date": "zhvi_latest_date",
                "latest_value": "zhvi_latest_value",
                "pct_change_5y": "zhvi_pct_change_5y",
                "slope_per_year_10y": "zhvi_slope_per_year_10y",
            })
        except Exception as e:
            print("Failed processing ZHVI:", e)
            zhvi_metrics = pd.DataFrame()
    else:
        zhvi_metrics = pd.DataFrame()

    if exists("market_temp"):
        print("Loading market temp:", file_path("market_temp"))
        temp_df = pd.read_csv(file_path("market_temp"), low_memory=False)
        try:
            date_cols_t = detect_date_columns(temp_df.columns)
            id_cols_t = [c for c in temp_df.columns if c not in date_cols_t]
            temp_long = melt_wide_to_long(temp_df, id_cols=id_cols_t)
            group_cols_t = ["RegionID"] if "RegionID" in temp_long.columns else (["RegionName"] if "RegionName" in temp_long.columns else [id_cols_t[0]])
            temp_metrics = compute_trend_metrics(temp_long, group_cols=group_cols_t)
            temp_metrics = temp_metrics.rename(columns={
                "latest_date": "temp_latest_date",
                "latest_value": "market_temp_latest",
                "pct_change_5y": "market_temp_pct_change_5y",
                "slope_per_year_10y": "market_temp_slope_per_year_10y",
            })
        except Exception as e:
            print("Failed processing Market Temp:", e)
            temp_metrics = pd.DataFrame()
    else:
        temp_metrics = pd.DataFrame()

    # Join the two metric tables
    if not zhvi_metrics.empty and not temp_metrics.empty:
        # prefer join on RegionID when possible
        if "RegionID" in zhvi_metrics.columns and "RegionID" in temp_metrics.columns:
            snapshot = zhvi_metrics.merge(temp_metrics, on="RegionID", how="outer")
            # bring in a few meta columns from original ZHVI if present
            meta_cols = [c for c in ["RegionID", "RegionName", "StateName", "State", "Metro", "SizeRank"] if c in zhvi_df.columns]
            if meta_cols:
                meta = zhvi_df[meta_cols].drop_duplicates(subset=["RegionID"])
                snapshot = meta.merge(snapshot, on="RegionID", how="right")
        else:
            key = list(zhvi_metrics.columns.intersection(temp_metrics.columns))
            # fallback: merge on intersection or on first group column
            if key:
                snapshot = zhvi_metrics.merge(temp_metrics, on=key[0], how="outer")
            else:
                left_key = [c for c in zhvi_metrics.columns if "Region" in c or "region" in c][:1]
                right_key = [c for c in temp_metrics.columns if "Region" in c or "region" in c][:1]
                if left_key and right_key and left_key[0] == right_key[0]:
                    snapshot = zhvi_metrics.merge(temp_metrics, on=left_key[0], how="outer")
                else:
                    # no good join key -> concatenate side-by-side with suffixes
                    snapshot = zhvi_metrics.merge(temp_metrics, left_index=True, right_index=True, how="outer", suffixes=("_zhvi", "_temp"))
    elif not zhvi_metrics.empty:
        snapshot = zhvi_metrics
    elif not temp_metrics.empty:
        snapshot = temp_metrics
    else:
        snapshot = pd.DataFrame()

    if snapshot.empty:
        print("No metro snapshot generated (no metric tables).")
        return None

    out_path = PROCESSED_DIR / "metro_snapshot_zhvi_markettemp.csv"
    snapshot.to_csv(out_path, index=False)
    print("Wrote metro snapshot:", out_path)
    return out_path

def process_nri_tables():
    """Consolidate NRI CSV tables into a processed county-level CSV."""
    if not exists("nri_table"):
        print("No NRI_Table_Counties.csv found. Skipping NRI processing.")
        return None

    print("Loading NRI table:", file_path("nri_table"))
    nri = pd.read_csv(file_path("nri_table"), low_memory=False)

    # Normalize column names (strip whitespace)
    nri.columns = [c.strip() if isinstance(c, str) else c for c in nri.columns]

    # If hazard info CSV exists, join additional metadata
    if exists("nri_hazard"):
        print("Loading NRI hazard info:", file_path("nri_hazard"))
        hazard = pd.read_csv(file_path("nri_hazard"), low_memory=False)
        # Attempt to merge on a common column if present (e.g., HAZARD_ID, HazardKey)
        common = set(nri.columns).intersection(set(hazard.columns))
        # if no natural join, try to join on a 'Hazard' or 'HAZARD' column
        if common:
            # choose a sensible small join (if shared)
            join_col = list(common)[0]
            try:
                nri = nri.merge(hazard, on=join_col, how="left", suffixes=("", "_haz"))
            except Exception:
                # fallback: concatenate columns (no join)
                pass
        else:
            # no joinable column: append hazard as extra table (skip merge)
            pass

    # If data dictionary is present, optionally annotate columns
    if exists("nri_dict"):
        try:
            print("Loading NRI data dictionary:", file_path("nri_dict"))
            dd = pd.read_csv(file_path("nri_dict"), low_memory=False)
            # If the dictionary maps column -> description, keep a small file for reference
            dd_path = PROCESSED_DIR / "nri_data_dictionary_preview.csv"
            dd.to_csv(dd_path, index=False)
            print("Wrote NRI data dictionary preview:", dd_path)
        except Exception as e:
            print("Failed reading NRI data dictionary:", e)

    # Keep a trimmed set of high-value columns if present
    preferred = []
    pref_candidates = ["GEOID", "FIPS", "STATE", "STATEABBRV", "COUNTY", "COUNTYNAME", "NAME",
                       "RISK_INDEX", "RISK_SCORE", "EAL_VALT", "EAL_VALB", "SOVI", "RESL"]
    for c in pref_candidates:
        if c in nri.columns:
            preferred.append(c)

    # Always keep all columns if preferred is empty
    if not preferred:
        preferred = list(nri.columns)

    # Create processed output (we avoid geometry here because input is CSV-only)
    nri_proc = nri[preferred].copy()
    # Standardize GEOID/FIPS to string with zero padding if numeric
    if "GEOID" in nri_proc.columns:
        nri_proc["GEOID"] = nri_proc["GEOID"].astype(str).str.zfill(5)
    if "FIPS" in nri_proc.columns:
        nri_proc["FIPS"] = nri_proc["FIPS"].astype(str)

    out_path = PROCESSED_DIR / "nri_counties_processed.csv"
    nri_proc.to_csv(out_path, index=False)
    print("Wrote NRI processed CSV:", out_path)
    return out_path

def process_county_zip():
    if not exists("county_zip"):
        print("No COUNTY_ZIP_CONVERSION.csv found. Skipping.")
        return None
    print("Loading county-zip mapping:", file_path("county_zip"))
    cz = pd.read_csv(file_path("county_zip"), low_memory=False)
    # Normalize column names and try to ensure GEOID/FIPS and ZIP fields exist
    cz.columns = [c.strip() for c in cz.columns]
    # Try to detect FIPS/GEOID and ZIP columns
    gcol = None
    zcol = None
    for c in cz.columns:
        if re.search(r"geo", c, re.I) or re.search(r"fips", c, re.I):
            gcol = c
        if re.search(r"zip", c, re.I):
            zcol = c
    # If no obvious columns, just save as-is
    out_path = PROCESSED_DIR / "county_zip_conversion_processed.csv"
    cz.to_csv(out_path, index=False)
    print("Wrote county-zip conversion processed CSV:", out_path)
    return out_path

def main():
    print("Starting preprocess.py")
    metro_out = process_metro_files()
    nri_out = process_nri_tables()
    cz_out = process_county_zip()

    print("\nSummary of outputs written (if any):")
    for p in [metro_out, nri_out, cz_out]:
        if p:
            print(" -", p)
    print("\nAll done.")

if __name__ == "__main__":
    main()

