import os
import zipfile
import requests
from pathlib import Path

import pandas as pd
import geopandas as gpd


# -------------------------
# Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUT_DIR = DATA_DIR / "out"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Your processed NRI totals (one row per county, GEOID = 5-digit county FIPS)
NRI_TOTAL_CSV = PROCESSED_DIR / "nri_county_total.csv"

# Census county boundaries (cartographic boundary, good for thematic maps)
# (Official download URL shown in catalog listing.) :contentReference[oaicite:2]{index=2}
COUNTY_ZIP_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
COUNTY_ZIP_PATH = RAW_DIR / "cb_2023_us_county_500k.zip"

EXTRACT_DIR = RAW_DIR / "cb_2023_us_county_500k"


def download_if_needed(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[OK] Already downloaded: {dest.name}")
        return
    print(f"[DL] Downloading {url}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print(f"[OK] Downloaded -> {dest}")


def unzip_if_needed(zip_path: Path, out_dir: Path):
    if out_dir.exists() and any(out_dir.rglob("*")):
        print(f"[OK] Already extracted: {out_dir}")
        return
    print(f"[UNZIP] Extracting {zip_path.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    print(f"[OK] Extracted -> {out_dir}")


def find_vector_file(folder: Path):
    shp = list(folder.rglob("*.shp"))
    if shp:
        return shp[0]
    raise FileNotFoundError("Could not find .shp in extracted folder")

def main():
    if not NRI_TOTAL_CSV.exists():
        raise FileNotFoundError(f"Missing {NRI_TOTAL_CSV}. Run preprocess.py first.")

    # 1) Get county boundaries (geometry)
    download_if_needed(COUNTY_ZIP_URL, COUNTY_ZIP_PATH)
    unzip_if_needed(COUNTY_ZIP_PATH, EXTRACT_DIR)

    vec_path = find_vector_file(EXTRACT_DIR)
    print(f"[READ] County boundaries: {vec_path}")

    # KML requires fiona + drivers; if KML read fails, switch to SHP download instead.
    counties = gpd.read_file(vec_path)

    # 2) Ensure GEOID field exists and is 5-digit county FIPS
    # Census county cartographic boundary often has a GEOID column.
    if "GEOID" in counties.columns:
        counties["GEOID"] = counties["GEOID"].astype(str).str.zfill(5)
    elif "GEOIDFP" in counties.columns:
        counties["GEOID"] = counties["GEOIDFP"].astype(str).str.zfill(5)
    else:
        raise KeyError(f"Could not find GEOID/GEOIDFP in county boundary file columns: {list(counties.columns)}")

    # Standardize CRS to WGS84
    if counties.crs is None:
        counties = counties.set_crs(epsg=4326)
    else:
        counties = counties.to_crs(epsg=4326)

    # 3) Load your NRI totals
    nri = pd.read_csv(NRI_TOTAL_CSV, dtype={"GEOID": str})
    nri["GEOID"] = nri["GEOID"].astype(str).str.zfill(5)

    # 4) Join (geometry + attributes)
    joined = counties.merge(nri, on="GEOID", how="left")

    # 5) Quick sanity checks
    missing = joined["RISK_SCORE"].isna().sum() if "RISK_SCORE" in joined.columns else None
    print(f"[JOIN] rows={len(joined):,}  missing_RISK_SCORE={missing:,}" if missing is not None else f"[JOIN] rows={len(joined):,}")

    # 6) Write full GeoJSON
    out_full = OUT_DIR / "nri_counties.geojson"
    joined.to_file(out_full, driver="GeoJSON")
    print(f"[OK] Wrote {out_full}")

    # 7) Optional: simplify for web (much smaller)
    # Tolerance is in degrees; tune for your use case.
    joined_simpl = joined.copy()
    joined_simpl["geometry"] = joined_simpl["geometry"].simplify(tolerance=0.01, preserve_topology=True)
    out_simpl = OUT_DIR / "nri_counties_simplified.geojson"
    joined_simpl.to_file(out_simpl, driver="GeoJSON")
    print(f"[OK] Wrote {out_simpl}")


if __name__ == "__main__":
    main()