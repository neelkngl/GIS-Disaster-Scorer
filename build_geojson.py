import zipfile
import requests
from pathlib import Path

import pandas as pd

try:
    import geopandas as gpd
except Exception as e:
    gpd = None
    _GEOPANDAS_IMPORT_ERROR = e

from project_paths import processed_dir as proj_processed_dir, out_dir as proj_out_dir, raw_dir as proj_raw_dir


# -------------------------
# Config
# -------------------------
PROCESSED_DIR = Path(proj_processed_dir()).resolve()
RAW_DIR = Path(proj_raw_dir()).resolve()
OUT_DIR = Path(proj_out_dir()).resolve()

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# GeoJSON output folder where app.py expects it
GEOJSON_OUT_DIR = PROCESSED_DIR / "geojson"
GEOJSON_OUT_DIR.mkdir(parents=True, exist_ok=True)

# NEW: join scored county table (produced by scoring.py)
COUNTY_SCORES_CSV = PROCESSED_DIR / "county_scores_balanced.csv"

# Optional: keep the NRI totals join too (set to True if you want both)
JOIN_NRI_TOTALS = False
NRI_TOTAL_CSV = PROCESSED_DIR / "nri_county_total.csv"

# Census county boundaries (cartographic boundary, good for thematic maps)
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
    # ✅ Only skip if we see a shapefile already (prevents “KML extracted” stale state)
    if out_dir.exists() and any(out_dir.rglob("*.shp")):
        print(f"[OK] Already extracted (shp found): {out_dir}")
        return

    # Clean dir if it exists but doesn't contain shp
    if out_dir.exists():
        for p in sorted(out_dir.rglob("*"), reverse=True):
            try:
                if p.is_file():
                    p.unlink()
                else:
                    p.rmdir()
            except OSError:
                pass

    print(f"[UNZIP] Extracting {zip_path.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    print(f"[OK] Extracted -> {out_dir}")


def find_shapefile(folder: Path) -> Path:
    shp = list(folder.rglob("*.shp"))
    if not shp:
        raise FileNotFoundError("Could not find .shp in extracted folder")
    return shp[0]


def main():
    if gpd is None:
        raise ModuleNotFoundError(
            f"geopandas is required for build_geojson.py. Import error was: {_GEOPANDAS_IMPORT_ERROR}"
        )

    if not COUNTY_SCORES_CSV.exists():
        raise FileNotFoundError(
            f"Missing {COUNTY_SCORES_CSV}. Run scoring.py first to produce county_scores_balanced.csv."
        )

    # 1) Get county boundaries (geometry)
    download_if_needed(COUNTY_ZIP_URL, COUNTY_ZIP_PATH)
    unzip_if_needed(COUNTY_ZIP_PATH, EXTRACT_DIR)

    shp_path = find_shapefile(EXTRACT_DIR)
    print(f"[READ] County boundaries: {shp_path}")
    counties = gpd.read_file(shp_path)

    # 2) Ensure GEOID exists
    if "GEOID" in counties.columns:
        counties["GEOID"] = counties["GEOID"].astype(str).str.zfill(5)
    elif "GEOIDFP" in counties.columns:
        counties["GEOID"] = counties["GEOIDFP"].astype(str).str.zfill(5)
    else:
        raise KeyError(f"No GEOID/GEOIDFP found in county boundary columns: {list(counties.columns)}")

    # 3) Standardize CRS to WGS84
    if counties.crs is None:
        counties = counties.set_crs(epsg=4326)
    else:
        counties = counties.to_crs(epsg=4326)

    # 4) Load county scores (your model output)
    scores = pd.read_csv(COUNTY_SCORES_CSV, dtype={"GEOID": str})
    scores["GEOID"] = scores["GEOID"].astype(str).str.zfill(5)

    # 5) Join geometry + scores
    joined = counties.merge(scores, on="GEOID", how="left")

    # Optional: also join NRI totals for reference fields (RISK_SCORE/EAL_VALT/etc)
    if JOIN_NRI_TOTALS:
        if not NRI_TOTAL_CSV.exists():
            raise FileNotFoundError(f"JOIN_NRI_TOTALS=True but missing {NRI_TOTAL_CSV}")
        nri = pd.read_csv(NRI_TOTAL_CSV, dtype={"GEOID": str})
        nri["GEOID"] = nri["GEOID"].astype(str).str.zfill(5)

        # Avoid duplicate column names: suffix NRI columns
        joined = joined.merge(nri, on="GEOID", how="left", suffixes=("", "_nri"))

    # 6) Sanity checks
    missing_score = joined["score_0_100"].isna().sum() if "score_0_100" in joined.columns else None
    if missing_score is not None:
        print(f"[JOIN] rows={len(joined):,} missing_score_0_100={missing_score:,}")
    else:
        print(f"[JOIN] rows={len(joined):,} (score_0_100 not found in joined properties)")

    # 7) Write GeoJSONs
    out_full = GEOJSON_OUT_DIR / "county_scores_balanced.geojson"
    joined.to_file(out_full, driver="GeoJSON")
    print(f"[OK] Wrote {out_full}")

    joined_simpl = joined.copy()
    joined_simpl["geometry"] = joined_simpl["geometry"].simplify(tolerance=0.01, preserve_topology=True)
    out_simpl = GEOJSON_OUT_DIR / "county_scores_balanced_simplified.geojson"
    joined_simpl.to_file(out_simpl, driver="GeoJSON")
    print(f"[OK] Wrote {out_simpl}")


if __name__ == "__main__":
    main()