from __future__ import annotations

import os
from typing import Dict, Optional, List, Any, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import box

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conlist

from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# Config / Paths
# ============================================================
try:
    import project_paths  # uses your repo path helpers

    GEOJSON_PATH = os.getenv(
        "COUNTY_GEOJSON_PATH",
        str(project_paths.processed_dir() / "geojson" / "county_scores_balanced_simplified.geojson"),
    )

except Exception:
    GEOJSON_PATH = os.getenv(
        "COUNTY_GEOJSON_PATH",
        os.path.join("datasets", "processed", "geojson", "county_scores_balanced_simplified.geojson"),
    )

# Scoring weights (match scoring.py defaults)
RISK_W = float(os.getenv("RISK_W", "0.65"))
COST_W = float(os.getenv("COST_W", "0.35"))
HAZARD_W = float(os.getenv("HAZARD_W", "1.25"))

# The 8 hazard slider codes your scoring pipeline uses
HAZARDS = ["CFLD", "ERQK", "HRCN", "TRND", "WFIR", "CWAV", "HWAV", "DRGT"]
HAZARD_LABELS = {
    "CFLD": "Coastal Flooding",
    "ERQK": "Earthquake",
    "HRCN": "Hurricane",
    "TRND": "Tornado",
    "WFIR": "Wildfire",
    "CWAV": "Cold Wave",
    "HWAV": "Heat Wave",
    "DRGT": "Drought",
}


# ============================================================
# API Models
# ============================================================
class HazardWeights(BaseModel):
    CFLD: float = Field(0.0, ge=0.0)
    ERQK: float = Field(0.0, ge=0.0)
    HRCN: float = Field(0.0, ge=0.0)
    TRND: float = Field(0.0, ge=0.0)
    WFIR: float = Field(0.0, ge=0.0)
    CWAV: float = Field(0.0, ge=0.0)
    HWAV: float = Field(0.0, ge=0.0)
    DRGT: float = Field(0.0, ge=0.0)


class FilterRequest(BaseModel):
    weights: HazardWeights
    threshold: float = Field(
        50.0,
        ge=0.0,
        le=100.0,
        description="Show counties with computed risk score <= threshold (0-100; lower is safer).",
    )
    bbox: Optional[conlist(float, min_length=4, max_length=4)] = Field(
        None, description="Optional [minLon, minLat, maxLon, maxLat] to limit results."
    )
    normalize_weights: bool = Field(
        True, description="Normalize weights to sum=1 (unless all zeros)."
    )
    include_debug_fields: bool = Field(
        True, description="If true, include computed fields in GeoJSON properties."
    )
    max_features: int = Field(4000, ge=1, le=20000)


class PreviewResponse(BaseModel):
    threshold: float
    bbox_used: Optional[List[float]]
    weight_sum_in: float
    weight_sum_used: float
    counties_total: int
    counties_in_view: int
    counties_passing: int
    risk_min: Optional[float]
    risk_median: Optional[float]
    risk_max: Optional[float]


# ============================================================
# App + Globals
# ============================================================
app = FastAPI(title="SafeHaven - County Map", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GDF: Optional[gpd.GeoDataFrame] = None
SINDEX = None

# cached numpy columns for fast scoring (aligned to GDF index order)
RISK_NORM: Optional[np.ndarray] = None
COST_NORM: Optional[np.ndarray] = None
HAZ_NORM: Optional[np.ndarray] = None  # shape (N, 8)


# ============================================================
# Helpers
# ============================================================
def _require_loaded():
    if GDF is None or SINDEX is None or RISK_NORM is None or COST_NORM is None or HAZ_NORM is None:
        raise HTTPException(status_code=500, detail="Server data not loaded.")


def _pct_rank_0_100(x: np.ndarray) -> np.ndarray:
    """
    Percentile rank scaled to 0..100 (stable, robust-ish).
    Similar idea to scoring.py's use of rank(pct=True). :contentReference[oaicite:4]{index=4}
    """
    x = np.asarray(x, dtype=float)
    # handle NaNs by pushing them to +inf so they end up worst; then replace later
    nan_mask = ~np.isfinite(x)
    x2 = x.copy()
    x2[nan_mask] = np.nanmax(x2[~nan_mask]) if (~nan_mask).any() else 0.0

    order = np.argsort(x2, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x2) + 1, dtype=float)
    pct = ranks / float(len(x2))
    out = 100.0 * pct
    out[nan_mask] = 100.0
    return out


def _normalize_weights(w: Dict[str, float], normalize: bool) -> Tuple[np.ndarray, float, float]:
    v = np.array([float(max(0.0, w.get(hz, 0.0))) for hz in HAZARDS], dtype=float)
    s_in = float(v.sum())
    if not normalize:
        return v, s_in, s_in
    if s_in <= 0.0:
        return v, s_in, s_in
    v = v / s_in
    return v, s_in, float(v.sum())


def _bbox_filter(gdf: gpd.GeoDataFrame, bbox_vals: List[float]) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bbox_vals
    bbox_geom = box(minx, miny, maxx, maxy)

    idx = list(SINDEX.intersection(bbox_geom.bounds))
    if not idx:
        return gdf.iloc[0:0]
    cand = gdf.iloc[idx]
    return cand[cand.intersects(bbox_geom)]


def _to_featurecollection(gdf: gpd.GeoDataFrame, max_features: int) -> Dict[str, Any]:
    if len(gdf) > max_features:
        raise HTTPException(
            status_code=413,
            detail=f"Too many features returned ({len(gdf)}). Use bbox or raise max_features.",
        )
    # ensure WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    return gdf.__geo_interface__


# ============================================================
# Startup: load geojson + cache numeric arrays
# ============================================================
@app.on_event("startup")
def startup_load():
    global GDF, SINDEX, RISK_NORM, COST_NORM, HAZ_NORM

    if not os.path.exists(GEOJSON_PATH):
        raise RuntimeError(f"Missing county GeoJSON: {GEOJSON_PATH}")

    gdf = gpd.read_file(GEOJSON_PATH)

    valid_state_fips = {
        "01","02","04","05","06","08","09","10","11","12","13","15","16","17","18","19",
        "20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35",
        "36","37","38","39","40","41","42","44","45","46","47","48","49","50","51","53",
        "54","55","56"
    }

    if "STATEFP" in gdf.columns:
        gdf["STATEFP"] = gdf["STATEFP"].astype(str).str.zfill(2)
    else:
        gdf["STATEFP"] = gdf["GEOID"].astype(str).str[:2]

    gdf = gdf[gdf["STATEFP"].isin(valid_state_fips)].copy()
    gdf = gdf.reset_index(drop=True)

    # required columns
    needed = ["GEOID", "risk_norm", "cost_norm"] + [f"{hz}_norm" for hz in HAZARDS]
    missing = [c for c in needed if c not in gdf.columns]
    if missing:
        raise RuntimeError(
            f"GeoJSON missing required columns: {missing}\n"
            f"Available columns: {list(gdf.columns)}"
        )

    gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(5)

    # CRS
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    # spatial index
    SINDEX = gdf.sindex

    # cache arrays
    RISK_NORM = np.asarray(gdf["risk_norm"].fillna(0.0), dtype=float)
    COST_NORM = np.asarray(gdf["cost_norm"].fillna(0.0), dtype=float)
    HAZ_NORM = np.stack([np.asarray(gdf[f"{hz}_norm"].fillna(0.0), dtype=float) for hz in HAZARDS], axis=1)

    GDF = gdf


# ============================================================
# Endpoints
# ============================================================
@app.get("/health")
def health():
    ok = GDF is not None
    return {
        "ok": ok,
        "geojson_path": GEOJSON_PATH,
        "counties": None if GDF is None else int(len(GDF)),
        "hazards": [{"code": hz, "label": HAZARD_LABELS.get(hz, hz)} for hz in HAZARDS],
        "weights": {"risk_w": RISK_W, "cost_w": COST_W, "hazard_w": HAZARD_W},
    }


@app.get("/hazards")
def hazards():
    return {
        "hazards": [{"code": hz, "label": HAZARD_LABELS.get(hz, hz)} for hz in HAZARDS],
        "columns_used": {
            "base": ["risk_norm", "cost_norm"],
            "hazards": [f"{hz}_norm" for hz in HAZARDS],
        },
        "scoring": {
            "base_penalty": f"{RISK_W}*risk_norm + {COST_W}*cost_norm",
            "hazard_penalty": "weighted_avg(hz_norm) with weights normalized to sum=1",
            "risk_raw": f"base_penalty + {HAZARD_W}*hazard_penalty  (higher = worse)",
            "risk_0_100": "percentile_rank(risk_raw) * 100",
        },
    }


@app.post("/score/preview", response_model=PreviewResponse)
def score_preview(req: FilterRequest):
    _require_loaded()

    w_vec, s_in, s_used = _normalize_weights(req.weights.model_dump(), req.normalize_weights)

    # hazard penalty is weighted average of hz_norms :contentReference[oaicite:5]{index=5}
    if s_in <= 0.0:
        hazard_penalty = np.zeros_like(RISK_NORM)
    else:
        hazard_penalty = (HAZ_NORM @ w_vec).astype(float)

    # base penalty :contentReference[oaicite:6]{index=6}
    base_penalty = (RISK_W * RISK_NORM + COST_W * COST_NORM).astype(float)

    # risk_raw: higher=worse (so "under threshold" means safer)
    risk_raw = base_penalty + HAZARD_W * hazard_penalty

    # stable-ish 0-100 risk index (higher is safer)
    risk_percentile = _pct_rank_0_100(risk_raw)
    safe_0_100 = 100.0 - risk_percentile

    gdf = GDF  # type: ignore[assignment]

    if req.bbox is not None:
        gdf_view = _bbox_filter(gdf, list(req.bbox))
        in_view = int(len(gdf_view))
        if in_view == 0:
            return PreviewResponse(
                threshold=req.threshold,
                bbox_used=list(req.bbox),
                weight_sum_in=s_in,
                weight_sum_used=s_used,
                counties_total=int(len(gdf)),
                counties_in_view=0,
                counties_passing=0,
                risk_min=None,
                risk_median=None,
                risk_max=None,
            )
        idx = gdf_view.index.to_numpy()
        risk_view = safe_0_100[idx]
    else:
        in_view = int(len(gdf))
        risk_view = safe_0_100

    passing = risk_view >= float(req.threshold)
    pass_count = int(passing.sum())

    if pass_count > 0:
        rmin = float(np.min(risk_view[passing]))
        rmed = float(np.median(risk_view[passing]))
        rmax = float(np.max(risk_view[passing]))
    else:
        rmin = rmed = rmax = None

    return PreviewResponse(
        threshold=req.threshold,
        bbox_used=list(req.bbox) if req.bbox is not None else None,
        weight_sum_in=s_in,
        weight_sum_used=s_used,
        counties_total=int(len(gdf)),
        counties_in_view=in_view,
        counties_passing=pass_count,
        risk_min=rmin,
        risk_median=rmed,
        risk_max=rmax,
    )


@app.post("/map/filter")
def map_filter(req: FilterRequest):
    _require_loaded()

    w_vec, s_in, s_used = _normalize_weights(req.weights.model_dump(), req.normalize_weights)

    if s_in <= 0.0:
        hazard_penalty = np.zeros_like(RISK_NORM)
    else:
        hazard_penalty = (HAZ_NORM @ w_vec).astype(float)

    base_penalty = (RISK_W * RISK_NORM + COST_W * COST_NORM).astype(float)
    risk_raw = base_penalty + HAZARD_W * hazard_penalty
    risk_percentile = _pct_rank_0_100(risk_raw)
    safe_0_100 = 100.0 - risk_percentile

    gdf = GDF.copy()  # type: ignore[union-attr]

    # bbox first for performance
    if req.bbox is not None:
        gdf = _bbox_filter(gdf, list(req.bbox))

    if gdf.empty:
        return JSONResponse(content={"type": "FeatureCollection", "features": []})

    idx = gdf.index.to_numpy()
    risk_subset = safe_0_100[idx]
    mask = risk_subset >= float(req.threshold)

    gdf = gdf.loc[gdf.index[mask]]

    if gdf.empty:
        return JSONResponse(content={"type": "FeatureCollection", "features": []})

    if req.include_debug_fields:
        gdf = gdf.copy()
        gdf["risk_raw"] = risk_raw[gdf.index]

    # add both so frontend won't break
        gdf["risk_0_100"] = risk_percentile[gdf.index]      # higher = worse (old behavior)
        gdf["safe_0_100"] = safe_0_100[gdf.index]           # higher = safer (new behavior)

        gdf["base_penalty"] = base_penalty[gdf.index]
        gdf["hazard_penalty"] = hazard_penalty[gdf.index]
        gdf["weight_sum_in"] = s_in
        gdf["weight_sum_used"] = s_used

    return JSONResponse(content=_to_featurecollection(gdf, max_features=req.max_features))

    resp.headers["Cache-Control"] = "no-store"
    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)