const API_BASE = "https://unpetulant-lenora-compensatory.ngrok-free.dev";

const NGROK_HEADERS: HeadersInit = {
  "Content-Type": "application/json",
  "ngrok-skip-browser-warning": "true",
};

export interface Weights {
  CFLD: number;
  ERQK: number;
  HRCN: number;
  TRND: number;
  WFIR: number;
  CWAV: number;
  HWAV: number;
  DRGT: number;
}

export interface FilterRequest {
  weights: Weights;
  threshold: number;
  bbox: [number, number, number, number] | null;
  normalize_weights: boolean;
  include_debug_fields: boolean;
  max_features: number;
}

// Matches the actual PreviewResponse from app.py
export interface PreviewResponse {
  threshold: number;
  bbox_used: number[] | null;
  weight_sum_in: number;
  weight_sum_used: number;
  counties_total: number;
  counties_in_view: number;
  counties_passing: number;
  risk_min: number | null;
  risk_median: number | null;
  risk_max: number | null;
}

export const defaultWeights: Weights = {
  CFLD: 0.5,
  ERQK: 0.5,
  HRCN: 0.5,
  TRND: 0.5,
  WFIR: 0.5,
  CWAV: 0.5,
  HWAV: 0.5,
  DRGT: 0.5,
};

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { headers: NGROK_HEADERS });
    return res.ok;
  } catch {
    return false;
  }
}

export async function fetchFilteredMap(
  weights: Weights,
  bbox: [number, number, number, number] | null = null
): Promise<GeoJSON.FeatureCollection> {
  const body: FilterRequest = {
    weights,
    threshold: 100,
    bbox,
    normalize_weights: true,
    include_debug_fields: true,
    max_features: 4000,
  };

  const res = await fetch(`${API_BASE}/map/filter`, {
    method: "POST",
    headers: NGROK_HEADERS,
    body: JSON.stringify(body),
  });

  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchPreview(
  weights: Weights,
  bbox: [number, number, number, number] | null = null
): Promise<PreviewResponse> {
  const body: FilterRequest = {
    weights,
    threshold: 100,
    bbox,
    normalize_weights: true,
    include_debug_fields: false,
    max_features: 4000,
  };

  const res = await fetch(`${API_BASE}/score/preview`, {
    method: "POST",
    headers: NGROK_HEADERS,
    body: JSON.stringify(body),
  });

  if (!res.ok) throw new Error(`Preview API error: ${res.status}`);
  return res.json();
}
