/**
 * Client-side scoring engine matching app.py / scoring.py logic.
 *
 * Scoring formula (from app.py):
 *   base_penalty  = RISK_W * risk_norm + COST_W * cost_norm
 *   hazard_penalty = weighted_avg(hz_norm values, slider weights)
 *   risk_raw       = base_penalty + HAZARD_W * hazard_penalty
 *   risk_0_100     = percentile_rank(risk_raw) * 100
 *
 * Lower risk_0_100 = safer county.
 */

export const RISK_W = 0.65;
export const COST_W = 0.35;
export const HAZARD_W = 1.25;

export const HAZARDS = ["CFLD", "ERQK", "HRCN", "TRND", "WFIR", "CWAV", "HWAV", "DRGT"] as const;

export type HazardCode = (typeof HAZARDS)[number];
export type Weights = Record<HazardCode, number>;

export const defaultWeights: Weights = {
  CFLD: 50,
  ERQK: 50,
  HRCN: 50,
  TRND: 50,
  WFIR: 50,
  CWAV: 50,
  HWAV: 50,
  DRGT: 50,
};

/**
 * Percentile-rank an array of numbers → 0..100 (matches _pct_rank_0_100 in app.py).
 */
/** Valid 50-state FIPS codes (excludes territories). */
export const VALID_STATE_FIPS = new Set([
  "01","02","04","05","06","08","09","10","11","12","13","15","16","17","18","19",
  "20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35",
  "36","37","38","39","40","41","42","44","45","46","47","48","49","50","51","53",
  "54","55","56",
]);

function percentileRank(values: number[]): number[] {
  const n = values.length;
  if (n === 0) return [];

  const indexed = values.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);

  const ranks = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    ranks[indexed[k].i] = (k + 1) / n;
  }

  return Array.from(ranks).map((r) => r * 100);
}

/**
 * Score all features in a GeoJSON FeatureCollection and return a new
 * FeatureCollection with `risk_0_100` added to each feature's properties.
 */
export function scoreFeatures(
  fc: GeoJSON.FeatureCollection,
  weights: Weights
): GeoJSON.FeatureCollection {
  const features = fc.features;
  if (features.length === 0) return fc;

  // Normalize weights
  const wArr = HAZARDS.map((hz) => Math.max(0, weights[hz] ?? 0));
  const wSum = wArr.reduce((a, b) => a + b, 0);
  const wNorm = wSum > 0 ? wArr.map((w) => w / wSum) : wArr.map(() => 0);

  // Compute risk_raw for each feature
  const riskRaw = features.map((f) => {
    const p = f.properties ?? {};
    const riskNorm = Number(p.risk_norm ?? 0);
    const costNorm = Number(p.cost_norm ?? 0);

    const basePenalty = RISK_W * riskNorm + COST_W * costNorm;

    let hazardPenalty = 0;
    if (wSum > 0) {
      for (let i = 0; i < HAZARDS.length; i++) {
        const hzNorm = Number(p[`${HAZARDS[i]}_norm`] ?? 0);
        hazardPenalty += wNorm[i] * hzNorm;
      }
    }

    return basePenalty + HAZARD_W * hazardPenalty;
  });

  // Percentile-rank → 0..100 (higher = worse)
  const risk0100 = percentileRank(riskRaw);

  // Invert: safe_0_100 where higher = safer (matches new backend)
  const safe0100 = risk0100.map((r) => 100 - r);

  // Clone features with new properties
  const scoredFeatures = features.map((f, i) => ({
    ...f,
    properties: {
      ...f.properties,
      risk_0_100: risk0100[i],
      safe_0_100: safe0100[i],
      risk_raw: riskRaw[i],
      base_penalty: RISK_W * Number(f.properties?.risk_norm ?? 0) + COST_W * Number(f.properties?.cost_norm ?? 0),
      hazard_penalty: riskRaw[i] - (RISK_W * Number(f.properties?.risk_norm ?? 0) + COST_W * Number(f.properties?.cost_norm ?? 0)),
    },
  }));

  return {
    type: "FeatureCollection",
    features: scoredFeatures,
  };
}
