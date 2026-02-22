import { useState, useEffect, useCallback, useRef } from "react";
import { type Weights, defaultWeights, scoreFeatures, VALID_STATE_FIPS } from "@/lib/scoring";

export type { Weights };
export { defaultWeights };

export function useMapFilter() {
  const [weights, setWeights] = useState<Weights>(defaultWeights);
  const [rawData, setRawData] = useState<GeoJSON.FeatureCollection | null>(null);
  const [geoData, setGeoData] = useState<GeoJSON.FeatureCollection | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const weightsRef = useRef<Weights>(defaultWeights);

  // Load GeoJSON once on mount
  useEffect(() => {
    setLoading(true);
    fetch("/data/counties.geojson")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load county data");
        return res.json();
      })
      .then((raw: any) => {
        // Strip non-standard fields and filter to 50 US states
        const filtered = raw.features.filter((f: any) => {
          const p = f.properties ?? {};
          const fips = String(p.STATEFP ?? String(p.GEOID ?? "").slice(0, 2)).padStart(2, "0");
          return VALID_STATE_FIPS.has(fips);
        });
        const fc: GeoJSON.FeatureCollection = {
          type: "FeatureCollection",
          features: filtered,
        };
        setRawData(fc);
        // Score with default weights
        const scored = scoreFeatures(fc, defaultWeights);
        setGeoData(scored);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const updateWeight = useCallback((key: keyof Weights, value: number) => {
    setWeights((prev) => {
      const next = { ...prev, [key]: value };
      weightsRef.current = next;
      return next;
    });
  }, []);

  // Re-score on submit
  const submit = useCallback(() => {
    if (!rawData) return;
    setLoading(true);
    // Use requestAnimationFrame to keep UI responsive
    requestAnimationFrame(() => {
      const scored = scoreFeatures(rawData, weightsRef.current);
      setGeoData(scored);
      setLoading(false);
    });
  }, [rawData]);

  return {
    weights,
    updateWeight,
    submit,
    geoData,
    loading,
    error,
  };
}
