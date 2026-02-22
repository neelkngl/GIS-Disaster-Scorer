import { useEffect, useRef, useCallback } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

interface MapViewProps {
  geoData: GeoJSON.FeatureCollection | null;
  loading: boolean;
  onBoundsChange: (bbox: [number, number, number, number]) => void;
}

function safetyColor(score: number): string {
  // score is safe_0_100: higher = safer
  if (score >= 80) return "#22c55e";
  if (score >= 60) return "#84cc16";
  if (score >= 40) return "#eab308";
  if (score >= 20) return "#f97316";
  return "#ef4444";
}

export default function MapView({ geoData, loading, onBoundsChange }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layerRef = useRef<L.GeoJSON | null>(null);
  const onBoundsChangeRef = useRef(onBoundsChange);
  onBoundsChangeRef.current = onBoundsChange;

  const fireBounds = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    const b = map.getBounds();
    onBoundsChangeRef.current([
      b.getWest(), b.getSouth(), b.getEast(), b.getNorth(),
    ]);
  }, []);

  // Initialize map once
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      center: [39.8, -98.5],
      zoom: 4,
      zoomControl: true,
    });

    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
      subdomains: "abcd",
      maxZoom: 19,
    }).addTo(map);

    map.on("moveend", fireBounds);
    mapRef.current = map;
    fireBounds();

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [fireBounds]);

  // Update GeoJSON layer when data changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !geoData) return;

    // Remove old layer
    if (layerRef.current) {
      map.removeLayer(layerRef.current);
    }

    const layer = L.geoJSON(geoData, {
      style: (feature) => {
        const score = feature?.properties?.safe_0_100 ?? 50;
        return {
          fillColor: safetyColor(score),
          fillOpacity: 0.7,
          color: "#ffffff",
          weight: 0.5,
          opacity: 0.6,
        };
      },
      onEachFeature: (feature, layer) => {
        const p = feature.properties || {};
        const name = p.NAME
          ? `${p.NAME}, ${p.STUSPS || p.STATE_NAME || ""}`
          : `County ${p.GEOID || "N/A"}`;
        const safeScore = p.safe_0_100 ?? 0;

        const hazards = [
          { label: "🌊 Coastal Flood", key: "CFLD_norm" },
          { label: "🫨 Earthquake", key: "ERQK_norm" },
          { label: "🌀 Hurricane", key: "HRCN_norm" },
          { label: "🌪️ Tornado", key: "TRND_norm" },
          { label: "🔥 Wildfire", key: "WFIR_norm" },
          { label: "❄️ Cold Wave", key: "CWAV_norm" },
          { label: "🌡️ Heat Wave", key: "HWAV_norm" },
          { label: "☀️ Drought", key: "DRGT_norm" },
        ];

        const hazardRows = hazards
          .map((h) => {
            const raw = Number(p[h.key] ?? 0);
            const score = (raw * 100).toFixed(0);
            return `<div style="display:flex;justify-content:space-between;gap:12px;"><span>${h.label}</span><strong>${score}</strong></div>`;
          })
          .join("");

        layer.bindTooltip(
          `<div style="font-size:12px;line-height:1.6;min-width:180px;">
            <div style="font-weight:700;margin-bottom:4px;">${name}</div>
            <div style="margin-bottom:6px;">Safety Score: <strong>${Number(safeScore).toFixed(1)}</strong></div>
            <div style="border-top:1px solid #e5e7eb;padding-top:4px;font-size:11px;">
              <div style="font-weight:600;margin-bottom:2px;color:#6b7280;">Hazard Risk (0–100)</div>
              ${hazardRows}
            </div>
          </div>`,
          { sticky: true }
        );
      },
    });

    layer.addTo(map);
    layerRef.current = layer;
    console.log("[MapView] Added", geoData.features.length, "features to Leaflet map");
  }, [geoData]);

  return (
    <div className="relative flex-1 h-screen">
      <div ref={containerRef} className="w-full h-full" />
      {loading && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-[1000] bg-card/90 backdrop-blur-sm px-4 py-2 rounded-full shadow-lg border border-border">
          <span className="text-sm text-muted-foreground animate-pulse">Updating map…</span>
        </div>
      )}
    </div>
  );
}
