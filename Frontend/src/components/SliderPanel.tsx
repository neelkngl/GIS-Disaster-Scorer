import { type Weights } from "@/lib/scoring";
import HazardSlider from "./HazardSlider";
import { Shield, Loader2 } from "lucide-react";

interface SliderPanelProps {
  weights: Weights;
  onWeightChange: (key: keyof Weights, value: number) => void;
  onSubmit: () => void;
  loading: boolean;
  error: string | null;
}

const HAZARDS: { label: string; code: keyof Weights; icon: string }[] = [
  { label: "Coastal Flooding", code: "CFLD", icon: "🌊" },
  { label: "Earthquake", code: "ERQK", icon: "🫨" },
  { label: "Hurricane", code: "HRCN", icon: "🌀" },
  { label: "Tornado", code: "TRND", icon: "🌪️" },
  { label: "Wildfire", code: "WFIR", icon: "🔥" },
  { label: "Cold Wave", code: "CWAV", icon: "❄️" },
  { label: "Heat Wave", code: "HWAV", icon: "🌡️" },
  { label: "Drought", code: "DRGT", icon: "☀️" },
];

export default function SliderPanel({
  weights,
  onWeightChange,
  onSubmit,
  loading,
  error,
}: SliderPanelProps) {
  return (
    <aside className="w-[360px] min-w-[320px] h-screen overflow-y-auto panel-dark flex flex-col border-r border-panel-border">
      {/* Header */}
      <div className="p-6 pb-4">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-lg bg-panel-slider-fill/20">
            <Shield className="w-6 h-6 text-accent" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-panel-foreground">
            SafeHaven
          </h1>
        </div>
        <p className="text-sm text-panel-muted leading-relaxed">
          Find safer places to live based on climate risk
        </p>
      </div>

      {/* Status */}
      <div className="px-6 pb-3">
        <div className="flex items-center gap-2 text-xs">
          {loading && <Loader2 className="w-3 h-3 animate-spin text-panel-muted" />}
          {loading && <span className="text-panel-muted">Scoring counties…</span>}
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="mx-6 mb-3 p-3 rounded-lg bg-destructive/20 border border-destructive/30">
          <p className="text-xs text-destructive-foreground">{error}</p>
        </div>
      )}

      {/* Sliders */}
      <div className="flex-1 px-6 pb-6 space-y-5">
        <p className="text-xs uppercase tracking-wider text-panel-muted font-semibold">
          Risk Avoidance
        </p>
        {HAZARDS.map((h) => (
          <HazardSlider
            key={h.code}
            label={h.label}
            code={h.code}
            icon={h.icon}
            value={weights[h.code]}
            onChange={(v) => onWeightChange(h.code, v)}
          />
        ))}

        {/* Submit Button */}
        <button
          onClick={onSubmit}
          disabled={loading}
          className="w-full mt-4 py-2.5 rounded-lg bg-accent text-accent-foreground font-semibold text-sm hover:opacity-90 transition-opacity disabled:opacity-50"
        >
          {loading ? "Updating…" : "Apply Filters"}
        </button>
      </div>

      {/* Legend */}
      <div className="px-6 py-4 border-t border-panel-border">
        <p className="text-xs text-panel-muted mb-2 font-medium">Safety Score</p>
        <div className="risk-gradient-legend h-2 rounded-full" style={{ transform: "scaleX(-1)" }} />
        <div className="flex justify-between text-[10px] text-panel-muted mt-1">
          <span>Dangerous</span>
          <span>Safe</span>
        </div>
      </div>
    </aside>
  );
}
