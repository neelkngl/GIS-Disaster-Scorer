import { Slider } from "@/components/ui/slider";

interface HazardSliderProps {
  label: string;
  code: string;
  value: number;
  onChange: (value: number) => void;
  icon: string;
}

export default function HazardSlider({ label, code, value, onChange, icon }: HazardSliderProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-base">{icon}</span>
          <span className="text-sm font-medium text-panel-foreground">{label}</span>
        </div>
        <span className="text-xs font-mono tabular-nums text-panel-muted min-w-[2.5rem] text-right">
          {value}
        </span>
      </div>
      <Slider
        min={0}
        max={100}
        step={1}
        value={[value]}
        onValueChange={(v) => onChange(v[0])}
        className="w-full"
      />
    </div>
  );
}
