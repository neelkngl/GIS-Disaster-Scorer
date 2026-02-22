import { useMemo, useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TrendingUp, TrendingDown, PanelRightClose, PanelRightOpen } from "lucide-react";

interface RankingPanelProps {
  geoData: GeoJSON.FeatureCollection | null;
}

interface CountyRank {
  name: string;
  state: string;
  value: number;
}

export default function RankingPanel({ geoData }: RankingPanelProps) {
  const [collapsed, setCollapsed] = useState(false);

  const rankings = useMemo(() => {
    if (!geoData) return null;

    const entries: { name: string; state: string; score: number }[] = [];

    for (const f of geoData.features) {
      const p = f.properties ?? {};
      const score = Number(p.safe_0_100 ?? 0);
      const name = p.NAME ?? p.name ?? "Unknown";
      const state = p.STATE_NAME ?? p.state_name ?? p.STUSPS ?? "";
      entries.push({ name, state, score });
    }

    entries.sort((a, b) => b.score - a.score); // higher = safer

    const safest = entries.slice(0, 3).map((e) => ({ name: e.name, state: e.state, value: e.score }));
    const riskiest = entries.slice(-3).reverse().map((e) => ({ name: e.name, state: e.state, value: e.score }));

    return { safest, riskiest };
  }, [geoData]);

  if (!rankings) return null;

  if (collapsed) {
    return (
      <button
        onClick={() => setCollapsed(false)}
        className="h-screen flex items-center justify-center w-10 panel-dark border-l border-panel-border hover:bg-panel-border/30 transition-colors"
        title="Show Rankings"
      >
        <PanelRightClose className="w-4 h-4 text-panel-muted" />
      </button>
    );
  }

  return (
    <aside className="w-[300px] min-w-[260px] h-screen flex flex-col panel-dark border-l border-panel-border">
      <div className="p-4 pb-3 border-b border-panel-border flex items-center justify-between">
        <div>
          <h2 className="text-sm font-bold tracking-tight text-panel-foreground uppercase">
            County Rankings
          </h2>
          <p className="text-xs text-panel-muted mt-1">Overall top & bottom 3</p>
        </div>
        <button
          onClick={() => setCollapsed(true)}
          className="p-1.5 rounded hover:bg-panel-border/30 transition-colors"
          title="Hide Rankings"
        >
          <PanelRightOpen className="w-4 h-4 text-panel-muted" />
        </button>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-5">
          {/* Safest */}
          <div className="space-y-2">
            <div className="flex items-center gap-1 text-[10px] text-emerald-400 uppercase tracking-wider font-medium">
              <TrendingUp className="w-3 h-3" />
              Safest Counties
            </div>
            {rankings.safest.map((c, i) => (
              <div key={i} className="flex items-center justify-between text-xs px-2 py-1.5 rounded bg-emerald-500/10">
                <span className="text-panel-foreground truncate mr-2">
                  {i + 1}. {c.name}{c.state ? `, ${c.state}` : ""}
                </span>
                <span className="text-emerald-400 font-mono text-[10px] shrink-0">
                  {c.value.toFixed(1)}
                </span>
              </div>
            ))}
          </div>

          {/* Riskiest */}
          <div className="space-y-2">
            <div className="flex items-center gap-1 text-[10px] text-red-400 uppercase tracking-wider font-medium">
              <TrendingDown className="w-3 h-3" />
              Riskiest Counties
            </div>
            {rankings.riskiest.map((c, i) => (
              <div key={i} className="flex items-center justify-between text-xs px-2 py-1.5 rounded bg-red-500/10">
                <span className="text-panel-foreground truncate mr-2">
                  {i + 1}. {c.name}{c.state ? `, ${c.state}` : ""}
                </span>
                <span className="text-red-400 font-mono text-[10px] shrink-0">
                  {c.value.toFixed(1)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </ScrollArea>
    </aside>
  );
}
