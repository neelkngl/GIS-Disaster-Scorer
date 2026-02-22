import { useCallback } from "react";
import SliderPanel from "@/components/SliderPanel";
import MapView from "@/components/MapView";
import RankingPanel from "@/components/RankingPanel";
import { useMapFilter } from "@/hooks/useMapFilter";

const noop = () => {};

const Index = () => {
  const { weights, updateWeight, submit, geoData, loading, error } = useMapFilter();

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <SliderPanel
        weights={weights}
        onWeightChange={updateWeight}
        onSubmit={submit}
        loading={loading}
        error={error}
      />
      <MapView geoData={geoData} loading={loading} onBoundsChange={noop} />
      <RankingPanel geoData={geoData} />
    </div>
  );
};

export default Index;
