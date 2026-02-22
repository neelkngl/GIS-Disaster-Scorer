# SafeHaven
https://safe-haven-investing.lovable.app/

SafeHaven is a personalized, risk‑adjusted housing desirability tool that helps people find where to live based on their tolerance for different natural disasters.

## Overview
Our goal is to build a risk‑adjusted housing desirability index using weather and disaster datasets along with current housing “hotness” data to generate a desirability score by county. Users can specify how averse they are to different natural disasters (for example: coastal flooding, hurricanes, wildfires, earthquakes, drought, heat waves), and SafeHaven will surface locations that best match their preferences through an interactive map.

## How It Works
1. We collect and aggregate:
   - Historical and projected disaster and weather data by region
   - Housing demand / “hotness” and market activity data by city/metro
2. We compute a risk‑adjusted desirability score that combines:
   - Baseline housing desirability (e.g., demand, stability, growth)
   - Hazard‑specific risk metrics (e.g., flood risk, wildfire risk, seismic risk)
   - User‑specified weights for each hazard type
3. We display results on an interactive map where users can:
   - Adjust sliders or weights for each disaster type
   - Hover over cities/metros to see their overall SafeHaven score
   - View a breakdown of sub‑scores by hazard and housing metric

## Features
- Personalized risk weighting for multiple natural disasters
- Risk‑adjusted housing desirability index by city/metropolitan area
- Interactive map UI with hover‑to‑inspect scores and breakdowns
- Transparent scoring logic so users can understand tradeoffs

## Tech Stack
- Backend: Python
- Data: FEMA NRI Data, Zillow
- Frontend: Lovable, Python, GeoJSON (interactive map)

## Example User Flow
1. User selects which natural disasters they care most about avoiding.
2. User adjusts sliders to indicate how risk‑averse they are for each hazard.
3. SafeHaven recalculates scores across counties in real time.
4. Interactive map highlights the best‑fit locations and shows detailed breakdowns on hover or click.

## File Descriptions
- project_paths.py: Utility module that standardizes and resolves project directory paths (raw, processed, out) so all scripts can reliably locate datasets without hardcoded file paths.
- preprocess.py: Ingests and cleans raw FEMA NRI, Zillow market value and temp indices, and ZIP/county crosswalk data, producing normalized, analysis-ready CSV files used for scoring and map generation.
- scoring.py: Builds the county-level risk model by combining normalized hazard risk, financial exposure, and cost metrics into composite safety scores and financial “climate tax” indicators.
- build_geojson.py: Joins scored county metrics to Census county boundary geometries and exports a GeoJSON layer used by the frontend map for interactive visualization.
- app.py: FastAPI backend that dynamically recalculates county safety scores based on slider-weighted hazard preferences and returns filtered GeoJSON for real-time map updates.

## Data Descriptions
- county_scores_balanced.csv: County-level scoring output combining hazard risk, financial exposure, and cost normalization into composite safety scores and financial impact metrics.
- county_scores_balanced_simplified.geojson: Geospatial boundary layer of U.S. counties enriched with computed safety and financial metrics, used by the frontend for dynamic, slider-driven map visualization.

## What's Next
The next phase of development focuses on increasing property-level precision and real-world financial relevance. Integrating the Zillow Zestimate API (which was unable to be obtained in a timely manner) will allow the platform to move from county-level modeling to address-specific risk scoring, translating hazard exposure directly into estimated property value impact and projected annual “climate tax” per home. This will enable users to enter an address and see personalized exposure, rather than relying on county averages. This upgrades would shift the app from an exploratory hazard visualization tool into a true climate-adjusted real estate decision engine.
