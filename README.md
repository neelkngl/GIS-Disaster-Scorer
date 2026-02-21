# SafeHaven
SafeHaven is a personalized, risk‑adjusted housing desirability tool that helps people find where to live based on their tolerance for different natural disasters.

## Overview
Our goal is to build a risk‑adjusted housing desirability index using weather and disaster datasets along with current housing “hotness” data to generate a desirability score by city/metropolitan area. Users can specify how averse they are to different natural disasters (for example: coastal flooding, hurricanes, wildfires, earthquakes, drought, heat waves), and SafeHaven will surface locations that best match their preferences through an interactive map.

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
- Data: Public weather and disaster datasets, housing hotness data sources
- Frontend: Lovable, Python (interactive map)

## Example User Flow
1. User selects which natural disasters they care most about avoiding.
2. User adjusts sliders to indicate how risk‑averse they are for each hazard.
3. SafeHaven recalculates scores across cities/metros in real time.
4. Interactive map highlights the best‑fit locations and shows detailed breakdowns on hover or click.
