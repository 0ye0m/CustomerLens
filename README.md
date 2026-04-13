# CustomerLens - Multi-Dimensional Segmentation Intelligence Platform

CustomerLens is a production-grade customer segmentation and analytics platform built with Streamlit. It combines RFM scoring, multi-algorithm clustering, churn prediction, CLV forecasting, and a strategy engine to deliver executive-ready insights.

## Highlights
- Multi-layer segmentation using RFM and clustering models
- Churn prediction with explainable feature importance and what-if simulation
- CLV forecasting with tiering and cluster-specific value breakdowns
- Strategy engine with channel, offer, and budget recommendations
- Fully interactive dashboards powered by Plotly

## Tech Stack
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37.0-red)
![Pandas](https://img.shields.io/badge/Pandas-2.2.2-brightgreen)
![Scikit%20Learn](https://img.shields.io/badge/Scikit--Learn-1.4.2-orange)
![Plotly](https://img.shields.io/badge/Plotly-5.22.0-blueviolet)

## Project Structure
```
customer_lens/
├── app.py
├── requirements.txt
├── data/
│   └── generate_data.py
├── modules/
├── pages/
└── utils/
```

## Setup
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate the dataset (optional - the app auto-generates if missing):
   ```bash
   python data/generate_data.py
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Screenshots
Add screenshots to this section after running the app.

- Overview dashboard
- RFM analysis
- Cluster explorer
- Strategy engine

## Notes
- All data is synthetic and generated locally.
- Models and analytics are cached for fast iteration in Streamlit.
