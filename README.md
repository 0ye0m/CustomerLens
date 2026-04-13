# CustomerLens - Multi-Dimensional Segmentation Intelligence Platform

CustomerLens is a production-grade customer segmentation and analytics platform built with Streamlit. It combines RFM scoring, multi-algorithm clustering, churn prediction, CLV forecasting, and an AI strategy layer to deliver executive-ready insights from real or demo data.

## Highlights
- Real data ingestion with auto column mapping and validation
- Multi-layer segmentation using RFM and clustering models
- Churn prediction with explainable feature importance and what-if simulation
- CLV forecasting with tiering and cluster-specific value breakdowns
- Strategy engine with budget allocation and PDF export
- AI Analyst page powered by Groq (optional)

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
│   ├── data_manager.py
│   ├── rfm_analysis.py
│   ├── clustering.py
│   ├── dimensionality.py
│   ├── churn_model.py
│   ├── clv_model.py
│   └── recommender.py
├── pages/
│   ├── 0_Data_Input.py
│   ├── 1_Overview.py
│   ├── 2_RFM_Analysis.py
│   ├── 3_Clustering.py
│   ├── 4_Churn_Prediction.py
│   ├── 5_CLV_Forecast.py
│   ├── 6_Segment_Personas.py
│   ├── 7_Strategy_Engine.py
│   └── 8_AI_Analyst.py
└── utils/
    ├── groq_client.py
    ├── helpers.py
    └── styling.py
```

## Setup
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Data Input
Use the **Your Data** page (page 0) to upload a CSV/Excel file, enter rows manually, or load demo datasets. Column names do not need to be exact; the app provides a mapping UI and will enrich missing fields.

### Required Columns
- customer_id
- last_purchase_date
- total_orders
- total_spend
- days_since_last_purchase (auto-derived if missing)

### Optional Columns
- age
- gender
- country
- city
- satisfaction_score
- churn_flag (auto-derived if missing)
- avg_order_value (auto-derived if missing)
- channel
- loyalty_points
- support_tickets_raised
- referral_count

### Data Cleaning Rules
- Date parsing supports multiple formats (dd/mm/yyyy, mm-dd-yyyy, etc.)
- Currency symbols are stripped from spend fields ($, EUR, GBP, INR)
- Duplicate customer_id values keep the last occurrence

## Demo Datasets
Choose a ready-made dataset from the **Use Demo Data** tab:
- E-commerce store (5,000 customers)
- SaaS company (2,000 customers)
- Retail chain (3,000 customers)

## AI Features (Optional)
CustomerLens integrates Groq for AI insights, strategy generation, and executive reporting. You can use the app without an API key; AI sections include toggles for safe fallback behavior.

### Configure Groq
Add a key in .streamlit/secrets.toml:
```toml
[groq]
api_key = "your_groq_api_key_here"
```
Or set the environment variable:
```bash
set GROQ_API_KEY=your_groq_api_key_here
```

## Notes
- All analytics and models are cached for fast iteration.
- The sidebar shows the active data source and AI model status.
