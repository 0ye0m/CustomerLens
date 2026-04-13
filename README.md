# CustomerLens - Multi-Dimensional Segmentation Intelligence Platform

CustomerLens is a portfolio-grade customer intelligence platform built with Streamlit. It turns raw customer activity into segmentation insights using RFM scoring, multi-algorithm clustering, churn prediction, CLV forecasting, and optional AI-driven strategy generation. The app supports real customer data uploads, manual entry, and multiple demo datasets.

## Highlights
- Real data ingestion with auto column mapping, validation, and enrichment
- Multi-layer segmentation using RFM and three clustering algorithms
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

## What the App Does (End-to-End)
1. Ingest data from **Your Data** (upload/manual/demo) and normalize columns.
2. Enrich the dataset with derived fields (recency, AOV, churn flags) when missing.
3. Compute RFM scores and named segments.
4. Run clustering (K-Means, DBSCAN, Hierarchical) and pick the best model.
5. Forecast churn probability and 12-month CLV.
6. Surface dashboards, personas, and strategy recommendations.
7. Optionally generate AI explanations, strategies, and executive reports.

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
- Marketplace (4,000 customers)
- D2C brand (2,500 customers)
- Hospitality (1,800 customers)

## Pages Overview
- **Your Data**: upload, map, validate, and load customer data
- **Overview**: executive KPI summary with geo and segment charts
- **RFM Analysis**: score distributions, 3D view, and segment explorer
- **Clustering**: algorithm comparison and 2D/3D embeddings
- **Churn Prediction**: ROC, feature importance, risk table, what-if simulator
- **CLV Forecast**: tier distribution, CLV vs churn scatter, top customers
- **Segment Personas**: persona cards with optional AI insights
- **Strategy Engine**: recommendations, budget allocation, PDF export
- **AI Analyst**: segment explainer, strategy generator, chat, executive report

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

## Deployment Notes
- For Streamlit Community Cloud, add your Groq key in the app secrets.
- Do not commit real API keys. Use environment variables or local secrets files.

## Troubleshooting
- If clustering errors mention missing columns, verify your mapping covers required fields.
- If PDF export fails, ensure the Strategy Engine has valid values and retry.
- If AI calls fail, confirm the Groq key and model selection in the sidebar.

## Notes
- All analytics and models are cached for fast iteration.
- The sidebar shows the active data source and AI model status.
