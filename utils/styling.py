from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    """Inject the CustomerLens theme and component styles."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

        :root {
            --bg: #0E1117;
            --card: #1C1F26;
            --card-muted: #161A21;
            --accent: #6C63FF;
            --accent-soft: rgba(108, 99, 255, 0.18);
            --text: #F5F6FA;
            --muted: #9AA4B2;
            --success: #2ECC71;
            --warning: #F5B041;
            --danger: #E74C3C;
            --green: #2ECC71;
            --yellow: #F5B041;
            --red: #E74C3C;
        }

        html, body, [class*="css"]  {
            font-family: 'IBM Plex Sans', sans-serif;
            background-color: var(--bg);
            color: var(--text);
        }

        .stApp {
            background: radial-gradient(circle at top left, rgba(108, 99, 255, 0.12), transparent 45%),
                        radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.03), transparent 40%),
                        var(--bg);
        }

        h1, h2, h3, h4, h5 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: 0.2px;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
        }

        .kpi-card {
            background: var(--card);
            border-radius: 16px;
            padding: 16px 18px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.25);
            animation: fadeIn 0.6s ease;
        }

        .kpi-title {
            font-size: 0.85rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .kpi-value {
            font-size: 1.6rem;
            font-weight: 600;
            margin-top: 6px;
        }

        .kpi-delta {
            font-size: 0.85rem;
            margin-top: 4px;
        }

        .card {
            background: var(--card);
            border-radius: 16px;
            padding: 18px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 10px 22px rgba(0, 0, 0, 0.25);
        }

        .segment-card {
            background: linear-gradient(135deg, rgba(108, 99, 255, 0.12), rgba(28, 31, 38, 0.9));
            border-radius: 18px;
            padding: 16px;
            border: 1px solid rgba(108, 99, 255, 0.2);
            min-height: 240px;
        }

        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge.green { background: rgba(46, 204, 113, 0.2); color: var(--success); }
        .badge.yellow { background: rgba(245, 176, 65, 0.2); color: var(--warning); }
        .badge.red { background: rgba(231, 76, 60, 0.2); color: var(--danger); }

        .strategy-box {
            background: var(--card-muted);
            border-left: 4px solid var(--accent);
            padding: 16px 18px;
            border-radius: 12px;
        }

        .email-preview {
            background: #0B0E14;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 0.9rem;
        }

        .stButton > button {
            background-color: var(--accent);
            color: white;
            border-radius: 10px;
            border: none;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
