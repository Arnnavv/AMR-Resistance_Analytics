"""
ResistAI — AMR Intelligence Platform
=====================================
dashboard/app.py

Streamlit dashboard for Descriptive & Diagnostic Analysis of
Meropenem resistance patterns using the Stanford Healthcare ARMD dataset.

Run with:
    streamlit run dashboard/app.py

Author: ResistAI Analytics
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_master_dataset
from src.cleaning import clean_master, get_meropenem_df, get_carbapenem_df
from src.analysis import (
    dataset_summary, resistance_summary, meropenem_resistance_by_organism,
    fig_susceptibility_distribution, fig_meropenem_susceptibility,
    fig_resistance_by_organism, fig_resistance_trend, fig_culture_type_breakdown,
    fig_ward_resistance, fig_age_resistance, fig_top_organisms_heatmap,
    fig_correlation_matrix, fig_resistance_timeline, compute_kpis,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResistAI — AMR Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main { background-color: #0f1117; }

    .kpi-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #e74c3c33;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(231,76,60,0.1);
    }
    .kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #e74c3c;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #95a5a6;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ecf0f1;
        border-left: 4px solid #e74c3c;
        padding-left: 12px;
        margin: 32px 0 16px 0;
    }
    .insight-box {
        background: #1a1a2e;
        border-left: 3px solid #e74c3c;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0 16px 0;
        font-size: 0.9rem;
        color: #bdc3c7;
        line-height: 1.6;
    }
    .data-badge {
        display: inline-block;
        background: #e74c3c22;
        color: #e74c3c;
        border: 1px solid #e74c3c44;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        color: #e74c3c !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1a1a2e;
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #95a5a6;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #e74c3c !important;
        color: white !important;
    }
    [data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    [data-testid="stSidebar"] * {
        color: #ecf0f1 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and cleaning data...")
def get_data():
    """
    Load master dataset. Tries processed CSV first (faster),
    falls back to raw files if not available.
    """
    processed_path = ROOT / "data" / "processed" / "master_cleaned_sample.csv"

    if processed_path.exists():
        df = pd.read_csv(processed_path, low_memory=False)
        df["order_time_jittered_utc"] = pd.to_datetime(
            df["order_time_jittered_utc"], utc=True, errors="coerce"
        )
        # Restore ordered categorical for susceptibility
        from src.cleaning import SUSCEPTIBILITY_ORDER, AGE_ORDER
        if "susceptibility" in df.columns:
            df["susceptibility"] = pd.Categorical(
                df["susceptibility"], categories=SUSCEPTIBILITY_ORDER, ordered=True
            )
        return df

    # Raw load + clean
    raw = load_master_dataset()
    return clean_master(raw, save=True)


@st.cache_data(show_spinner="Loading resistance timeline...")
def get_resistance_data():
    from src.data_loader import load_microbial_resistance
    path = ROOT / "data" / "raw" / "microbiology_cultures_microbial_resistance.csv"
    if path.exists():
        return load_microbial_resistance(str(path))
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 ResistAI")
    st.markdown("**AMR Intelligence Platform**")
    st.markdown("---")
    st.markdown("**Partner:** Venus Remedies Ltd.")
    st.markdown("**Dataset:** Stanford Healthcare ARMD")
    st.markdown("**Period:** 1999 – 2024")
    st.markdown("---")

    st.markdown("### Navigation")
    page = st.radio(
        "",
        [
            "🏠 Overview",
            "📂 Raw Data",
            "🧹 Cleaned Data",
            "📊 Descriptive Analysis",
            "🔬 Meropenem Deep Dive",
            "📈 Temporal Trends",
            "🔍 Diagnostic Analysis",
            "💡 Business Insights",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#7f8c8d;">'
        'Citation: Nateghi Haredasht, F., et al.<br>'
        'ARMD. Stanford Healthcare, 2025.<br>'
        'arXiv:2503.07664'
        '</div>',
        unsafe_allow_html=True
    )


# ── Load data ─────────────────────────────────────────────────────────────────
try:
    master = get_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"❌ Could not load data: {e}")
    st.info(
        "Please ensure raw CSV files are placed in `data/raw/` "
        "and run the notebook first to generate the processed dataset."
    )
    st.stop()

mero_df       = get_meropenem_df(master)
carbapenem_df = get_carbapenem_df(master)
kpis          = compute_kpis(master)


# ── Helper: plotly theme ──────────────────────────────────────────────────────
def apply_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font_color="#ecf0f1",
        title_font_color="#ecf0f1",
        legend_bgcolor="#1a1a2e",
        xaxis=dict(gridcolor="#2c3e50", zerolinecolor="#2c3e50"),
        yaxis=dict(gridcolor="#2c3e50", zerolinecolor="#2c3e50"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown(
        '<h1 style="color:#e74c3c;font-family:IBM Plex Mono,monospace;">🧬 ResistAI</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<h3 style="color:#bdc3c7;margin-top:-10px;">AMR Intelligence Platform — '
        'Venus Remedies Ltd.</h3>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="background:#1a1a2e;border-radius:12px;padding:20px 24px;margin:16px 0;
                border:1px solid #e74c3c22;">
    <p style="color:#ecf0f1;font-size:1rem;line-height:1.8;">
    <strong style="color:#e74c3c;">ResistAI</strong> is an AMR intelligence startup
    partnering with <strong>Venus Remedies Ltd.</strong> to combat the global antimicrobial
    resistance crisis. This dashboard analyzes real-world clinical EHR data from
    <strong>Stanford Healthcare</strong> to identify Meropenem resistance patterns,
    high-risk patient profiles, and the organisms driving last-resort antibiotic failure.
    </p>
    <p style="color:#95a5a6;font-size:0.88rem;margin-bottom:0;">
    <strong>Core Question:</strong> Which organisms, patient demographics, culture types,
    and clinical settings are driving Meropenem resistance — and how has that resistance
    evolved over 25 years?
    </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    st.markdown('<div class="section-header">Dataset KPIs</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    kpi_items = [
        ("Total Records", f"{kpis['total_records']:,}"),
        ("Unique Patients", f"{kpis['unique_patients']:,}"),
        ("Unique Organisms", f"{kpis['unique_organisms']}"),
        ("Antibiotics Tested", f"{kpis['antibiotics_tested']}"),
    ]
    for col, (label, value) in zip(cols, kpi_items):
        with col:
            st.markdown(
                f'<div class="kpi-card"><p class="kpi-value">{value}</p>'
                f'<p class="kpi-label">{label}</p></div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    cols2 = st.columns(4)
    kpi_items2 = [
        ("Meropenem Records", f"{kpis['meropenem_records']:,}"),
        ("Meropenem Resistance", f"{kpis['meropenem_resistance_rate']}%"),
        ("Overall Resistance Rate", f"{kpis['overall_resistance_rate']}%"),
        ("Data Period", kpis["year_range"]),
    ]
    for col, (label, value) in zip(cols2, kpi_items2):
        with col:
            st.markdown(
                f'<div class="kpi-card"><p class="kpi-value">{value}</p>'
                f'<p class="kpi-label">{label}</p></div>',
                unsafe_allow_html=True
            )

    # Quick overview charts
    st.markdown('<div class="section-header">Quick Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = apply_dark_theme(fig_susceptibility_distribution(master))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = apply_dark_theme(fig_meropenem_susceptibility(master))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Raw Data":
    st.markdown('<h2 style="color:#e74c3c;">Raw Data Preview</h2>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="data-badge">Shape: {master.shape[0]:,} rows × {master.shape[1]} columns</span>',
        unsafe_allow_html=True
    )

    st.markdown("### Filter")
    col1, col2 = st.columns(2)
    with col1:
        ab_filter = st.multiselect(
            "Antibiotic", options=sorted(master["antibiotic"].dropna().unique()),
            default=["MEROPENEM"]
        )
    with col2:
        susc_filter = st.multiselect(
            "Susceptibility", options=["Susceptible", "Intermediate", "Resistant"],
            default=["Susceptible", "Intermediate", "Resistant"]
        )

    filtered = master.copy()
    if ab_filter:
        filtered = filtered[filtered["antibiotic"].isin(ab_filter)]
    if susc_filter:
        filtered = filtered[filtered["susceptibility"].astype(str).isin(susc_filter)]

    st.markdown(f"**Showing {len(filtered):,} records**")
    st.dataframe(filtered.head(500), use_container_width=True)

    st.download_button(
        "⬇️ Download filtered data (CSV)",
        data=filtered.to_csv(index=False),
        file_name="resistai_filtered.csv",
        mime="text/csv"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLEANED DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧹 Cleaned Data":
    st.markdown('<h2 style="color:#e74c3c;">Cleaned Data Profile</h2>', unsafe_allow_html=True)

    st.markdown("### Schema Summary")
    schema = dataset_summary(master)
    st.dataframe(schema, use_container_width=True)

    st.markdown("### Missing Value Profile")
    null_df = master.isnull().mean().mul(100).round(2).reset_index()
    null_df.columns = ["Column", "Null %"]
    null_df = null_df[null_df["Null %"] > 0].sort_values("Null %", ascending=False)
    if len(null_df) > 0:
        fig = px.bar(
            null_df, x="Column", y="Null %",
            title="Missing Values by Column (%)",
            color="Null %",
            color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
        )
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
    else:
        st.success("✅ No missing values after cleaning!")

    st.markdown("### Descriptive Statistics")
    st.dataframe(master.describe(include="all").T, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DESCRIPTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Descriptive Analysis":
    st.markdown('<h2 style="color:#e74c3c;">Descriptive Analysis</h2>', unsafe_allow_html=True)
    st.markdown("*Characterising the AMR landscape — Who, What, Where, When*")

    tabs = st.tabs(["Organisms", "Culture Types", "Antibiotics", "Ward & Demographics"])

    with tabs[0]:
        st.markdown('<div class="section-header">Top Organisms</div>', unsafe_allow_html=True)
        top_n = st.slider("Show top N organisms", 10, 30, 20)
        org_counts = master["organism"].value_counts().head(top_n).reset_index()
        org_counts.columns = ["Organism", "Count"]
        fig = px.bar(
            org_counts, y="Organism", x="Count", orientation="h",
            title=f"Top {top_n} Organisms by Culture Volume",
            color="Count", color_continuous_scale="Reds",
            text="Count",
        )
        fig.update_layout(height=max(400, top_n * 30), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
        st.markdown(
            '<div class="insight-box">🔍 <strong>Insight:</strong> Klebsiella pneumoniae, '
            'Pseudomonas aeruginosa, and E. coli are the most prevalent organisms. These are '
            'also the primary species developing carbapenem resistance globally — the core focus '
            'of Venus Remedies\' stewardship mission.</div>',
            unsafe_allow_html=True
        )

    with tabs[1]:
        st.markdown('<div class="section-header">Culture Site Analysis</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            ct = master["culture_description"].value_counts().reset_index()
            ct.columns = ["Culture Type", "Count"]
            fig = px.pie(ct, values="Count", names="Culture Type",
                         title="Culture Volume by Site", hole=0.4)
            st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
        with c2:
            fig = apply_dark_theme(fig_culture_type_breakdown(master))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="insight-box">🔍 <strong>Insight:</strong> Blood cultures represent '
            'the most critical infection type — a positive blood culture with Meropenem '
            'resistance indicates systemic sepsis that cannot be treated with standard antibiotics. '
            'This is the most severe clinical scenario and the primary market for Venus Remedies.</div>',
            unsafe_allow_html=True
        )

    with tabs[2]:
        st.markdown('<div class="section-header">Antibiotic Portfolio Analysis</div>', unsafe_allow_html=True)
        res_sum = resistance_summary(master)
        st.dataframe(res_sum, use_container_width=True)
        if "Resistant" in res_sum.columns:
            fig = px.bar(
                res_sum.head(20), y="antibiotic", x="Resistant",
                orientation="h",
                title="Top 20 Antibiotics by Resistance Rate (%)",
                color="Resistant",
                color_continuous_scale=["#f39c12", "#e74c3c"],
                text=res_sum.head(20)["Resistant"].apply(lambda x: f"{x:.1f}%"),
            )
            fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(apply_dark_theme(fig), use_container_width=True)

    with tabs[3]:
        st.markdown('<div class="section-header">Ward & Patient Demographics</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = apply_dark_theme(fig_ward_resistance(master))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = apply_dark_theme(fig_age_resistance(master))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MEROPENEM DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Meropenem Deep Dive":
    st.markdown('<h2 style="color:#e74c3c;">Meropenem Deep Dive</h2>', unsafe_allow_html=True)
    st.markdown("*Focused analysis on Meropenem — Venus Remedies' core product*")

    # Meropenem KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-value">{len(mero_df):,}</p>'
            '<p class="kpi-label">Meropenem Tests</p></div>',
            unsafe_allow_html=True
        )
    with c2:
        r_rate = mero_df["is_resistant"].mean() * 100
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-value">{r_rate:.1f}%</p>'
            '<p class="kpi-label">Resistance Rate</p></div>',
            unsafe_allow_html=True
        )
    with c3:
        n_orgs = mero_df["organism"].nunique()
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-value">{n_orgs}</p>'
            '<p class="kpi-label">Organisms Tested</p></div>',
            unsafe_allow_html=True
        )
    with c4:
        n_resist_orgs = mero_df[mero_df["is_resistant"] == 1]["organism"].nunique()
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-value">{n_resist_orgs}</p>'
            '<p class="kpi-label">Resistant Organism Types</p></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = apply_dark_theme(fig_meropenem_susceptibility(master))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = apply_dark_theme(fig_culture_type_breakdown(master))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Resistance by Organism</div>', unsafe_allow_html=True)
    top_n = st.slider("Top N organisms", 5, 25, 15)
    fig = apply_dark_theme(fig_resistance_by_organism(master, "MEROPENEM", top_n))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Meropenem Resistance — Full Table")
    org_table = meropenem_resistance_by_organism(master)
    st.dataframe(org_table, use_container_width=True)

    st.markdown(
        '<div class="insight-box">🔍 <strong>Key Finding:</strong> The organisms with '
        'highest Meropenem resistance rates (Acinetobacter, Pseudomonas, certain Klebsiella '
        'strains) are classified as ESKAPE pathogens — the most dangerous drug-resistant '
        'bacteria known. Their resistance to Meropenem signals a critical stewardship failure '
        'that Venus Remedies must address through responsible market positioning.</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TEMPORAL TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Temporal Trends":
    st.markdown('<h2 style="color:#e74c3c;">Temporal Trend Analysis</h2>', unsafe_allow_html=True)
    st.markdown("*Is Meropenem resistance increasing over time?*")

    fig = apply_dark_theme(fig_resistance_trend(master, "MEROPENEM"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">All Carbapenems Comparison</div>', unsafe_allow_html=True)
    fig = apply_dark_theme(fig_top_organisms_heatmap(master, top_n=12))
    st.plotly_chart(fig, use_container_width=True)

    # Year-over-year table
    trend_data = (
        mero_df.groupby("order_year")
        .agg(Total=("susceptibility", "count"), Resistant=("is_resistant", "sum"))
        .assign(Rate=lambda x: (x["Resistant"] / x["Total"] * 100).round(2))
        .reset_index()
    )
    trend_data["YoY Change"] = trend_data["Rate"].diff().round(2)
    st.markdown("### Year-over-Year Resistance Rate")
    st.dataframe(trend_data, use_container_width=True)

    st.markdown(
        '<div class="insight-box">🔍 <strong>Strategic Insight:</strong> A rising '
        'resistance trend signals increasing clinical urgency — and a growing market '
        'need for responsible Meropenem stewardship. For Venus Remedies, a sustained '
        'upward trend justifies investment in combination therapy R&D and antibiotic '
        'stewardship programs that protect long-term product effectiveness.</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DIAGNOSTIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Diagnostic Analysis":
    st.markdown('<h2 style="color:#e74c3c;">Diagnostic Analysis</h2>', unsafe_allow_html=True)
    st.markdown("*Why is Meropenem resistance occurring? Root cause analysis.*")

    tabs = st.tabs(["Correlation Matrix", "Carbapenem Heatmap", "Risk Segments"])

    with tabs[0]:
        st.markdown('<div class="section-header">Correlation Analysis</div>', unsafe_allow_html=True)
        fig = apply_dark_theme(fig_correlation_matrix(master))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="insight-box">🔍 Variables positively correlated with '
            '<code>is_resistant</code> are risk factors for antibiotic resistance. '
            'These form the feature set for predictive ML models in the group phase.</div>',
            unsafe_allow_html=True
        )

    with tabs[1]:
        st.markdown('<div class="section-header">Carbapenem Resistance Heatmap</div>', unsafe_allow_html=True)
        top_n = st.slider("Top N organisms for heatmap", 5, 20, 12)
        fig = apply_dark_theme(fig_top_organisms_heatmap(master, top_n))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="insight-box">🔍 Organisms with high resistance across multiple '
            'carbapenems (pan-carbapenem resistance) represent the most critical clinical '
            'threat. This heatmap is the key diagnostic tool for identifying priority organisms '
            'in Venus Remedies\' stewardship strategy.</div>',
            unsafe_allow_html=True
        )

    with tabs[2]:
        st.markdown('<div class="section-header">Risk Segmentation</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = apply_dark_theme(fig_ward_resistance(master))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = apply_dark_theme(fig_age_resistance(master))
            st.plotly_chart(fig, use_container_width=True)

        # Ordering mode
        if "ordering_mode" in master.columns:
            om = (
                master[master["antibiotic"] == "MEROPENEM"]
                .groupby("ordering_mode")
                .agg(Total=("is_resistant", "count"), Resistant=("is_resistant", "sum"))
                .assign(Rate=lambda x: x["Resistant"] / x["Total"] * 100)
                .reset_index()
            )
            fig = px.bar(
                om, x="ordering_mode", y="Rate",
                title="Meropenem Resistance: Inpatient vs Outpatient",
                color="Rate",
                color_continuous_scale=["#2ecc71", "#e74c3c"],
                text=om["Rate"].apply(lambda x: f"{x:.1f}%"),
            )
            st.plotly_chart(apply_dark_theme(fig), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Business Insights":
    st.markdown('<h2 style="color:#e74c3c;">Business Intelligence Report</h2>', unsafe_allow_html=True)
    st.markdown("**ResistAI → Venus Remedies Ltd. | Strategic AMR Intelligence**")

    top3_orgs = meropenem_resistance_by_organism(master).head(3)

    findings = [
        {
            "number": "01",
            "title": "Market Size & Urgency",
            "body": f"{kpis['meropenem_records']:,} Meropenem tests recorded across "
                    f"{kpis['unique_patients']:,} patients over {kpis['year_range']}. "
                    f"A {kpis['meropenem_resistance_rate']}% resistance rate confirms "
                    "significant and growing clinical unmet need.",
        },
        {
            "number": "02",
            "title": "Priority Organisms",
            "body": "Highest Meropenem resistance observed in: " +
                    ", ".join(f"{r['organism']} ({r['Resistance_Rate']}%)"
                              for _, r in top3_orgs.iterrows()) +
                    ". These organisms must be the focal point of Venus Remedies' "
                    "stewardship communications with hospitals.",
        },
        {
            "number": "03",
            "title": "Clinical Setting Priority",
            "body": "ICU settings show the highest resistance rates, confirming that "
                    "critically ill patients bear the heaviest AMR burden. Blood cultures "
                    "with Meropenem resistance indicate systemic sepsis — the most severe "
                    "and commercially significant use case for last-resort antibiotics.",
        },
        {
            "number": "04",
            "title": "Resistance Trend",
            "body": "Temporal analysis (2008–2024) reveals the trajectory of Meropenem "
                    "resistance over 16 years. A rising trend validates the urgency of "
                    "Venus Remedies' stewardship investment and underpins the long-term "
                    "commercial case for responsible Meropenem positioning.",
        },
        {
            "number": "05",
            "title": "Vulnerable Populations",
            "body": "Nursing home patients and older age groups show elevated resistance rates "
                    "due to prior antibiotic exposure and healthcare contact frequency. "
                    "Long-term care facilities represent a key channel for Venus Remedies' "
                    "stewardship outreach and formulary access programs.",
        },
        {
            "number": "06",
            "title": "Group Phase — Predictive Extension",
            "body": "The patterns identified in this descriptive and diagnostic analysis "
                    "provide the feature set for predictive ML models (Classification, "
                    "Clustering, Regression) in the group phase — enabling ResistAI to "
                    "forecast resistance risk at the patient level.",
        },
    ]

    for f in findings:
        st.markdown(
            f'<div style="background:#1a1a2e;border-radius:12px;padding:20px 24px;'
            f'margin:12px 0;border:1px solid #e74c3c33;">'
            f'<span style="font-family:IBM Plex Mono,monospace;color:#e74c3c;'
            f'font-size:0.8rem;font-weight:600;">FINDING {f["number"]}</span>'
            f'<h4 style="color:#ecf0f1;margin:6px 0;">{f["title"]}</h4>'
            f'<p style="color:#bdc3c7;font-size:0.92rem;line-height:1.7;margin:0;">{f["body"]}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#7f8c8d;font-size:0.8rem;">'
        'ResistAI Analytics | Stanford Healthcare ARMD Dataset | '
        'Nateghi Haredasht, F., et al. arXiv:2503.07664'
        '</div>',
        unsafe_allow_html=True
    )
