"""
StrataMed Analytics — AMR Intelligence Platform
=====================================
src/analysis.py

Reusable analytical functions for descriptive and diagnostic analysis.
All functions return either DataFrames (for tables) or Plotly figures
(for visualisations), making them usable in both the notebook and the
Streamlit dashboard without duplication.

Author: StrataMed Analytics Analytics
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Colour palette ────────────────────────────────────────────────────────────
# Clinical / professional colour scheme aligned with StrataMed Analytics brand
COLORS = {
    "Susceptible":   "#2ecc71",   # green  — safe
    "Intermediate":  "#f39c12",   # amber  — caution
    "Resistant":     "#e74c3c",   # red    — danger
    "primary":       "#1a1a2e",   # dark navy
    "accent":        "#e74c3c",   # red accent
    "muted":         "#7f8c8d",
    "bg":            "#f8f9fa",
}

SUSC_COLOR_MAP = {
    "Susceptible":  COLORS["Susceptible"],
    "Intermediate": COLORS["Intermediate"],
    "Resistant":    COLORS["Resistant"],
}


# ── Descriptive summary ───────────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a concise summary table of the master dataset.

    Returns a DataFrame with per-column stats:
    dtype, non-null count, null %, and top values.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "Column":       col,
            "Dtype":        str(s.dtype),
            "Non-Null":     s.notna().sum(),
            "Null %":       f"{s.isna().mean()*100:.1f}%",
            "Unique":       s.nunique(),
            "Top Values":   str(s.dropna().value_counts().head(3).index.tolist()),
        })
    return pd.DataFrame(rows)


def resistance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary of resistance rates across all antibiotics tested,
    sorted by resistance rate descending.

    Business use: Identifies which antibiotics are failing most often
    across all organisms — giving a portfolio view of AMR severity.
    """
    summary = (
        df[df["susceptibility"].notna()]
        .groupby("antibiotic")["susceptibility"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .rename("Percentage")
        .reset_index()
    )
    pivot = summary.pivot_table(
        index="antibiotic",
        columns="susceptibility",
        values="Percentage",
        fill_value=0
    ).reset_index()
    if "Resistant" in pivot.columns:
        pivot = pivot.sort_values("Resistant", ascending=False)
    return pivot


def meropenem_resistance_by_organism(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resistance rate of Meropenem broken down by organism.

    This is the core StrataMed Analytics metric: which bacteria are becoming
    resistant to our last-resort antibiotic?

    Returns DataFrame with organism, total tests, resistant count,
    and resistance rate (%).
    """
    mero = df[(df["antibiotic"] == "MEROPENEM") & (df["susceptibility"].notna())]
    result = (
        mero.groupby("organism")
        .agg(
            Total_Tests     = ("susceptibility", "count"),
            Resistant_Count = ("is_resistant", "sum"),
        )
        .assign(Resistance_Rate=lambda x: (x["Resistant_Count"] / x["Total_Tests"] * 100).round(2))
        .sort_values("Resistance_Rate", ascending=False)
        .reset_index()
    )
    return result


# ── Plotly figures ────────────────────────────────────────────────────────────

def fig_susceptibility_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Pie/donut chart of overall susceptibility distribution across all antibiotics.

    Provides a high-level view of how often antibiotics are working vs failing.
    """
    counts = df["susceptibility"].value_counts().reset_index()
    counts.columns = ["Susceptibility", "Count"]
    counts = counts[counts["Susceptibility"].isin(["Susceptible", "Intermediate", "Resistant"])]

    fig = px.pie(
        counts, values="Count", names="Susceptibility",
        color="Susceptibility",
        color_discrete_map=SUSC_COLOR_MAP,
        hole=0.45,
        title="Overall Susceptibility Distribution (All Antibiotics)",
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(
        showlegend=True,
        title_font_size=16,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def fig_meropenem_susceptibility(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of Meropenem susceptibility breakdown.

    The most important single chart for StrataMed Analytics:
    How often does Meropenem fail?
    """
    mero = df[(df["antibiotic"] == "MEROPENEM") & (df["susceptibility"].notna())]
    counts = mero["susceptibility"].value_counts().reset_index()
    counts.columns = ["Susceptibility", "Count"]

    fig = px.bar(
        counts, x="Susceptibility", y="Count",
        color="Susceptibility",
        color_discrete_map=SUSC_COLOR_MAP,
        title="Meropenem Susceptibility Results",
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        xaxis_title="Susceptibility",
        yaxis_title="Number of Records",
        title_font_size=16,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def fig_resistance_by_organism(df: pd.DataFrame, antibiotic: str = "MEROPENEM",
                                 top_n: int = 15) -> go.Figure:
    """
    Horizontal bar chart of resistance rate by organism for a given antibiotic.

    Sorted by resistance rate. Top N organisms shown for readability.
    Red bars signal high resistance — organisms where last-resort antibiotics fail.
    """
    sub = df[(df["antibiotic"] == antibiotic) & (df["susceptibility"].notna())]
    summary = (
        sub.groupby("organism")
        .agg(Total=("susceptibility", "count"), Resistant=("is_resistant", "sum"))
        .assign(Rate=lambda x: x["Resistant"] / x["Total"] * 100)
        .query("Total >= 5")   # only organisms with sufficient data
        .sort_values("Rate", ascending=True)
        .tail(top_n)
        .reset_index()
    )

    fig = px.bar(
        summary, y="organism", x="Rate",
        orientation="h",
        color="Rate",
        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
        range_color=[0, 100],
        title=f"{antibiotic.title()} Resistance Rate by Organism (Top {top_n})",
        labels={"Rate": "Resistance Rate (%)", "organism": "Organism"},
        text=summary["Rate"].apply(lambda x: f"{x:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=True,
        title_font_size=16,
        height=max(400, top_n * 35),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def fig_resistance_trend(df: pd.DataFrame, antibiotic: str = "MEROPENEM") -> go.Figure:
    """
    Line chart of resistance rate over time (by year) for a given antibiotic.

    This is the most critical diagnostic chart for StrataMed Analytics:
    Is resistance increasing, stable, or declining?
    A rising trend = growing market urgency for Venus Remedies.
    """
    sub = df[(df["antibiotic"] == antibiotic) & (df["susceptibility"].notna())]
    trend = (
        sub.groupby("order_year")
        .agg(Total=("susceptibility", "count"), Resistant=("is_resistant", "sum"))
        .assign(Rate=lambda x: x["Resistant"] / x["Total"] * 100)
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["order_year"], y=trend["Rate"],
        mode="lines+markers",
        name="Resistance Rate (%)",
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=8),
        fill="tozeroy",
        fillcolor="rgba(231,76,60,0.1)",
    ))
    fig.add_trace(go.Bar(
        x=trend["order_year"], y=trend["Total"],
        name="Total Tests",
        yaxis="y2",
        opacity=0.3,
        marker_color=COLORS["muted"],
    ))
    fig.update_layout(
        title=f"{antibiotic.title()} Resistance Rate Over Time (2008–2024)",
        xaxis_title="Year",
        yaxis=dict(title="Resistance Rate (%)", ticksuffix="%"),
        yaxis2=dict(title="Total Tests", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
        title_font_size=16,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
    )
    return fig


def fig_culture_type_breakdown(df: pd.DataFrame) -> go.Figure:
    """
    Stacked bar chart of susceptibility by culture type (Blood, Urine, Respiratory).

    Clinical context:
        Blood cultures (bacteremia) represent the most severe infections
        and are the primary indication for Meropenem. High resistance in
        blood cultures is the most alarming signal.
    """
    mero = df[(df["antibiotic"] == "MEROPENEM") & (df["susceptibility"].notna())]
    ct = (
        mero.groupby(["culture_description", "susceptibility"])
        .size()
        .reset_index(name="Count")
    )
    fig = px.bar(
        ct, x="culture_description", y="Count",
        color="susceptibility",
        color_discrete_map=SUSC_COLOR_MAP,
        barmode="stack",
        title="Meropenem Results by Culture Type",
        labels={"culture_description": "Culture Type", "Count": "Number of Records"},
    )
    fig.update_layout(
        title_font_size=16,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title="Susceptibility",
    )
    return fig


def fig_ward_resistance(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart of Meropenem resistance rate by ward type.

    Business insight:
        ICU typically shows highest resistance rates — confirming that
        the most vulnerable patients face the highest AMR burden.
        This helps Venus Remedies prioritise hospital-level interventions.
    """
    mero = df[(df["antibiotic"] == "MEROPENEM") & (df["susceptibility"].notna())]
    if "ward_type" not in mero.columns or mero["ward_type"].isna().all():
        return go.Figure().update_layout(title="Ward data not available")

    ward = (
        mero.groupby("ward_type")
        .agg(Total=("susceptibility", "count"), Resistant=("is_resistant", "sum"))
        .assign(Rate=lambda x: x["Resistant"] / x["Total"] * 100)
        .reset_index()
        .sort_values("Rate", ascending=False)
    )
    fig = px.bar(
        ward, x="ward_type", y="Rate",
        color="Rate",
        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
        range_color=[0, 100],
        title="Meropenem Resistance Rate by Ward Type",
        labels={"ward_type": "Ward Type", "Rate": "Resistance Rate (%)"},
        text=ward["Rate"].apply(lambda x: f"{x:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        title_font_size=16,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def fig_age_resistance(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of Meropenem resistance rate by patient age group.

    Business insight:
        Older patients typically show higher resistance rates due to more
        frequent prior antibiotic exposure and healthcare contact.
        This identifies demographic targeting for Venus Remedies stewardship.
    """
    mero = df[(df["antibiotic"] == "MEROPENEM") & (df["susceptibility"].notna())]
    if "age" not in mero.columns or mero["age"].isna().all():
        return go.Figure().update_layout(title="Age data not available")

    age = (
        mero.groupby("age", observed=True)
        .agg(Total=("susceptibility", "count"), Resistant=("is_resistant", "sum"))
        .assign(Rate=lambda x: x["Resistant"] / x["Total"] * 100)
        .reset_index()
    )
    fig = px.bar(
        age, x="age", y="Rate",
        color="Rate",
        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
        range_color=[0, 100],
        title="Meropenem Resistance Rate by Patient Age Group",
        labels={"age": "Age Group", "Rate": "Resistance Rate (%)"},
        text=age["Rate"].apply(lambda x: f"{x:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        title_font_size=16,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_tickangle=-30,
    )
    return fig


def fig_top_organisms_heatmap(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """
    Heatmap of susceptibility rates across top organisms × carbapenem antibiotics.

    Shows the full carbapenem resistance landscape — which organisms are
    resistant to which last-resort antibiotics. The most complex diagnostic
    chart in the analysis.
    """
    carbapenems = ["MEROPENEM", "ERTAPENEM", "IMIPENEM", "DORIPENEM"]
    sub = df[df["antibiotic"].isin(carbapenems) & df["susceptibility"].notna()]

    # Get top organisms by total test count
    top_orgs = (
        sub.groupby("organism")["susceptibility"].count()
        .sort_values(ascending=False)
        .head(top_n).index.tolist()
    )

    pivot = (
        sub[sub["organism"].isin(top_orgs)]
        .groupby(["organism", "antibiotic"])["is_resistant"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
        .pivot(index="organism", columns="antibiotic", values="is_resistant")
    )

    fig = px.imshow(
        pivot,
        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
        range_color=[0, 100],
        title=f"Carbapenem Resistance Rate Heatmap — Top {top_n} Organisms",
        labels=dict(color="Resistance Rate (%)"),
        aspect="auto",
        text_auto=True,
    )
    fig.update_layout(
        title_font_size=16,
        paper_bgcolor="white",
        height=max(400, top_n * 40),
        xaxis_title="Antibiotic",
        yaxis_title="Organism",
    )
    return fig


def fig_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """
    Correlation heatmap of numeric features.

    Used in diagnostic analysis to identify which variables correlate
    with Meropenem resistance. Guides feature selection for ML models.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Keep only meaningful numeric columns
    keep = [c for c in num_cols if c not in
            ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"]]
    corr = df[keep].corr().round(3)

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
        title="Correlation Matrix — Numeric Features",
        text_auto=True,
        aspect="auto",
    )
    fig.update_layout(
        title_font_size=16,
        paper_bgcolor="white",
        height=600,
    )
    return fig


def fig_resistance_timeline(resist_df: pd.DataFrame) -> go.Figure:
    """
    Histogram of days-to-resistance for all antibiotics in the resistance table.

    Shows the distribution of how long before/after a culture a patient
    had a prior resistance event. Negative = resistance confirmed BEFORE
    this culture (chronic carrier). Positive = new resistance.

    Args:
        resist_df: Raw microbial resistance table (not master)
    """
    fig = px.histogram(
        resist_df,
        x="resistant_time_to_culturetime",
        color_discrete_sequence=[COLORS["accent"]],
        nbins=60,
        title="Distribution of Resistance Timeline (Days to Culture Order)",
        labels={"resistant_time_to_culturetime": "Days to Culture (negative = prior resistance)"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="black",
                  annotation_text="Culture Order Date")
    fig.update_layout(
        title_font_size=16,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute top-level KPI metrics for the dashboard summary cards.

    Returns:
        dict with keys: total_records, unique_patients, unique_organisms,
                        meropenem_records, meropenem_resistance_rate,
                        overall_resistance_rate, year_range
    """
    mero = df[(df["antibiotic"] == "MEROPENEM") & (df["susceptibility"].notna())]
    all_tested = df[df["susceptibility"].notna()]

    return {
        "total_records":            len(df),
        "unique_patients":          df["anon_id"].nunique(),
        "unique_organisms":         df["organism"].nunique(),
        "antibiotics_tested":       df["antibiotic"].nunique(),
        "meropenem_records":        len(mero),
        "meropenem_resistance_rate": round(mero["is_resistant"].mean() * 100, 1)
                                      if len(mero) > 0 else 0,
        "overall_resistance_rate":  round(all_tested["is_resistant"].mean() * 100, 1),
        "year_range":               f"{int(df['order_year'].min())}–{int(df['order_year'].max())}"
                                     if "order_year" in df.columns else "N/A",
    }
