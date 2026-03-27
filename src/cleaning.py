"""
ResistAI — AMR Intelligence Platform
=====================================
src/cleaning.py

Data cleaning and transformation pipeline for the ARMD dataset.
Every transformation is documented with the clinical and business
rationale for making it.

Author: ResistAI Analytics
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT_DIR / "data" / "processed"


# ── Susceptibility standardisation ───────────────────────────────────────────

# Mapping raw susceptibility strings to clean standard labels
SUSCEPTIBILITY_MAP = {
    "Susceptible":    "Susceptible",
    "Resistant":      "Resistant",
    "Intermediate":   "Intermediate",
    "Inconclusive":   np.nan,   # analytically unusable — treated as missing
    "Null":           np.nan,
    "null":           np.nan,
    "":               np.nan,
}

# Ordered category for resistance severity (used in plots and sorting)
SUSCEPTIBILITY_ORDER = ["Susceptible", "Intermediate", "Resistant"]


# ── Age bin ordering ──────────────────────────────────────────────────────────

AGE_ORDER = [
    "18-24 years", "25-34 years", "35-44 years",
    "45-54 years", "55-64 years", "65-74 years",
    "75-84 years", "85-89 years", "90+ years",
]


def clean_master(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Full cleaning and transformation pipeline for the master dataset.

    Steps:
        1.  Drop exact duplicate rows
        2.  Standardise 'Null' string → actual NaN across all object columns
        3.  Clean and standardise susceptibility values
        4.  Create binary resistance flag (is_resistant)
        5.  Create Meropenem-specific flag (is_meropenem)
        6.  Derive ward_type as a single categorical column from binary flags
        7.  Clean age column → ordered categorical
        8.  Extract temporal features from order_time_jittered_utc
        9.  Standardise organism and antibiotic names (uppercase strip)
        10. Convert ADI score to numeric
        11. Drop columns with >80% missing values
        12. Reset index

    Args:
        df:   Raw master dataset from data_loader.load_master_dataset()
        save: If True, saves cleaned CSV to data/processed/

    Returns:
        pd.DataFrame: Cleaned and feature-engineered dataset
    """
    print("━" * 55)
    print("  ResistAI — Data Cleaning Pipeline")
    print("━" * 55)
    original_shape = df.shape

    # ── Step 1: Drop exact duplicates ────────────────────────────────────────
    df = df.drop_duplicates()
    print(f"[1] Dropped exact duplicates: {original_shape[0] - df.shape[0]:,} rows removed")

    # ── Step 2: Replace string 'Null' / 'null' with NaN ──────────────────────
    # The dataset uses the string 'Null' for missing values instead of
    # actual NaN — we need to convert these for proper pandas handling
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].replace({"Null": np.nan, "null": np.nan, "": np.nan})
    print(f"[2] Standardised 'Null' strings to NaN across {len(obj_cols)} text columns")

    # ── Step 3: Standardise susceptibility ───────────────────────────────────
    # Raw values include: Susceptible, Resistant, Intermediate, Inconclusive, Null
    # 'Inconclusive' has no analytical value and is mapped to NaN
    df["susceptibility"] = df["susceptibility"].map(SUSCEPTIBILITY_MAP)
    df["susceptibility"] = pd.Categorical(
        df["susceptibility"], categories=SUSCEPTIBILITY_ORDER, ordered=True
    )
    print(f"[3] Susceptibility cleaned — distribution:\n"
          f"    {df['susceptibility'].value_counts().to_dict()}")

    # ── Step 4: Binary resistance flag ───────────────────────────────────────
    # is_resistant = 1 only for confirmed Resistant results.
    # This is our primary target variable for EDA and later ML models.
    df["is_resistant"] = (df["susceptibility"] == "Resistant").astype(int)
    print(f"[4] Created is_resistant flag — {df['is_resistant'].sum():,} resistant records "
          f"({df['is_resistant'].mean()*100:.1f}%)")

    # ── Step 5: Meropenem flag ───────────────────────────────────────────────
    # Isolates Meropenem records for focused analysis.
    # Meropenem is our primary antibiotic of interest for Venus Remedies.
    df["is_meropenem"] = (df["antibiotic"] == "Meropenem").astype(int)
    mero_count = df["is_meropenem"].sum()
    mero_resist = df[df["is_meropenem"] == 1]["is_resistant"].sum()
    print(f"[5] Meropenem records: {mero_count:,} total | "
          f"{mero_resist} resistant ({mero_resist/mero_count*100:.1f}% resistance rate)")

    # ── Step 6: Derive single ward_type column ───────────────────────────────
    # The raw data has 4 binary columns (IP, OP, ER, ICU).
    # We consolidate into one readable categorical column.
    # Priority: ICU > ER > IP > OP (most to least acute)
    def resolve_ward(row):
        if row.get("hosp_ward_ICU") == 1: return "ICU"
        if row.get("hosp_ward_ER")  == 1: return "ER"
        if row.get("hosp_ward_IP")  == 1: return "Inpatient"
        if row.get("hosp_ward_OP")  == 1: return "Outpatient"
        return np.nan

    ward_cols = ["hosp_ward_ICU", "hosp_ward_ER", "hosp_ward_IP", "hosp_ward_OP"]
    if all(c in df.columns for c in ward_cols):
        df["ward_type"] = df[ward_cols].apply(resolve_ward, axis=1)
        print(f"[6] Derived ward_type: {df['ward_type'].value_counts().to_dict()}")
    else:
        print(f"[6] Ward columns not found — skipping ward_type derivation")

    # ── Step 7: Age as ordered categorical ───────────────────────────────────
    # Converts string age bins to an ordered categorical type so that
    # plots and groupbys respect the natural age ordering.
    if "age" in df.columns:
        valid_ages = [a for a in AGE_ORDER if a in df["age"].dropna().unique()]
        df["age"] = pd.Categorical(df["age"], categories=valid_ages, ordered=True)
        print(f"[7] Age bins ordered: {valid_ages}")

    # ── Step 8: Temporal feature extraction ──────────────────────────────────
    # Extracts analytically useful date/time components from the culture
    # timestamp. Year is essential for resistance trend analysis (2008–2024).
    if "order_time_jittered_utc" in df.columns:
        dt = df["order_time_jittered_utc"]
        df["order_year"]    = dt.dt.year
        df["order_month"]   = dt.dt.month
        df["order_quarter"] = dt.dt.quarter
        df["order_dow"]     = dt.dt.day_name()   # day of week
        print(f"[8] Temporal features extracted — year range: "
              f"{int(df['order_year'].min())}–{int(df['order_year'].max())}")

    # ── Step 9: Standardise organism and antibiotic names ────────────────────
    # Strips whitespace and uppercases organism names for consistent grouping.
    # Some entries have trailing spaces or mixed case variants.
    for col in ["organism", "antibiotic", "culture_description"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()
    print(f"[9] Organism/antibiotic/culture names standardised to uppercase")

    # ── Step 10: ADI score to numeric ────────────────────────────────────────
    if "adi_score" in df.columns:
        df["adi_score"] = pd.to_numeric(df["adi_score"], errors="coerce")
        df["adi_state_rank"] = pd.to_numeric(df["adi_state_rank"], errors="coerce")
        print(f"[10] ADI scores converted to numeric — "
              f"{df['adi_score'].notna().sum():,} valid values")

    # ── Step 11: Drop high-missingness columns ────────────────────────────────
    # Columns with more than 80% missing values add noise without analytical value.
    thresh = 0.80
    missing_pct = df.isnull().mean()
    drop_cols = missing_pct[missing_pct > thresh].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"[11] Dropped {len(drop_cols)} columns with >{thresh*100:.0f}% missingness: "
              f"{drop_cols}")
    else:
        print(f"[11] No columns exceeded {thresh*100:.0f}% missingness threshold")

    # ── Step 12: Reset index ─────────────────────────────────────────────────
    df = df.reset_index(drop=True)

    print("━" * 55)
    print(f"  Cleaning complete: {original_shape} → {df.shape}")
    print(f"  Rows retained: {df.shape[0]/original_shape[0]*100:.1f}%")
    print("━" * 55)

    # ── Save ─────────────────────────────────────────────────────────────────
    if save:
        out_path = PROC_DIR / "master_cleaned.csv"
        df.to_csv(out_path, index=False)
        print(f"\n✅ Cleaned dataset saved → {out_path}")

    return df


def get_meropenem_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the master dataset to Meropenem records only.

    This is the primary analytical slice for ResistAI's core business question:
    'Where and in whom is Meropenem failing?'

    Args:
        df: Cleaned master dataset

    Returns:
        pd.DataFrame: Meropenem-only records with susceptibility results
    """
    mero = df[df["antibiotic"] == "MEROPENEM"].copy()
    mero = mero[mero["susceptibility"].notna()]
    return mero


def get_carbapenem_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to all carbapenem-class antibiotics.

    Carbapenems are the last-resort antibiotic class. Meropenem is the
    most commonly used, but Imipenem, Ertapenem, and Doripenem are also
    in this class. Analysing all carbapenems together gives a complete
    picture of last-resort antibiotic resistance.

    Carbapenem members in this dataset:
        Meropenem, Ertapenem, Imipenem, Doripenem

    Args:
        df: Cleaned master dataset

    Returns:
        pd.DataFrame: All carbapenem records
    """
    carbapenems = ["MEROPENEM", "ERTAPENEM", "IMIPENEM", "DORIPENEM"]
    return df[df["antibiotic"].isin(carbapenems)].copy()
