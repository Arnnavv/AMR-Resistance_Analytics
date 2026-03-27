"""
StrataMed Analytics — AMR Intelligence Platform
=====================================
src/data_loader.py

Handles loading and merging of all ARMD (Antibiotic Resistance Microbiology Dataset)
tables from Stanford Healthcare EHR data. Each function is documented with the
clinical and business context of what it loads.

Author: StrataMed Analytics Analytics
Dataset: ARMD — Stanford Healthcare (Dryad, 2025)
Citation: Nateghi Haredasht, F., et al. arXiv:2503.07664
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


# ── Path resolution ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT_DIR / "data" / "raw"
PROC_DIR = ROOT_DIR / "data" / "processed"

# Ensure processed directory exists
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ── Individual table loaders ──────────────────────────────────────────────────

def load_cohort(path: str = None) -> pd.DataFrame:
    """
    Load the primary cohort table.

    This is the backbone of the entire dataset. Each row represents a unique
    (patient, culture order, antibiotic) combination — i.e., the result of
    testing one antibiotic against one organism found in one culture.

    Clinical context:
        When a patient presents with an infection, clinicians collect a sample
        (blood, urine, respiratory secretion) and send it to the microbiology lab.
        The lab identifies what organism is growing and tests it against a panel
        of antibiotics. The result for each antibiotic is S (Susceptible),
        I (Intermediate), or R (Resistant).

    Business context (StrataMed Analytics):
        Meropenem records where susceptibility = 'Resistant' are our primary
        signal — these are the cases where last-resort antibiotics are failing.

    Columns:
        anon_id                  — De-identified patient ID
        pat_enc_csn_id_coded     — Encounter ID (links to one hospital visit)
        order_proc_id_coded      — Unique culture order ID (primary join key)
        order_time_jittered_utc  — Culture collection timestamp (jittered ±2 weeks)
        ordering_mode            — Inpatient / Outpatient / Null
        culture_description      — Sample type: BLOOD, URINE, RESPIRATORY
        was_positive             — 1 if culture confirmed an organism, 0 otherwise
        organism                 — Bacteria identified (e.g. KLEBSIELLA PNEUMONIAE)
        antibiotic               — Antibiotic tested (e.g. Meropenem)
        susceptibility           — S / I / R / Inconclusive / Null
    """
    fpath = path or RAW_DIR / "microbiology_cultures_cohort.csv"
    df = pd.read_csv(fpath, low_memory=False)
    df["order_time_jittered_utc"] = pd.to_datetime(
        df["order_time_jittered_utc"], utc=True, errors="coerce"
    )
    return df


def load_demographics(path: str = None) -> pd.DataFrame:
    """
    Load patient demographic information.

    Ages are binned (e.g. '25-34 years') for HIPAA compliance. Gender is
    encoded as binary (0/1) without specification of which value maps to
    which gender — intentional de-identification by the dataset authors.

    Business context:
        Allows StrataMed Analytics to segment resistance burden by patient age group,
        critical for understanding which demographics face the highest AMR risk.

    Columns:
        anon_id              — Patient ID (join key)
        order_proc_id_coded  — Culture order ID (join key)
        age                  — Age bin: '18-24 years', '25-34 years', etc.
        gender               — Binary encoded: 0 or 1 (or Null)
    """
    fpath = path or RAW_DIR / "microbiology_cultures_demographics.csv"
    df = pd.read_csv(fpath, low_memory=False)
    # Replace string 'Null' with actual NaN for proper handling
    df.replace("Null", np.nan, inplace=True)
    return df


def load_ward_info(path: str = None) -> pd.DataFrame:
    """
    Load ward/setting information for each culture order.

    Each culture is associated with one care setting encoded as binary flags.
    A patient can only be in one setting at a time, so exactly one flag = 1.

    Clinical context:
        ICU patients are critically ill and often have the highest rates of
        carbapenem-resistant organisms. ER patients represent acute presentations.
        Outpatient cultures often reflect community-acquired infections.

    Business context:
        ICU and Inpatient settings are the primary markets for Meropenem.
        Venus Remedies should prioritize hospital-level stewardship programs
        targeting these settings.

    Columns:
        hosp_ward_IP   — 1 if Inpatient ward
        hosp_ward_OP   — 1 if Outpatient setting
        hosp_ward_ER   — 1 if Emergency Room
        hosp_ward_ICU  — 1 if Intensive Care Unit
    """
    fpath = path or RAW_DIR / "microbiology_cultures_ward_info.csv"
    df = pd.read_csv(fpath, low_memory=False)
    df["order_time_jittered_utc"] = pd.to_datetime(
        df["order_time_jittered_utc"], utc=True, errors="coerce"
    )
    return df


def load_microbial_resistance(path: str = None) -> pd.DataFrame:
    """
    Load the resistance timeline table.

    Each row records how many days before or after the current culture order
    a patient had a PRIOR confirmed resistance event for a given organism-
    antibiotic pair. A negative value means resistance was confirmed BEFORE
    the culture; positive means AFTER.

    Clinical context:
        This table reveals resistance chronology — e.g., a patient who was
        resistant to Meropenem 300 days before a new culture is a 'chronic
        carrier' of resistant organisms.

    Business context:
        Patients with prior Meropenem resistance represent the highest-risk
        segment — and a core focus for StrataMed Analytics's early warning system.

    Columns:
        organism                       — Bacteria
        antibiotic                     — Antibiotic (including Meropenem)
        resistant_time_to_culturetime  — Days from prior resistance to current culture
    """
    fpath = path or RAW_DIR / "microbiology_cultures_microbial_resistance.csv"
    df = pd.read_csv(fpath, low_memory=False)
    df["order_time_jittered_utc"] = pd.to_datetime(
        df["order_time_jittered_utc"], utc=True, errors="coerce"
    )
    return df


def load_adi_scores(path: str = None) -> pd.DataFrame:
    """
    Load Area Deprivation Index (ADI) scores.

    The ADI measures socioeconomic disadvantage at the ZIP code level.
    Higher ADI score = greater deprivation. This is a proxy for patient
    socioeconomic status since individual income is not recorded.

    Business context:
        Socioeconomic deprivation correlates with higher AMR burden due to
        limited access to care, overuse of antibiotics in lower-income settings,
        and delayed treatment. For Venus Remedies, ADI-stratified analysis
        reveals underserved market segments with high unmet need.

    Columns:
        adi_score       — Raw ADI score (higher = more deprived)
        adi_state_rank  — State-level percentile rank
    """
    fpath = path or RAW_DIR / "microbiology_cultures_adi_scores.csv"
    df = pd.read_csv(fpath, low_memory=False)
    df.replace("Null", np.nan, inplace=True)
    return df


def load_nursing_home(path: str = None) -> pd.DataFrame:
    """
    Load nursing home visit history.

    Records whether a patient had a nursing home visit relative to the
    culture order. Value = number of days between nursing home visit and
    culture (0 = same day).

    Clinical context:
        Nursing home patients are a high-risk population for carbapenem-
        resistant infections due to shared environments, frequent antibiotic
        exposure, and compromised immune status.

    Business context:
        Nursing homes represent a key channel for Venus Remedies — facilities
        with high AMR burden need formulary access to Meropenem.

    Columns:
        nursing_home_visit_culture — Days between nursing home visit & culture order
    """
    fpath = path or RAW_DIR / "microbiology_cultures_nursing_home_visits.csv"
    df = pd.read_csv(fpath, low_memory=False)
    df["order_time_jittered_utc"] = pd.to_datetime(
        df["order_time_jittered_utc"], utc=True, errors="coerce"
    )
    return df


def load_implied_susceptibility(path: str = None) -> pd.DataFrame:
    """
    Load the implied susceptibility rules reference table.

    Defines logic rules for inferring antibiotic susceptibility based on
    known cross-resistance patterns. Used during data cleaning to fill
    in susceptibility values that were not directly tested.

    Example rule: 'Susceptible if susceptible to Imipenem' means that if
    a Meropenem result is missing but Imipenem is Susceptible, we can
    infer Meropenem Susceptibility.

    Columns:
        Organism    — Organism group
        Antibiotic  — Antibiotic
        Rule        — Inference logic
    """
    fpath = path or RAW_DIR / "implied_susceptibility_rules.csv"
    return pd.read_csv(fpath, low_memory=False)


# ── Master merge function ─────────────────────────────────────────────────────

def load_master_dataset(
    cohort_path=None,
    demo_path=None,
    ward_path=None,
    resist_path=None,
    adi_path=None,
    nursing_path=None,
) -> pd.DataFrame:
    """
    Load and merge all tables into one master analytical dataset.

    Join strategy:
        - Primary key: order_proc_id_coded (unique per culture order)
        - All joins are LEFT joins on cohort — we never drop cohort records
        - Ward info joined on order_proc_id_coded
        - Demographics joined on order_proc_id_coded
        - ADI scores joined on order_proc_id_coded
        - Nursing home joined on order_proc_id_coded

    The resistance table is kept separate because it is a 'timeline' table
    (multiple rows per culture order) and is used independently for timeline
    analysis rather than merged into the flat master dataset.

    Returns:
        pd.DataFrame: Merged master dataset ready for cleaning & EDA
    """
    print("[ 1/5 ] Loading cohort table...")
    cohort = load_cohort(cohort_path)

    print("[ 2/5 ] Loading demographics table...")
    demo = load_demographics(demo_path)
    demo_cols = ["order_proc_id_coded", "age", "gender"]
    cohort = cohort.merge(demo[demo_cols], on="order_proc_id_coded", how="left")

    print("[ 3/5 ] Loading ward info table...")
    ward = load_ward_info(ward_path)
    ward_cols = ["order_proc_id_coded", "hosp_ward_IP", "hosp_ward_OP",
                 "hosp_ward_ER", "hosp_ward_ICU"]
    cohort = cohort.merge(ward[ward_cols], on="order_proc_id_coded", how="left")

    print("[ 4/5 ] Loading ADI scores table...")
    adi = load_adi_scores(adi_path)
    adi_cols = ["order_proc_id_coded", "adi_score", "adi_state_rank"]
    cohort = cohort.merge(adi[adi_cols], on="order_proc_id_coded", how="left")

    print("[ 5/5 ] Loading nursing home visits table...")
    nursing = load_nursing_home(nursing_path)
    nursing_cols = ["order_proc_id_coded", "nursing_home_visit_culture"]
    cohort = cohort.merge(nursing[nursing_cols], on="order_proc_id_coded", how="left")

    print(f"\n✅ Master dataset loaded: {cohort.shape[0]:,} rows × {cohort.shape[1]} columns")
    return cohort
