# 🧬 ResistAI — AMR Intelligence Platform

> **Antimicrobial Resistance Analytics for Venus Remedies Ltd.**  
> Powered by Stanford Healthcare ARMD Dataset (1999–2024)

---

## Overview

**ResistAI** is an AMR (Antimicrobial Resistance) intelligence platform built as an analytics partnership with **Venus Remedies Ltd.**, an Indian pharmaceutical company manufacturing Meropenem — a last-resort carbapenem antibiotic classified as essential medicine by the WHO.

This project analyzes 25 years of real-world clinical EHR data from Stanford Healthcare to answer one core business question:

> *"Which organisms, patient demographics, culture types, and clinical settings are driving Meropenem resistance — and how has that resistance evolved over time?"*

---

## Business Context

Antimicrobial resistance (AMR) is one of the most critical global health crises. Bacteria are evolving to resist antibiotics faster than new drugs are developed. Meropenem, as a last-resort antibiotic, is increasingly being used inappropriately — accelerating resistance and threatening its effectiveness precisely when it is needed most.

**ResistAI** helps Venus Remedies:
- Identify high-risk organisms and clinical settings driving Meropenem resistance
- Understand resistance trends over 25 years to inform long-term strategy
- Segment patient populations by AMR risk for stewardship program targeting
- Build the analytical foundation for predictive resistance modeling

---

## Analytics Roadmap

| Phase | Type | Status |
|---|---|---|
| EDA & Data Understanding | Descriptive | ✅ Complete |
| Resistance Drivers | Diagnostic | ✅ Complete |
| Resistance Prediction | Predictive | 🔄 Group Phase |
| Stewardship Recommendations | Prescriptive | 🔄 Group Phase |

---

## Dataset

**Antibiotic Resistance Microbiology Dataset (ARMD)**  
Stanford Healthcare, De-identified EHR data

| Attribute | Value |
|---|---|
| Patients | 283,715 unique |
| Culture Records | 751,075 |
| Time Period | 1999 – 2024 |
| Antibiotics Tested | 55 (including Meropenem) |
| Source | Stanford Healthcare via Dryad |
| License | CC0 (Public Domain) |

**Citation:**  
Nateghi Haredasht, F., et al. *Antibiotic Resistance Microbiology Dataset (ARMD).* Stanford Healthcare, 2025. [arXiv:2503.07664](https://doi.org/10.48550/arXiv.2503.07664)

---

## Project Structure

```
resistai-amr-analytics/
│
├── data/
│   ├── raw/                          ← Place downloaded ARMD CSV files here
│   └── processed/                    ← Auto-generated cleaned datasets
│
├── notebooks/
│   └── eda_analysis.ipynb            ← Main EDA notebook (start here)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                ← Table loading & merging functions
│   ├── cleaning.py                   ← Cleaning & transformation pipeline
│   └── analysis.py                   ← Reusable analytical functions & Plotly figures
│
├── dashboard/
│   ├── app.py                        ← Streamlit dashboard
│   └── index.html                    ← Standalone HTML backup dashboard
│
├── reports/
│   └── figures/                      ← Exported chart images
│
├── .streamlit/
│   └── config.toml                   ← Streamlit theme configuration
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/resistai-amr-analytics.git
cd resistai-amr-analytics
```

### 2. Create virtual environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Download data
Download the following files from [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.jq2bvq8kp) and place them in `data/raw/`:

- `microbiology_cultures_cohort.csv`
- `microbiology_cultures_demographics.csv`
- `microbiology_cultures_ward_info.csv`
- `microbiology_cultures_microbial_resistance.csv`
- `microbiology_cultures_adi_scores.csv`
- `microbiology_cultures_nursing_home_visits.csv`
- `implied_susceptibility_rules.csv`

### 4. Run the notebook
```bash
jupyter notebook notebooks/eda_analysis.ipynb
```

### 5. Launch Streamlit dashboard
```bash
streamlit run dashboard/app.py
```

### 6. Open HTML backup dashboard
Simply open `dashboard/index.html` in any browser — no server required.

---

## Key Findings

1. **Meropenem resistance** is present across multiple organism types — with certain ESKAPE pathogens showing critically high resistance rates
2. **ICU settings** consistently show the highest resistance burden — confirming the most vulnerable patients face the greatest AMR risk
3. **Blood cultures** with Meropenem resistance signal systemic sepsis — the most critical clinical and commercial use case
4. **Temporal trends** reveal how resistance has evolved across 25 years — informing Venus Remedies' long-term stewardship strategy
5. **Nursing home patients** represent a high-risk, high-priority population for targeted intervention

---

## Tech Stack

| Component | Technology |
|---|---|
| Data Processing | Python, pandas, numpy |
| Visualisation | Plotly, matplotlib, seaborn |
| Interactive Dashboard | Streamlit |
| Backup Dashboard | HTML, CSS, Chart.js |
| Notebook | Jupyter |
| Version Control | Git + GitHub |

---

## Academic Context

**Course:** Data Analytics (MGB) — Core Course  
**Program:** Masters in Finance  
**Component:** Individual Submission (50 marks)  
**Analysis Scope:** Descriptive & Diagnostic Analytics  

---

## License

MIT License — see `LICENSE` for details.

Data source (ARMD) is published under CC0 (Public Domain) by Stanford Healthcare.
