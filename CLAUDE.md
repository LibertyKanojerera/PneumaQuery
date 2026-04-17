# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PneumaQuery is an AI-powered digital twin monitoring platform for transplanted lung patients. It predicts rejection risk in real-time by analyzing 8 clinical vitals. This is a prototype built for American University's Kogod School of Business (Spring 2026, Team Black) for a bioengineering client specializing in 3D-printed transplant lungs.

All data is synthetic (50-patient simulation dataset).

## Running the Project

There is no build system. Install dependencies manually:

```bash
pip install pandas scikit-learn matplotlib seaborn simple_salesforce python-dotenv requests
```

Run each script directly (in order for first-time setup):

```bash
python train_model.py        # Train and save the Random Forest model → pneumaquery_model.pkl
python predict.py            # Batch-score 3 hardcoded patients
python live_predict.py       # Interactive CLI predictor for clinician use
python dashboard.py          # Generate 6-panel visualization → pneumaquery_dashboard.png
python salesforce_connect.py # Push patient predictions to Salesforce (requires .env)
```

### Salesforce Integration

Requires a `.env` file (not committed) with:
```
SF_CONSUMER_KEY=...
SF_CONSUMER_SECRET=...
SF_USERNAME=...
SF_PASSWORD=...
SF_SECURITY_TOKEN=...
```

## Architecture

The pipeline is linear and stateless — each script is independent and runs top-to-bottom with no class abstractions:

```
patients.csv (50 patients, 13 columns)
    └─► train_model.py → pneumaquery_model.pkl (RandomForest, 100 trees, 8 features, 3 risk classes)
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         predict.py      live_predict.py   dashboard.py
         (batch, 3        (interactive      (6-panel PNG,
          hardcoded)       CLI w/ input      all 50 patients)
                           validation)
                                            salesforce_connect.py
                                            (OAuth2 → Patient__c records)
```

**ML Model:**
- Algorithm: `RandomForestClassifier(n_estimators=100)`
- Features: oxygen level, breathing rate, inflammation score, blood pressure systolic, cough frequency, activity level, mechanical strain, days since transplant
- Target: 3-class label — High / Medium / Low risk
- Train/test split: 80/20 stratified; 5-fold cross-validation
- Safety invariant: **zero High Risk patients are ever classified as Low Risk**

**Clinical data schema** (`patients.csv`): `patient_id`, `patient_name`, `lung_model` (BioLung-X3 / LungTech-A1 / LungTech-B2), `days_since_transplant`, `oxygen_level`, `breathing_rate`, `inflammation_score`, `blood_pressure_systolic`, `cough_frequency`, `activity_level`, `mechanical_strain`, `risk_score`, `risk_label`

**Salesforce custom object:** `Patient__c` with 12 fields mapped from the CSV + model predictions (includes `Alert_Triggered__c` for High Risk patients).

## Key Constraints

- `patients.csv` and `generate_data.py` are excluded from git (see `.gitignore`) — the CSV must exist locally to run any script
- `pneumaquery_model.pkl` must be trained before running `predict.py`, `live_predict.py`, `dashboard.py`, or `salesforce_connect.py`
- No test suite exists — this is a prototype
- Color coding throughout: High Risk = `#E74C3C` (red), Medium = `#F39C12` (orange), Low = `#27AE60` (green)
