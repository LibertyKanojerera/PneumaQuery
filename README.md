# PneumaQuery — AI-Powered Digital Twin Monitoring Platform

A prototype AI system for real-time monitoring of transplanted lung patients, 
built for a leading bioengineering company specialising in 3D-printed 
transplant lungs, as part of a Digital Transformation course project at American University's Kogod School of Business.

## Overview

PneumaQuery creates a digital twin of each transplanted lung, continuously 
monitoring patient vitals and predicting rejection risk before symptoms appear. 
The system transforms post-transplant care from reactive clinic visits into 
continuous, data-driven monitoring.

This digital twin serves two audiences:
- **Clinicians** — real-time risk scores and alerts for early intervention
- **Manufacturers** — anonymized outcome data to help the company improve 
  future lung designs and materials

> Note: This prototype was built on synthetic patient data generated to simulate 
> realistic clinical scenarios for a 50-patient transplant cohort.

## Clinical Vitals Monitored

| Vital | Source | Normal Range |
|---|---|---|
| Oxygen Level | Implant sensor | 95–99% |
| Breathing Rate | Wearable | 12–20 breaths/min |
| Inflammation Score | Blood marker | 0–4 |
| Blood Pressure (Systolic) | Wearable | 90–120 mmHg |
| Cough Frequency | Wearable | 0–5 coughs/hr |
| Activity Level | Wearable | 7–10/10 |
| Mechanical Strain | Implant sensor | 0–3/10 |

## Project Structure

| File | Description |
|---|---|
| `train_model.py` | Trains Random Forest model with 5-fold cross-validation |
| `predict.py` | Scores a batch of new patients |
| `live_predict.py` | Interactive live risk predictor |
| `dashboard.py` | Full visual clinical dashboard with 6 panels |
| `pneumaquery_model.pkl` | Trained and saved ML model |

## Model Performance

| Metric | Result |
|---|---|
| Test Set Accuracy | 90% |
| Cross-Validated Accuracy (5-Fold) | 62% |
| High Risk patients misclassified as Low | 0 |

The gap between test and CV accuracy is expected with a 50-patient prototype 
dataset. In a real deployment with thousands of patients, CV accuracy would 
converge toward test accuracy. Critically, the model makes only clinically 
safe errors — it never misses a High Risk patient.

## How To Run

```bash
# Install dependencies
pip install pandas scikit-learn matplotlib seaborn

# Train the model
python train_model.py

# Run live predictor
python live_predict.py

# View dashboard
python dashboard.py
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML Model | Random Forest (scikit-learn) |
| Data Processing | pandas |
| Visualization | matplotlib |
| CRM | Salesforce (custom Patient object + dashboards) |

## CRM Integration

Patient records, risk scores and clinical reports are managed in Salesforce 
with a custom Patient object containing 12 fields. Two key reports power 
the clinical workflow:

- **High Risk Patient Watch List** — 8 flagged patients sorted by risk score
- **Lung Model Performance Report** — average risk score by lung model for 
  manufacturer feedback

## Customer Journey

1. **Patient Selection** — lung is 3D-printed and matched to patient anatomy
2. **Digital Birth** — lung implanted, digital twin activated in Salesforce
3. **Silent Monitoring** — wearables and implant sensors stream vitals 24/7
4. **The Signal** — risk score rises as vitals deteriorate
5. **The Alert** — clinician receives immediate notification
6. **Intervention** — treatment adjusted before rejection crisis occurs
7. **Feedback Loop** — outcomes logged to improve future lung designs

## Project Context

Built as part of the Digital Transformation course at Kogod School of Business,
American University — Spring 2026.

**Client:**  Bioengineering company specialising in 3D-printed transplant lungs 
**Team:** Team Black  
**Scope:** Prototype demonstrating AI-powered predictive monitoring 
integrated with Salesforce CRM
