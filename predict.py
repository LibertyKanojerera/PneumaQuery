import pickle
import pandas as pd

with open("pneumaquery_model.pkl", "rb") as f:
    model = pickle.load(f)

print("PneumaQuery Risk Predictor")
print("=" * 50)

new_patients = [
    {
        "patient_id":               "P051",
        "patient_name":             "Alice Monroe",
        "lung_model":               "LungTech-A1",
        "days_since_transplant":    45,
        "oxygen_level":             87.5,
        "breathing_rate":           27,
        "inflammation_score":       8.2,
        "blood_pressure_systolic":  148,
        "cough_frequency":          15,
        "activity_level":           2.1,
        "mechanical_strain":        8.7,
    },
    {
        "patient_id":               "P052",
        "patient_name":             "James Okafor",
        "lung_model":               "BioLung-X3",
        "days_since_transplant":    200,
        "oxygen_level":             97.2,
        "breathing_rate":           14,
        "inflammation_score":       1.3,
        "blood_pressure_systolic":  112,
        "cough_frequency":          2,
        "activity_level":           8.5,
        "mechanical_strain":        1.2,
    },
    {
        "patient_id":               "P053",
        "patient_name":             "Maria Santos",
        "lung_model":               "LungTech-B2",
        "days_since_transplant":    120,
        "oxygen_level":             93.1,
        "breathing_rate":           22,
        "inflammation_score":       5.8,
        "blood_pressure_systolic":  135,
        "cough_frequency":          8,
        "activity_level":           5.2,
        "mechanical_strain":        5.1,
    },
]

features = [
    "days_since_transplant",
    "oxygen_level",
    "breathing_rate",
    "inflammation_score",
    "blood_pressure_systolic",
    "cough_frequency",
    "activity_level",
    "mechanical_strain"
]

df_new = pd.DataFrame(new_patients)
predictions  = model.predict(df_new[features])
probabilities = model.predict_proba(df_new[features])

print()
for i, row in df_new.iterrows():
    risk  = predictions[i]
    probs = probabilities[i]
    classes = model.classes_

    prob_str = "  |  ".join(
        f"{cls}: {prb:.0%}" for cls, prb in zip(classes, probs)
    )

    if risk == "High":
        alert  = "🔴 HIGH RISK — Immediate clinical review recommended"
        border = "!" * 50
    elif risk == "Medium":
        alert  = "🟡 MEDIUM RISK — Schedule follow-up within 48 hours"
        border = "-" * 50
    else:
        alert  = "🟢 LOW RISK — Patient is stable"
        border = "-" * 50

    print(f"{border}")
    print(f"  PNEUMAQUERY RISK ASSESSMENT")
    print(border)
    print(f"  Patient:          {row['patient_name']} ({row['patient_id']})")
    print(f"  Lung Model:       {row['lung_model']}")
    print(f"  Days Post-Op:     {row['days_since_transplant']}")
    print()
    print(f"  Vitals:")
    print(f"    Oxygen Level:          {row['oxygen_level']}%")
    print(f"    Breathing Rate:        {row['breathing_rate']} breaths/min")
    print(f"    Inflammation Score:    {row['inflammation_score']}")
    print(f"    Blood Pressure:        {row['blood_pressure_systolic']} mmHg")
    print(f"    Cough Frequency:       {row['cough_frequency']} coughs/hr")
    print(f"    Activity Level:        {row['activity_level']}/10")
    print(f"    Mechanical Strain:     {row['mechanical_strain']}/10")
    print()
    print(f"  {alert}")
    print()
    print(f"  Confidence breakdown:")
    print(f"    {prob_str}")
    print(border)
    print()
    