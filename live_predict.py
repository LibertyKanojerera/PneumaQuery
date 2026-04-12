import pickle
import pandas as pd

with open("pneumaquery_model.pkl", "rb") as f:
    model = pickle.load(f)

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

def get_input(prompt, min_val, max_val, decimal=False):
    while True:
        try:
            value = float(input(prompt)) if decimal else int(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"  Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  Invalid input — please enter a number.")

def predict_patient(name, lung_model, days, oxygen, breathing,
                    inflammation, bp, cough, activity, strain):
    patient_df = pd.DataFrame([{
        "days_since_transplant":   days,
        "oxygen_level":            oxygen,
        "breathing_rate":          breathing,
        "inflammation_score":      inflammation,
        "blood_pressure_systolic": bp,
        "cough_frequency":         cough,
        "activity_level":          activity,
        "mechanical_strain":       strain
    }])

    prediction    = model.predict(patient_df[features])[0]
    probabilities = model.predict_proba(patient_df[features])[0]
    classes       = model.classes_

    prob_str = "  |  ".join(
        f"{cls}: {prb:.0%}" for cls, prb in zip(classes, probabilities)
    )

    if prediction == "High":
        alert  = "🔴 HIGH RISK — Immediate clinical review recommended"
        border = "!" * 55
    elif prediction == "Medium":
        alert  = "🟡 MEDIUM RISK — Schedule follow-up within 48 hours"
        border = "-" * 55
    else:
        alert  = "🟢 LOW RISK — Patient is stable"
        border = "-" * 55

    print(f"\n{border}")
    print(f"  PNEUMAQUERY RISK ASSESSMENT")
    print(border)
    print(f"  Patient:          {name}")
    print(f"  Lung Model:       {lung_model}")
    print(f"  Days Post-Op:     {days}")
    print()
    print(f"  Vitals:")
    print(f"    Oxygen Level:          {oxygen}%")
    print(f"    Breathing Rate:        {breathing} breaths/min")
    print(f"    Inflammation Score:    {inflammation}/10")
    print(f"    Blood Pressure:        {bp} mmHg")
    print(f"    Cough Frequency:       {cough} coughs/hr")
    print(f"    Activity Level:        {activity}/10")
    print(f"    Mechanical Strain:     {strain}/10")
    print()
    print(f"  {alert}")
    print()
    print(f"  Confidence breakdown:")
    print(f"    {prob_str}")
    print(border)

def main():
    print("\n" + "=" * 55)
    print("  PNEUMAQUERY — Live Patient Risk Predictor")
    print("=" * 55)
    print("  Enter patient vitals to get an instant")
    print("  rejection risk assessment.")
    print("=" * 55)

    lung_models = ["LungTech-A1", "LungTech-B2", "BioLung-X3"]

    while True:
        print("\n--- New Patient Entry ---")

        name = input("Patient name: ").strip()
        if not name:
            name = "Unknown Patient"

        print("Lung model:")
        for i, lm in enumerate(lung_models, 1):
            print(f"  {i}. {lm}")
        while True:
            try:
                choice = int(input("Select (1/2/3): "))
                if 1 <= choice <= 3:
                    lung_model = lung_models[choice - 1]
                    break
                else:
                    print("  Please enter 1, 2, or 3.")
            except ValueError:
                print("  Please enter 1, 2, or 3.")

        print("\nEnter vitals (normal ranges shown):")
        days        = get_input("  Days since transplant  (1–365):     ", 1,   365)
        oxygen      = get_input("  Oxygen level %         (88–99):     ", 88,  99,  decimal=True)
        breathing   = get_input("  Breathing rate         (12–30):     ", 12,  30)
        inflammation = get_input("  Inflammation score     (0.5–9.5):   ", 0.5, 9.5, decimal=True)
        bp          = get_input("  Blood pressure systolic(90–160):    ", 90,  160)
        cough       = get_input("  Cough frequency        (0–20):      ", 0,   20)
        activity    = get_input("  Activity level         (1–10):      ", 1,   10,  decimal=True)
        strain      = get_input("  Mechanical strain      (0.1–9.9):   ", 0.1, 9.9, decimal=True)

        predict_patient(name, lung_model, days, oxygen, breathing,
                       inflammation, bp, cough, activity, strain)

        again = input("\nAssess another patient? (y/n): ").strip().lower()
        if again != "y":
            print("\nPneumaQuery session ended. Stay vigilant.\n")
            break

if __name__ == "__main__":
    main()
    