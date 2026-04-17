import os
import requests
import pandas as pd
import pickle
from simple_salesforce import Salesforce
from dotenv import load_dotenv

load_dotenv()

print("Connecting to Salesforce via OAuth (Client Credentials)...")

my_domain = os.getenv("SF_MY_DOMAIN_URL", "").strip().rstrip("/")

if not my_domain:
    raise ValueError(
        "SF_MY_DOMAIN_URL is missing. Add this to your .env file:\n"
        "SF_MY_DOMAIN_URL=https://pneumaquery-dev-ed.trailblaze.my.salesforce.com"
    )

token_url = f"{my_domain}/services/oauth2/token"

payload = {
    "grant_type": "client_credentials",
    "client_id": os.getenv("SF_CONSUMER_KEY", "").strip(),
    "client_secret": os.getenv("SF_CONSUMER_SECRET", "").strip(),
}

print("My Domain:", my_domain)
print("Token URL:", token_url)

response = requests.post(token_url, data=payload)
print(f"Status code: {response.status_code}")
print(f"Full response: {response.text}")

if response.status_code != 200:
    print(f"Login failed: {response.text}")
    raise SystemExit(1)

token_data   = response.json()
access_token = token_data["access_token"]
instance_url = token_data.get("instance_url", my_domain)

print(f"Connected! Instance: {instance_url}\n")

sf = Salesforce(
    instance_url=instance_url,
    session_id=access_token
)

# ── Clear existing records to avoid duplicates ────────────────
print("Clearing existing patient records...")
existing = sf.query("SELECT Id FROM Patient__c")
records  = existing["records"]

if records:
    for record in records:
        sf.Patient__c.delete(record["Id"])
    print(f"Deleted {len(records)} existing records.\n")
else:
    print("No existing records found.\n")

# ── Load model and data ───────────────────────────────────────
with open("pneumaquery_model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("patients.csv")

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

df["predicted_risk"]  = model.predict(df[features])
df["alert_triggered"] = df["predicted_risk"] == "High"

print(f"Pushing {len(df)} patients to Salesforce...\n")

success = 0
failed  = 0

for _, row in df.iterrows():
    try:
        sf.Patient__c.create({
            "Name":                        row["patient_name"],
            "Lung_Model__c":               row["lung_model"],
            "Days_Since_Transplant__c":    int(row["days_since_transplant"]),
            "Oxygen_Level__c":             float(row["oxygen_level"]),
            "Breathing_Rate__c":           int(row["breathing_rate"]),
            "Inflammation_Score__c":       float(row["inflammation_score"]),
            "Blood_Pressure_Systolic__c":  int(row["blood_pressure_systolic"]),
            "Cough_Frequency__c":          int(row["cough_frequency"]),
            "Activity_Level__c":           float(row["activity_level"]),
            "Mechanical_Strain__c":        float(row["mechanical_strain"]),
            "Risk_Score__c":               int(row["risk_score"]),
            "Risk_Level__c":               row["predicted_risk"],
            "Alert_Triggered__c":          bool(row["alert_triggered"]),
        })
        print(f"  ✓ {row['patient_name']} — {row['predicted_risk']} Risk")
        success += 1

    except Exception as e:
        print(f"  ✗ {row['patient_name']} failed: {e}")
        failed += 1

print(f"\nDone! {success} patients pushed, {failed} failed.")
print("Go check your Salesforce Patients tab!")
