import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import numpy as np

df = pd.read_csv("patients.csv")

print("Loading patient data...")
print(f"Total patients: {len(df)}\n")

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

X = df[features]
y = df["risk_label"]

# ── Train/Test Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training on {len(X_train)} patients...")
print(f"Testing on  {len(X_test)} patients...\n")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Test Set Performance ──────────────────────────────────────
y_pred = model.predict(X_test)
print("=" * 50)
print("MODEL PERFORMANCE ON TEST PATIENTS:")
print("=" * 50)
print(classification_report(y_test, y_pred))

# ── Confusion Matrix ──────────────────────────────────────────
print("Confusion Matrix:")
print("(rows = actual, columns = predicted)\n")
cm = confusion_matrix(y_test, y_pred, labels=["High", "Medium", "Low"])
cm_df = pd.DataFrame(
    cm,
    index=["Actual High", "Actual Medium", "Actual Low"],
    columns=["Pred High", "Pred Medium", "Pred Low"]
)
print(cm_df.to_string())
print()

# ── Cross Validation ─────────────────────────────────────────
print("=" * 50)
print("CROSS VALIDATION (5-Fold Stratified):")
print("=" * 50)
print("What this means: we split the data into 5 groups,")
print("train on 4 and test on 1, repeating 5 times.")
print("This gives a more reliable accuracy estimate.\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
cv_f1       = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")

print(f"Accuracy per fold:  {[f'{s:.2f}' for s in cv_accuracy]}")
print(f"Mean CV Accuracy:   {cv_accuracy.mean():.2%}")
print(f"Standard Deviation: {cv_accuracy.std():.2%}")
print()
print(f"F1 Score per fold:  {[f'{s:.2f}' for s in cv_f1]}")
print(f"Mean CV F1 Score:   {cv_f1.mean():.2%}")
print()

# Interpret the std
if cv_accuracy.std() < 0.05:
    print("[OK] Low variance -- model is stable and consistent across folds")
elif cv_accuracy.std() < 0.10:
    print("[!!] Moderate variance -- model is reasonably stable")
else:
    print("[XX] High variance -- model may be overfitting, consider more data")

# ── Feature Importance ────────────────────────────────────────
print()
print("=" * 50)
print("WHICH VITALS MATTER MOST:")
print("=" * 50)
importance = pd.Series(model.feature_importances_, index=features)
importance = importance.sort_values(ascending=False)
for feature, score in importance.items():
    bar = "#" * int(score * 40)
    print(f"  {feature:<28} {bar} {score:.2f}")

# ── Save Model ────────────────────────────────────────────────
with open("pneumaquery_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as pneumaquery_model.pkl")
print("Ready for predictions.")
