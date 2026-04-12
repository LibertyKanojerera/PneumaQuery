import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pickle

# ── Load data and model ───────────────────────────────────────
df = pd.read_csv("patients.csv")

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

df["predicted_risk"] = model.predict(df[features])
color_map = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#27AE60"}
df["color"] = df["predicted_risk"].map(color_map)

# ── Page setup ────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12), facecolor="#F4F6F9")
fig.suptitle("PneumaQuery — Clinician Risk Dashboard",
             fontsize=20, fontweight="bold", color="#1A252F", y=0.98)

# ── 1. Patient Risk Score Bar Chart (top left) ────────────────
ax1 = fig.add_axes([0.03, 0.55, 0.38, 0.38])
ax1.set_facecolor("#FFFFFF")

sorted_df = df.sort_values("risk_score", ascending=False)
ax1.barh(sorted_df["patient_id"], sorted_df["risk_score"],
         color=sorted_df["color"], edgecolor="none", height=0.6)

ax1.axvline(x=60, color="#E74C3C", linestyle="--", linewidth=1.2, alpha=0.7)
ax1.axvline(x=30, color="#F39C12", linestyle="--", linewidth=1.2, alpha=0.7)
ax1.text(61, len(df) - 1, "High", color="#E74C3C", fontsize=7, va="top")
ax1.text(31, len(df) - 1, "Med",  color="#F39C12", fontsize=7, va="top")

ax1.set_xlabel("Risk Score", fontsize=9, color="#555")
ax1.set_title("Patient Risk Scores (50 Patients)",
              fontsize=11, fontweight="bold", color="#1A252F", pad=8)
ax1.tick_params(axis="both", labelsize=6, colors="#555")
ax1.set_xlim(0, 110)
ax1.spines[["top", "right", "left"]].set_visible(False)
ax1.xaxis.grid(True, linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)

# ── 2. Risk Distribution Donut (top middle) ───────────────────
ax2 = fig.add_axes([0.45, 0.55, 0.20, 0.38])
ax2.set_facecolor("#FFFFFF")

counts = df["predicted_risk"].value_counts()
labels = counts.index.tolist()
sizes  = counts.values.tolist()
colors = [color_map[l] for l in labels]

wedges, texts, autotexts = ax2.pie(
    sizes, colors=colors, autopct="%1.0f%%", startangle=90,
    wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
    pctdistance=0.75
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_color("white")
    at.set_fontweight("bold")

ax2.text(0, 0, f"{len(df)}\nPatients", ha="center", va="center",
         fontsize=11, fontweight="bold", color="#1A252F")

legend_patches = [mpatches.Patch(color=color_map[l], label=f"{l} ({c})")
                  for l, c in zip(labels, sizes)]
ax2.legend(handles=legend_patches, loc="lower center",
           bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)
ax2.set_title("Risk Distribution", fontsize=11,
              fontweight="bold", color="#1A252F", pad=8)

# ── 3. Stat Cards (top right) ─────────────────────────────────
high_count   = (df["predicted_risk"] == "High").sum()
medium_count = (df["predicted_risk"] == "Medium").sum()
low_count    = (df["predicted_risk"] == "Low").sum()
avg_o2       = df["oxygen_level"].mean()
avg_bp       = df["blood_pressure_systolic"].mean()
avg_cough    = df["cough_frequency"].mean()

cards = [
    (f"{high_count}",      "High Risk Patients",   "#E74C3C", "#FDEDEC"),
    (f"{medium_count}",    "Needs Watching",        "#F39C12", "#FEF9E7"),
    (f"{low_count}",       "Stable",                "#27AE60", "#EAFAF1"),
    (f"{avg_o2:.1f}%",     "Avg O₂ Level",          "#2980B9", "#EBF5FB"),
    (f"{avg_bp:.0f} mmHg", "Avg Blood Pressure",    "#8E44AD", "#F5EEF8"),
    (f"{avg_cough:.1f}",   "Avg Cough Frequency",   "#16A085", "#E8F8F5"),
]

for idx, (value, label, text_color, bg_color) in enumerate(cards):
    left = 0.78
    top  = 0.905 - idx * 0.065
    card_ax = fig.add_axes([left, top, 0.19, 0.055])
    card_ax.set_facecolor(bg_color)
    card_ax.set_xlim(0, 1)
    card_ax.set_ylim(0, 1)
    card_ax.axis("off")
    card_ax.text(0.5, 0.62, value, ha="center", va="center",
                 fontsize=14, fontweight="bold", color=text_color)
    card_ax.text(0.5, 0.22, label, ha="center", va="center",
                 fontsize=7, color="#555")

# ── 4. Inflammation vs Oxygen Scatter (bottom left) ───────────
ax4 = fig.add_axes([0.03, 0.06, 0.30, 0.38])
ax4.set_facecolor("#FFFFFF")

for risk_level, group in df.groupby("predicted_risk"):
    ax4.scatter(group["inflammation_score"], group["oxygen_level"],
                c=color_map[risk_level], label=risk_level,
                s=80, edgecolors="white", linewidth=0.8, alpha=0.85)

ax4.axhspan(88, 92, alpha=0.06, color="#E74C3C")
ax4.axvspan(7, 10, alpha=0.06, color="#E74C3C")
ax4.text(7.05, 99.2, "High inflammation zone", fontsize=7,
         color="#E74C3C", alpha=0.8)
ax4.text(0.3, 91.8, "Low O₂ zone", fontsize=7,
         color="#E74C3C", alpha=0.8)

ax4.set_xlabel("Inflammation Score", fontsize=9, color="#555")
ax4.set_ylabel("Oxygen Level (%)",   fontsize=9, color="#555")
ax4.set_title("Inflammation vs Oxygen Level", fontsize=11,
              fontweight="bold", color="#1A252F", pad=8)
ax4.legend(title="Risk", fontsize=8, title_fontsize=8, frameon=False)
ax4.tick_params(labelsize=8, colors="#555")
ax4.spines[["top", "right"]].set_visible(False)
ax4.xaxis.grid(True, linestyle="--", alpha=0.3)
ax4.yaxis.grid(True, linestyle="--", alpha=0.3)
ax4.set_axisbelow(True)

# ── 5. Blood Pressure vs Cough Frequency (bottom middle) ──────
ax5 = fig.add_axes([0.36, 0.06, 0.28, 0.38])
ax5.set_facecolor("#FFFFFF")

for risk_level, group in df.groupby("predicted_risk"):
    ax5.scatter(group["cough_frequency"], group["blood_pressure_systolic"],
                c=color_map[risk_level], label=risk_level,
                s=80, edgecolors="white", linewidth=0.8, alpha=0.85)

ax5.axhspan(140, 160, alpha=0.06, color="#E74C3C")
ax5.axvspan(12, 20,  alpha=0.06, color="#E74C3C")
ax5.text(12.2, 159, "High BP zone",    fontsize=7, color="#E74C3C", alpha=0.8)
ax5.text(0.3,  145, "High cough zone", fontsize=7, color="#E74C3C", alpha=0.8)

ax5.set_xlabel("Cough Frequency (coughs/hr)", fontsize=9, color="#555")
ax5.set_ylabel("Blood Pressure (mmHg)",        fontsize=9, color="#555")
ax5.set_title("Blood Pressure vs Cough Frequency", fontsize=11,
              fontweight="bold", color="#1A252F", pad=8)
ax5.legend(title="Risk", fontsize=8, title_fontsize=8, frameon=False)
ax5.tick_params(labelsize=8, colors="#555")
ax5.spines[["top", "right"]].set_visible(False)
ax5.xaxis.grid(True, linestyle="--", alpha=0.3)
ax5.yaxis.grid(True, linestyle="--", alpha=0.3)
ax5.set_axisbelow(True)

# ── 6. Lung Model Performance (bottom right) ──────────────────
ax6 = fig.add_axes([0.67, 0.06, 0.30, 0.38])
ax6.set_facecolor("#FFFFFF")

model_summary = df.groupby("lung_model").agg(
    avg_risk_score    =("risk_score",            "mean"),
    avg_oxygen        =("oxygen_level",           "mean"),
    avg_inflammation  =("inflammation_score",     "mean"),
    avg_bp            =("blood_pressure_systolic","mean"),
    patient_count     =("patient_id",             "count")
).reset_index()

x     = np.arange(len(model_summary))
width = 0.20

ax6.bar(x - width*1.5, model_summary["avg_risk_score"],
        width, label="Avg Risk Score",      color="#E74C3C", alpha=0.8)
ax6.bar(x - width*0.5, model_summary["avg_oxygen"],
        width, label="Avg O₂ Level",        color="#2980B9", alpha=0.8)
ax6.bar(x + width*0.5, model_summary["avg_inflammation"] * 10,
        width, label="Avg Inflammation ×10",color="#F39C12", alpha=0.8)
ax6.bar(x + width*1.5, model_summary["avg_bp"] / 5,
        width, label="Avg BP ÷5",           color="#8E44AD", alpha=0.8)

ax6.set_xticks(x)
ax6.set_xticklabels(model_summary["lung_model"], fontsize=8, color="#555")
ax6.set_title("Lung Model Performance\n(Manufacturer Feedback)",
              fontsize=11, fontweight="bold", color="#1A252F", pad=8)
ax6.legend(fontsize=6, frameon=False)
ax6.tick_params(axis="y", labelsize=8, colors="#555")
ax6.spines[["top", "right"]].set_visible(False)
ax6.yaxis.grid(True, linestyle="--", alpha=0.3)
ax6.set_axisbelow(True)

for i, row in model_summary.iterrows():
    ax6.text(i - width*1.5, row["avg_risk_score"] + 0.5,
             f'n={int(row["patient_count"])}',
             ha="center", fontsize=7, color="#555")

plt.savefig("pneumaquery_dashboard.png", dpi=150,
            bbox_inches="tight", facecolor="#F4F6F9")
plt.show()
print("Dashboard saved as pneumaquery_dashboard.png")
