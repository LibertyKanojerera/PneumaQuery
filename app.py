import os
import pickle

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="PneumaQuery",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "days_since_transplant",
    "oxygen_level",
    "breathing_rate",
    "inflammation_score",
    "blood_pressure_systolic",
    "cough_frequency",
    "activity_level",
    "mechanical_strain",
]

COLOR_MAP = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#27AE60"}
BG_COLOR = "#F4F6F9"
CARD_COLORS = {
    "High":   ("#E74C3C", "#FDEDEC"),
    "Medium": ("#F39C12", "#FEF9E7"),
    "Low":    ("#27AE60", "#EAFAF1"),
}
LUNG_MODELS = ["LungTech-A1", "LungTech-B2", "BioLung-X3"]


# ── Model loading ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "pneumaquery_model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_model():
    m = load_model()
    if m is None:
        st.error(
            "pneumaquery_model.pkl not found. "
            "Run `python train_model.py` first, then refresh this page."
        )
        st.stop()
    return m


# ── Data loading ──────────────────────────────────────────────

def _run_predictions_local(df: pd.DataFrame) -> pd.DataFrame:
    model = get_model()
    df = df.copy()
    df["predicted_risk"] = model.predict(df[FEATURES])
    return df


def _run_predictions_api(df: pd.DataFrame, api_url: str) -> pd.DataFrame:
    df = df.copy()
    results = []
    errors = []
    endpoint = api_url.rstrip("/") + "/predict"

    progress = st.progress(0, text="Fetching predictions from API...")
    for i, (_, row) in enumerate(df.iterrows()):
        payload = {f: row[f] for f in FEATURES}
        payload["patient_name"] = row.get("patient_name", "")
        payload["lung_model"] = row.get("lung_model", "")
        try:
            resp = requests.post(endpoint, json=payload, timeout=10)
            resp.raise_for_status()
            results.append(resp.json()["risk_label"])
        except Exception as e:
            errors.append(str(e))
            results.append("Unknown")
        progress.progress((i + 1) / len(df), text=f"Fetching predictions... {i+1}/{len(df)}")

    progress.empty()
    if errors:
        st.warning(f"{len(errors)} prediction(s) failed via API. Check the API server.")
    df["predicted_risk"] = results
    return df


def load_local_csv() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "patients.csv")
    if not os.path.exists(path):
        st.error("patients.csv not found in the project directory.")
        st.stop()
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_salesforce_data() -> pd.DataFrame:
    domain = os.getenv("SF_MY_DOMAIN_URL", "").strip().rstrip("/")
    key = os.getenv("SF_CONSUMER_KEY", "").strip()
    secret = os.getenv("SF_CONSUMER_SECRET", "").strip()

    if not all([domain, key, secret]):
        raise ValueError(
            "Missing Salesforce credentials. Set these environment variables:\n"
            "  SF_MY_DOMAIN_URL -- your org domain (e.g. https://yourorg.my.salesforce.com)\n"
            "  SF_CONSUMER_KEY  -- connected app consumer key\n"
            "  SF_CONSUMER_SECRET -- connected app consumer secret"
        )

    token_url = f"{domain}/services/oauth2/token"
    resp = requests.post(
        token_url,
        data={"grant_type": "client_credentials", "client_id": key, "client_secret": secret},
        timeout=15,
    )
    if resp.status_code != 200:
        raise ConnectionError(f"Salesforce OAuth failed ({resp.status_code}): {resp.text}")

    token_data = resp.json()
    access_token = token_data["access_token"]
    instance_url = token_data.get("instance_url", domain)

    soql = (
        "SELECT Name, Lung_Model__c, Days_Since_Transplant__c, Oxygen_Level__c, "
        "Breathing_Rate__c, Inflammation_Score__c, Blood_Pressure_Systolic__c, "
        "Cough_Frequency__c, Activity_Level__c, Mechanical_Strain__c, Risk_Score__c "
        "FROM Patient__c"
    )
    query_url = f"{instance_url}/services/data/v59.0/query"
    headers = {"Authorization": f"Bearer {access_token}"}
    qresp = requests.get(query_url, params={"q": soql}, headers=headers, timeout=15)
    if qresp.status_code != 200:
        raise ConnectionError(f"Salesforce query failed ({qresp.status_code}): {qresp.text}")

    records = qresp.json().get("records", [])
    if not records:
        raise ValueError("No Patient__c records found in Salesforce.")

    rows = []
    for i, r in enumerate(records):
        rows.append({
            "patient_id":               f"P{i+1:03d}",
            "patient_name":             r.get("Name", ""),
            "lung_model":               r.get("Lung_Model__c", ""),
            "days_since_transplant":    r.get("Days_Since_Transplant__c", 0),
            "oxygen_level":             r.get("Oxygen_Level__c", 0.0),
            "breathing_rate":           r.get("Breathing_Rate__c", 0),
            "inflammation_score":       r.get("Inflammation_Score__c", 0.0),
            "blood_pressure_systolic":  r.get("Blood_Pressure_Systolic__c", 0),
            "cough_frequency":          r.get("Cough_Frequency__c", 0),
            "activity_level":           r.get("Activity_Level__c", 0.0),
            "mechanical_strain":        r.get("Mechanical_Strain__c", 0.0),
            "risk_score":               r.get("Risk_Score__c", 0),
        })
    return pd.DataFrame(rows)


# ── Dashboard chart builders ──────────────────────────────────

def _chart_risk_bar(df: pd.DataFrame) -> go.Figure:
    sorted_df = df.sort_values("risk_score", ascending=True)
    colors = sorted_df["predicted_risk"].map(COLOR_MAP)

    fig = go.Figure(go.Bar(
        x=sorted_df["risk_score"],
        y=sorted_df["patient_id"].astype(str),
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{customdata[0]}</b><br>Risk Score: %{x}<br>Risk: %{customdata[1]}<extra></extra>",
        customdata=list(zip(sorted_df["patient_name"], sorted_df["predicted_risk"])),
    ))
    fig.add_vline(x=60, line_dash="dash", line_color="#E74C3C", line_width=1.2, opacity=0.7,
                  annotation_text="High", annotation_font_color="#E74C3C", annotation_position="top right")
    fig.add_vline(x=30, line_dash="dash", line_color="#F39C12", line_width=1.2, opacity=0.7,
                  annotation_text="Med", annotation_font_color="#F39C12", annotation_position="top right")
    fig.update_layout(
        title=dict(text="Patient Risk Scores", font=dict(size=14, color="#1A252F"), x=0),
        xaxis=dict(title="Risk Score", range=[0, 110], gridcolor="#E0E0E0"),
        yaxis=dict(tickfont=dict(size=7), showticklabels=len(df) <= 60),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
        showlegend=False,
    )
    return fig


def _chart_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["predicted_risk"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [COLOR_MAP[l] for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="percent",
        textfont=dict(size=12, color="white"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{len(df)}</b><br>Patients",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#1A252F"),
    )
    fig.update_layout(
        title=dict(text="Risk Distribution", font=dict(size=14, color="#1A252F"), x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


def _chart_scatter_inflammation(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=88, y1=92, fillcolor="#E74C3C", opacity=0.06, line_width=0)
    fig.add_vrect(x0=7, x1=10, fillcolor="#E74C3C", opacity=0.06, line_width=0)
    fig.add_annotation(x=7.1, y=99.2, text="High inflammation zone",
                       showarrow=False, font=dict(size=8, color="#E74C3C"))
    fig.add_annotation(x=0.4, y=91.8, text="Low O2 zone",
                       showarrow=False, font=dict(size=8, color="#E74C3C"))

    for risk_level, group in df.groupby("predicted_risk"):
        fig.add_trace(go.Scatter(
            x=group["inflammation_score"],
            y=group["oxygen_level"],
            mode="markers",
            name=risk_level,
            marker=dict(color=COLOR_MAP[risk_level], size=9, line=dict(color="white", width=0.8)),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Inflammation: %{x}<br>O2: %{y}%<extra></extra>"
            ),
            customdata=group[["patient_name"]].values,
        ))

    fig.update_layout(
        title=dict(text="Inflammation vs Oxygen Level", font=dict(size=14, color="#1A252F"), x=0),
        xaxis=dict(title="Inflammation Score", gridcolor="#E0E0E0"),
        yaxis=dict(title="Oxygen Level (%)", gridcolor="#E0E0E0"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        legend=dict(title="Risk"),
    )
    return fig


def _chart_scatter_bp(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=140, y1=165, fillcolor="#E74C3C", opacity=0.06, line_width=0)
    fig.add_vrect(x0=12, x1=21, fillcolor="#E74C3C", opacity=0.06, line_width=0)
    fig.add_annotation(x=12.2, y=162, text="High BP zone",
                       showarrow=False, font=dict(size=8, color="#E74C3C"))
    fig.add_annotation(x=0.3, y=145, text="High cough zone",
                       showarrow=False, font=dict(size=8, color="#E74C3C"))

    for risk_level, group in df.groupby("predicted_risk"):
        fig.add_trace(go.Scatter(
            x=group["cough_frequency"],
            y=group["blood_pressure_systolic"],
            mode="markers",
            name=risk_level,
            marker=dict(color=COLOR_MAP[risk_level], size=9, line=dict(color="white", width=0.8)),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Cough: %{x}/hr<br>BP: %{y} mmHg<extra></extra>"
            ),
            customdata=group[["patient_name"]].values,
        ))

    fig.update_layout(
        title=dict(text="Blood Pressure vs Cough Frequency", font=dict(size=14, color="#1A252F"), x=0),
        xaxis=dict(title="Cough Frequency (coughs/hr)", gridcolor="#E0E0E0"),
        yaxis=dict(title="Blood Pressure (mmHg)", gridcolor="#E0E0E0"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        legend=dict(title="Risk"),
    )
    return fig


def _chart_lung_model(df: pd.DataFrame) -> go.Figure:
    summary = df.groupby("lung_model").agg(
        avg_risk_score=("risk_score", "mean"),
        avg_oxygen=("oxygen_level", "mean"),
        avg_inflammation=("inflammation_score", "mean"),
        avg_bp=("blood_pressure_systolic", "mean"),
        patient_count=("patient_id", "count"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avg Risk Score",
        x=summary["lung_model"],
        y=summary["avg_risk_score"],
        marker_color="#E74C3C",
        opacity=0.85,
        text=[f"n={int(c)}" for c in summary["patient_count"]],
        textposition="outside",
        textfont=dict(size=9, color="#555"),
        hovertemplate="<b>%{x}</b><br>Avg Risk Score: %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Avg O2 Level",
        x=summary["lung_model"],
        y=summary["avg_oxygen"],
        marker_color="#2980B9",
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Avg O2: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Avg Inflammation x10",
        x=summary["lung_model"],
        y=summary["avg_inflammation"] * 10,
        marker_color="#F39C12",
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Avg Inflammation x10: %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Avg BP / 5",
        x=summary["lung_model"],
        y=summary["avg_bp"] / 5,
        marker_color="#8E44AD",
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Avg BP / 5: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Lung Model Performance (Manufacturer Feedback)",
                   font=dict(size=14, color="#1A252F"), x=0),
        barmode="group",
        xaxis=dict(title="Lung Model"),
        yaxis=dict(gridcolor="#E0E0E0"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5,
                    font=dict(size=9)),
    )
    return fig


# ── Stat cards ────────────────────────────────────────────────

def _render_stat_cards(df: pd.DataFrame):
    high = int((df["predicted_risk"] == "High").sum())
    medium = int((df["predicted_risk"] == "Medium").sum())
    low = int((df["predicted_risk"] == "Low").sum())
    avg_o2 = df["oxygen_level"].mean()
    avg_bp = df["blood_pressure_systolic"].mean()
    avg_cough = df["cough_frequency"].mean()

    cards = [
        (str(high),             "High Risk Patients",  "#E74C3C", "#FDEDEC"),
        (str(medium),           "Needs Watching",       "#F39C12", "#FEF9E7"),
        (str(low),              "Stable",               "#27AE60", "#EAFAF1"),
        (f"{avg_o2:.1f}%",      "Avg O2 Level",         "#2980B9", "#EBF5FB"),
        (f"{avg_bp:.0f} mmHg",  "Avg Blood Pressure",   "#8E44AD", "#F5EEF8"),
        (f"{avg_cough:.1f}",    "Avg Cough Frequency",  "#16A085", "#E8F8F5"),
    ]

    st.markdown("**Population Summary**")
    for value, label, text_color, bg_color in cards:
        st.markdown(
            f"""<div style="background:{bg_color}; border-radius:8px; padding:10px 14px;
                            margin-bottom:8px; text-align:center;">
                  <div style="font-size:22px; font-weight:700; color:{text_color};">{value}</div>
                  <div style="font-size:11px; color:#555; margin-top:2px;">{label}</div>
                </div>""",
            unsafe_allow_html=True,
        )


# ── Pages ─────────────────────────────────────────────────────

def page_home():
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem 1rem 1.5rem;">
          <h1 style="font-size:3rem; font-weight:800; color:#1A252F; margin-bottom:0.25rem;">
            🫁 PneumaQuery
          </h1>
          <p style="font-size:1.25rem; color:#555; margin-bottom:2rem;">
            AI-Powered Digital Twin Monitoring for Transplanted Lung Patients
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(
            """
            <div style="background:#fff; border-radius:12px; padding:2rem 2.5rem;
                        box-shadow:0 2px 12px rgba(0,0,0,0.08);">
              <h3 style="color:#1A252F; margin-top:0;">What is PneumaQuery?</h3>
              <p style="color:#444; line-height:1.7;">
                PneumaQuery is a real-time rejection risk monitoring platform for patients who
                have received a 3D-printed transplant lung. It analyzes eight clinical vitals
                and produces an instant High / Medium / Low risk classification using a
                Random Forest model trained on synthetic patient data.
              </p>
              <hr style="border:none; border-top:1px solid #eee; margin:1.2rem 0;"/>
              <h3 style="color:#1A252F;">The Digital Twin Concept</h3>
              <p style="color:#444; line-height:1.7;">
                Each patient has a continuous digital representation -- a <em>digital twin</em>
                -- built from live sensor data. Clinicians see risk predictions updated in
                real time, while device manufacturers receive aggregate performance feedback
                across every lung model deployed in the field.
              </p>
              <hr style="border:none; border-top:1px solid #eee; margin:1.2rem 0;"/>
              <h3 style="color:#1A252F;">Two Audiences</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        aud_l, aud_r = st.columns(2)
        with aud_l:
            st.markdown(
                """<div style="background:#EBF5FB; border-radius:10px; padding:1.2rem;
                              text-align:center; margin-top:1rem;">
                    <div style="font-size:2rem;">🩺</div>
                    <div style="font-weight:700; color:#1A252F; margin:0.4rem 0;">Clinicians</div>
                    <div style="color:#444; font-size:0.9rem; line-height:1.5;">
                      Monitor patient risk in real time, review individual vital trends, and
                      receive instant alerts for high-risk cases.
                    </div>
                   </div>""",
                unsafe_allow_html=True,
            )
        with aud_r:
            st.markdown(
                """<div style="background:#EAFAF1; border-radius:10px; padding:1.2rem;
                              text-align:center; margin-top:1rem;">
                    <div style="font-size:2rem;">🏭</div>
                    <div style="font-weight:700; color:#1A252F; margin:0.4rem 0;">Manufacturers</div>
                    <div style="color:#444; font-size:0.9rem; line-height:1.5;">
                      Track aggregate outcomes by lung model (BioLung-X3, LungTech-A1,
                      LungTech-B2) to guide product improvements.
                    </div>
                   </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "Use the sidebar to navigate to the **Dashboard** for population-level analytics "
            "or the **Patient Predictor** for a single-patient assessment.",
            icon="ℹ️",
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """<div style="background:#fff; border-radius:10px; padding:1.5rem;
                          box-shadow:0 1px 6px rgba(0,0,0,0.07); text-align:center;">
                <div style="font-size:2rem;">🌲</div>
                <div style="font-weight:700; color:#1A252F; margin:0.5rem 0;">Random Forest</div>
                <div style="color:#666; font-size:0.88rem;">
                  100-tree ensemble trained on 8 clinical features with stratified
                  cross-validation. Zero High Risk patients classified as Low Risk.
                </div>
               </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """<div style="background:#fff; border-radius:10px; padding:1.5rem;
                          box-shadow:0 1px 6px rgba(0,0,0,0.07); text-align:center;">
                <div style="font-size:2rem;">☁️</div>
                <div style="font-weight:700; color:#1A252F; margin:0.5rem 0;">Salesforce Integration</div>
                <div style="color:#666; font-size:0.88rem;">
                  Patient records live in a Salesforce Patient__c custom object.
                  The dashboard can pull live data via OAuth client credentials flow.
                </div>
               </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """<div style="background:#fff; border-radius:10px; padding:1.5rem;
                          box-shadow:0 1px 6px rgba(0,0,0,0.07); text-align:center;">
                <div style="font-size:2rem;">⚡</div>
                <div style="font-weight:700; color:#1A252F; margin:0.5rem 0;">REST API</div>
                <div style="color:#666; font-size:0.88rem;">
                  A FastAPI server (api.py) exposes a /predict endpoint so any system
                  can request risk predictions with a simple HTTP POST.
                </div>
               </div>""",
            unsafe_allow_html=True,
        )


def page_dashboard():
    st.markdown(
        "<h2 style='color:#1A252F; margin-bottom:0.2rem;'>Clinician Risk Dashboard</h2>"
        "<p style='color:#666; margin-top:0;'>Population-level monitoring for all transplant patients</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Sidebar controls ──────────────────────────────────────
    with st.sidebar:
        st.markdown("### Data Source")
        source = st.radio(
            "Load patients from:",
            ["Local CSV", "Upload CSV", "Salesforce"],
            key="data_source",
        )

        st.markdown("### Predictions")
        use_api = st.checkbox("Use FastAPI endpoint for predictions", value=False)
        if use_api:
            api_url = st.text_input("API base URL", value="http://localhost:8000")
        else:
            api_url = "http://localhost:8000"

        if source == "Salesforce":
            st.markdown("---")
            st.markdown("**Salesforce credentials** are read from environment variables:")
            st.code("SF_MY_DOMAIN_URL\nSF_CONSUMER_KEY\nSF_CONSUMER_SECRET", language="bash")
            st.caption(
                "Set these in your .env file or shell environment. "
                "Results are cached for 5 minutes."
            )

    # ── Load data ──────────────────────────────────────────────
    df = None

    if source == "Local CSV":
        df = load_local_csv()

    elif source == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload a patient CSV (same schema as patients.csv)",
            type="csv",
            key="csv_upload",
        )
        if uploaded is None:
            st.info("Upload a CSV file to continue.")
            return
        df = pd.read_csv(uploaded)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Uploaded CSV is missing required columns: {missing}")
            return

    elif source == "Salesforce":
        with st.spinner("Connecting to Salesforce and querying Patient__c records..."):
            try:
                df = load_salesforce_data()
                st.success(f"Loaded {len(df)} records from Salesforce (cached for 5 min).")
            except Exception as e:
                st.error(f"Salesforce connection failed: {e}")
                st.markdown(
                    "**Fix:** ensure these env vars are set in your `.env` file:\n"
                    "```\nSF_MY_DOMAIN_URL=https://yourorg.my.salesforce.com\n"
                    "SF_CONSUMER_KEY=your_key\nSF_CONSUMER_SECRET=your_secret\n```"
                )
                return

    if df is None or df.empty:
        return

    # Add patient_id if missing
    if "patient_id" not in df.columns:
        df.insert(0, "patient_id", [f"P{i+1:03d}" for i in range(len(df))])
    if "patient_name" not in df.columns:
        df["patient_name"] = df["patient_id"]
    if "risk_score" not in df.columns:
        df["risk_score"] = 50

    # ── Run predictions ────────────────────────────────────────
    if use_api:
        df = _run_predictions_api(df, api_url)
    else:
        df = _run_predictions_local(df)

    # ── Row 1: bar + donut + stat cards ───────────────────────
    col1, col2, col3 = st.columns([2.2, 1.6, 1.2])
    with col1:
        st.plotly_chart(_chart_risk_bar(df), use_container_width=True, config={"displayModeBar": False})
    with col2:
        st.plotly_chart(_chart_donut(df), use_container_width=True, config={"displayModeBar": False})
    with col3:
        _render_stat_cards(df)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: three scatter/bar plots ────────────────────────
    col4, col5, col6 = st.columns(3)
    with col4:
        st.plotly_chart(_chart_scatter_inflammation(df), use_container_width=True,
                        config={"displayModeBar": False})
    with col5:
        st.plotly_chart(_chart_scatter_bp(df), use_container_width=True,
                        config={"displayModeBar": False})
    with col6:
        st.plotly_chart(_chart_lung_model(df), use_container_width=True,
                        config={"displayModeBar": False})

    # ── Patient table toggle ───────────────────────────────────
    with st.expander("View patient data table"):
        display_cols = ["patient_id", "patient_name", "lung_model", "predicted_risk",
                        "risk_score", "oxygen_level", "breathing_rate", "inflammation_score",
                        "blood_pressure_systolic", "cough_frequency", "activity_level",
                        "mechanical_strain", "days_since_transplant"]
        show_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True)


def page_predictor():
    st.markdown(
        "<h2 style='color:#1A252F; margin-bottom:0.2rem;'>Single Patient Predictor</h2>"
        "<p style='color:#666; margin-top:0;'>Enter patient vitals to receive an instant risk assessment</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    model = get_model()

    with st.sidebar:
        st.markdown("### Prediction Source")
        use_api = st.checkbox("Use FastAPI endpoint", value=False, key="pred_use_api")
        if use_api:
            api_url = st.text_input("API base URL", value="http://localhost:8000", key="pred_api_url")
        else:
            api_url = "http://localhost:8000"

    col_form, col_result = st.columns([1, 1.2], gap="large")

    with col_form:
        st.markdown("#### Patient Information")
        patient_name = st.text_input("Patient name", placeholder="e.g. Jane Smith")
        lung_model = st.selectbox("Lung model", LUNG_MODELS)

        st.markdown("#### Clinical Vitals")
        col_a, col_b = st.columns(2)
        with col_a:
            days = st.number_input("Days since transplant", min_value=1, max_value=365,
                                   value=90, step=1)
            oxygen = st.number_input("Oxygen level (%)", min_value=88.0, max_value=99.0,
                                     value=95.0, step=0.1, format="%.1f")
            breathing = st.number_input("Breathing rate (breaths/min)", min_value=12, max_value=30,
                                        value=16, step=1)
            inflammation = st.number_input("Inflammation score (0.5-9.5)", min_value=0.5,
                                           max_value=9.5, value=2.5, step=0.1, format="%.1f")
        with col_b:
            bp = st.number_input("Blood pressure systolic (mmHg)", min_value=90, max_value=160,
                                  value=120, step=1)
            cough = st.number_input("Cough frequency (coughs/hr)", min_value=0, max_value=20,
                                     value=3, step=1)
            activity = st.number_input("Activity level (1-10)", min_value=1.0, max_value=10.0,
                                        value=6.0, step=0.1, format="%.1f")
            strain = st.number_input("Mechanical strain (0.1-9.9)", min_value=0.1, max_value=9.9,
                                      value=2.0, step=0.1, format="%.1f")

        submitted = st.button("Assess Risk", type="primary", use_container_width=True)

    with col_result:
        if not submitted:
            st.markdown(
                """<div style="background:#F4F6F9; border-radius:12px; padding:3rem 2rem;
                              text-align:center; margin-top:2rem;">
                    <div style="font-size:3rem;">🫁</div>
                    <div style="color:#888; margin-top:1rem;">
                      Fill in the vitals form and click <strong>Assess Risk</strong>
                      to see the prediction.
                    </div>
                   </div>""",
                unsafe_allow_html=True,
            )
        else:
            payload = {
                "days_since_transplant":   float(days),
                "oxygen_level":            float(oxygen),
                "breathing_rate":          float(breathing),
                "inflammation_score":      float(inflammation),
                "blood_pressure_systolic": float(bp),
                "cough_frequency":         float(cough),
                "activity_level":          float(activity),
                "mechanical_strain":       float(strain),
                "patient_name":            patient_name or "Unknown Patient",
                "lung_model":              lung_model,
            }

            if use_api:
                try:
                    resp = requests.post(
                        api_url.rstrip("/") + "/predict",
                        json=payload,
                        timeout=10,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    risk_label = result["risk_label"]
                    confidence = result["confidence"]
                    classes = list(confidence.keys())
                    probs = list(confidence.values())
                except Exception as e:
                    st.error(f"API request failed: {e}")
                    return
            else:
                row_df = pd.DataFrame([payload])
                risk_label = model.predict(row_df[FEATURES])[0]
                probs_arr = model.predict_proba(row_df[FEATURES])[0]
                classes = list(model.classes_)
                probs = [float(p) for p in probs_arr]
                confidence = dict(zip(classes, probs))

            risk_color = COLOR_MAP[risk_label]
            if risk_label == "High":
                alert_text = "HIGH RISK -- Immediate clinical review recommended"
                border_style = f"4px solid {risk_color}"
                icon = "🔴"
            elif risk_label == "Medium":
                alert_text = "MEDIUM RISK -- Schedule follow-up within 48 hours"
                border_style = f"4px solid {risk_color}"
                icon = "🟡"
            else:
                alert_text = "LOW RISK -- Patient is stable"
                border_style = f"4px solid {risk_color}"
                icon = "🟢"

            name_display = patient_name or "Unknown Patient"
            st.markdown(
                f"""<div style="border:{border_style}; border-radius:12px; padding:1.5rem 2rem;
                               background:#fff; box-shadow:0 2px 10px rgba(0,0,0,0.08);">
                      <div style="font-size:0.85rem; color:#888; text-transform:uppercase;
                                  letter-spacing:0.05em; margin-bottom:0.5rem;">
                        PNEUMAQUERY RISK ASSESSMENT
                      </div>
                      <div style="font-size:1.3rem; font-weight:700; color:#1A252F;">
                        {name_display}
                      </div>
                      <div style="color:#666; font-size:0.9rem; margin-bottom:1rem;">
                        {lung_model} &nbsp;|&nbsp; {int(days)} days post-op
                      </div>
                      <div style="background:{COLOR_MAP[risk_label]}18; border-radius:8px;
                                  padding:0.8rem 1rem; margin-bottom:1.2rem;">
                        <span style="font-size:1.4rem; font-weight:800;
                                     color:{risk_color};">{icon} {risk_label} Risk</span><br/>
                        <span style="color:{risk_color}; font-size:0.9rem;">{alert_text}</span>
                      </div>
                      <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem 1.5rem;
                                  font-size:0.88rem; color:#444; margin-bottom:1.2rem;">
                        <div>O2 Level: <b>{oxygen}%</b></div>
                        <div>Breathing Rate: <b>{breathing} br/min</b></div>
                        <div>Inflammation: <b>{inflammation}/10</b></div>
                        <div>Blood Pressure: <b>{bp} mmHg</b></div>
                        <div>Cough Freq: <b>{cough}/hr</b></div>
                        <div>Activity Level: <b>{activity}/10</b></div>
                        <div>Mech. Strain: <b>{strain}/10</b></div>
                        <div>Days Post-Op: <b>{int(days)}</b></div>
                      </div>
                    </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("#### Confidence Breakdown")
            conf_fig = go.Figure(go.Bar(
                x=[confidence.get(c, 0) for c in ["High", "Medium", "Low"]],
                y=["High", "Medium", "Low"],
                orientation="h",
                marker_color=[COLOR_MAP["High"], COLOR_MAP["Medium"], COLOR_MAP["Low"]],
                text=[f"{confidence.get(c, 0):.0%}" for c in ["High", "Medium", "Low"]],
                textposition="auto",
                hovertemplate="%{y}: %{x:.1%}<extra></extra>",
            ))
            conf_fig.update_layout(
                xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#E0E0E0"),
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                height=160,
                showlegend=False,
            )
            st.plotly_chart(conf_fig, use_container_width=True, config={"displayModeBar": False})


# ── Sidebar navigation ────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """<div style="text-align:center; padding:0.5rem 0 1rem;">
             <span style="font-size:2rem;">🫁</span><br/>
             <span style="font-size:1.2rem; font-weight:800; color:#1A252F;">PneumaQuery</span><br/>
             <span style="font-size:0.78rem; color:#888;">
               Digital Twin Lung Monitor
             </span>
           </div>""",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "Navigate",
        ["Home", "Dashboard", "Patient Predictor"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Prototype -- American University\nKogod School of Business, Spring 2026\nTeam Black")

# ── Render selected page ──────────────────────────────────────

if page == "Home":
    page_home()
elif page == "Dashboard":
    page_dashboard()
else:
    page_predictor()
