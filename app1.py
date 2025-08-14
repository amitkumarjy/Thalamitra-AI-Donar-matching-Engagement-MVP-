# app.py
# ThalaMitra – MVP for Syrotech: AI-Based Donor Matching & Engagement
# ---------------------------------------------
# Features:
# 1) Predict likelihood of repeat donation (RandomForest on transfusion.csv)
# 2) Eligibility checker (configurable cooldown days)
# 3) Donor leaderboard (sample or user-uploaded CSV)
# 4) Daily eligible donor count (from donor list)
# 5) Notification simulation (SMS/WhatsApp placeholder)
# 6) e-RaktKosh API placeholder (demo fetch)
# ---------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta, date
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="ThalaMitra – AI Donor Matching MVP",
    page_icon="🩸",
    layout="wide"
)

# ================ Helpers ================

@st.cache_data
def load_transfusion(csv_path: str = "transfusion.csv"):
    df = pd.read_csv(csv_path)
    # Normalize column names (dataset variants exist)
    col_map = {
        'Recency (months)': 'Recency',
        'Frequency (times)': 'Frequency',
        'Monetary (c.c. blood)': 'Monetary',
        'Time (months)': 'Time',
        'whether he/she donated blood in March 2007': 'Target'
    }
    for k, v in col_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
    # Some datasets label target as 0/1 or True/False—force to int 0/1
    if 'Target' not in df.columns:
        # try to find last column as target
        last_col = df.columns[-1]
        df.rename(columns={last_col: 'Target'}, inplace=True)
    df['Target'] = df['Target'].astype(int)
    return df

def train_model(df: pd.DataFrame):
    X = df[['Recency', 'Frequency', 'Monetary', 'Time']].copy()
    y = df['Target'].copy()
    # scale optional (RF not required, but helps interpret sliders)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, class_weight="balanced_subsample"
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = np.nan
    return model, scaler, acc, auc

def simulate_notification(name, phone, channel, message):
    # Placeholder only – no real sending
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "timestamp": ts, "name": name, "phone": phone,
        "channel": channel, "message": message, "status": "SIMULATED_SENT"
    }

def compute_eligibility(last_donation: date, cooldown_days: int):
    next_eligible = last_donation + timedelta(days=cooldown_days)
    today = date.today()
    eligible = today >= next_eligible
    days_left = 0 if eligible else (next_eligible - today).days
    return eligible, next_eligible, days_left

def gen_sample_donors(n=200, seed=42):
    np.random.seed(seed)
    names = [f"Donor-{i+1:03d}" for i in range(n)]
    genders = np.random.choice(["Male", "Female"], size=n, p=[0.6, 0.4])
    blood_groups = np.random.choice(
        ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
        size=n, p=[0.28,0.04,0.29,0.03,0.08,0.02,0.22,0.04]
    )
    last_days_ago = np.random.randint(1, 365, size=n)
    last_dates = [date.today() - timedelta(days=int(d)) for d in last_days_ago]
    total_units = np.random.randint(1, 15, size=n)
    score = total_units * 10 + (365 - last_days_ago)  # simple score
    df = pd.DataFrame({
        "name": names,
        "gender": genders,
        "blood_group": blood_groups,
        "last_donation_date": last_dates,
        "total_donations": total_units,
        "contribution_score": score
    })
    return df

def parse_donors_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=["last_donation_date"])
    # ensure date only
    df["last_donation_date"] = df["last_donation_date"].dt.date
    # basic required columns check
    required = {"name", "gender", "blood_group", "last_donation_date", "total_donations"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in donors CSV: {missing}")
    if "contribution_score" not in df.columns:
        df["contribution_score"] = df["total_donations"] * 10
    return df

# ================ Sidebar ================
st.sidebar.title("🩸 ThalaMitra – Controls")

cooldown = st.sidebar.number_input(
    "Donation cooldown (days) – demo only",
    min_value=30, max_value=200, value=90, step=5,
    help="For demo only. Actual medical eligibility varies. Consult local guidelines."
)

uploaded_donors = st.sidebar.file_uploader(
    "Upload donors CSV (optional)",
    type=["csv"],
    help=(
        "Columns: name, gender, blood_group, last_donation_date (YYYY-MM-DD), "
        "total_donations, [contribution_score optional]"
    )
)

if uploaded_donors:
    try:
        donors_df = parse_donors_csv(uploaded_donors)
        st.sidebar.success("Donor list uploaded.")
    except Exception as e:
        st.sidebar.error(f"Error reading donor CSV: {e}")
        donors_df = gen_sample_donors()
else:
    donors_df = gen_sample_donors()

with st.sidebar.expander("📥 Download sample donors CSV"):
    sample = gen_sample_donors(50)
    buf = io.StringIO()
    # ensure ISO date
    sample["last_donation_date"] = sample["last_donation_date"].astype(str)
    sample.to_csv(buf, index=False)
    st.download_button(
        "Download sample_donors.csv",
        data=buf.getvalue(),
        file_name="sample_donors.csv",
        mime="text/csv"
    )

# ================ Header ================
st.title("🩸 ThalaMitra – AI Donor Matching & Engagement (MVP)")
st.caption("Built for Syrotech MVP Hackathon • Social Impact • Demo only (no real PHI transmitted)")

# ================ Data & Model ================
with st.spinner("Loading dataset & training model..."):
    df = load_transfusion("transfusion.csv")
    model, scaler, acc, auc = train_model(df)

top_kpis = st.container()
with top_kpis:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy", f"{acc*100:.1f}%")
    c2.metric("ROC AUC", f"{(auc*100):.1f}%" if not np.isnan(auc) else "—")
    # Eligible donors today
    today = date.today()
    eligible_today = 0
    for _, r in donors_df.iterrows():
        elig, _, _ = compute_eligibility(r["last_donation_date"], cooldown)
        if elig:
            eligible_today += 1
    c3.metric("Eligible Donors Today", f"{eligible_today}")
    c4.metric("Total Donors (demo)", f"{len(donors_df)}")

st.divider()

# ================ Tabs ================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Donation Prediction", "✅ Eligibility Checker",
    "🏆 Leaderboard", "📨 Notifications", "🏥 e-RaktKosh (Demo)"
])

# ---------- Tab 1: Prediction ----------
with tab1:
    st.subheader("Predict Likelihood of Repeat Donation")
    st.caption("Model trained on UCI 'Blood Transfusion Service Center' dataset. Demo purpose only.")

    colA, colB, colC, colD = st.columns(4)
    recency = colA.number_input("Recency (months since last donation)", min_value=0, max_value=50, value=5, step=1)
    frequency = colB.number_input("Frequency (times donated)", min_value=0, max_value=50, value=2, step=1)
    monetary = colC.number_input("Monetary (c.c. blood donated total)", min_value=0, max_value=20000, value=750, step=50)
    tmonths = colD.number_input("Time (months since first donation)", min_value=0, max_value=200, value=20, step=1)

    X_in = pd.DataFrame([[recency, frequency, monetary, tmonths]], columns=["Recency","Frequency","Monetary","Time"])
    X_in_scaled = scaler.transform(X_in)

    if st.button("Predict", type="primary"):
        proba = model.predict_proba(X_in_scaled)[0, 1]
        pred = int(proba >= 0.5)
        if pred == 1:
            st.success(f"✅ Likely to donate again (confidence ~ {proba*100:.1f}%).")
        else:
            st.warning(f"⚠️ Unlikely to donate again (confidence ~ {(1-proba)*100:.1f}%).")
        with st.expander("How this works"):
            st.write(
                "This is a classification model (RandomForest) trained on past donation patterns "
                "to estimate likelihood of future donation. Threshold is set at 0.5 for demo."
            )

# ---------- Tab 2: Eligibility Checker ----------
with tab2:
    st.subheader("Donation Eligibility Checker (Demo)")
    st.caption("This is a demo calculator with configurable cooldown days. Eligibility rules vary by jurisdiction.")

    name = st.text_input("Donor Name", value="Rohan")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    blood_group = st.selectbox("Blood Group", ["A+","A-","B+","B-","AB+","AB-","O+","O-"])
    last_date = st.date_input("Last Donation Date", value=date.today() - timedelta(days=120))
    eligible, next_date, days_left = compute_eligibility(last_date, cooldown)

    if eligible:
        st.success(f"✅ {name} is eligible to donate today. Next eligible date was {next_date}.")
    else:
        st.info(f"🗓️ {name} will be eligible on **{next_date}** (in {days_left} days).")

    st.markdown(
        "> **Disclaimer:** This is a simplified demo. Actual medical eligibility depends on multiple factors (weight, Hb, vitals, medications, etc.)."
    )

# ---------- Tab 3: Leaderboard ----------
with tab3:
    st.subheader("Top Contributors (Demo Leaderboard)")
    donors_df = donors_df.sort_values("contribution_score", ascending=False).reset_index(drop=True)
    donors_df["rank"] = np.arange(1, len(donors_df) + 1)
    st.dataframe(
        donors_df[["rank","name","blood_group","total_donations","last_donation_date","contribution_score"]],
        use_container_width=True, height=420
    )

    st.markdown("#### Badge Rules (Demo)")
    st.markdown(
        """
        - 🥉 **Bronze**: 3+ donations  
        - 🥈 **Silver**: 6+ donations  
        - 🥇 **Gold**: 10+ donations  
        - 👑 **Platinum**: 15+ donations
        """
    )

# ---------- Tab 4: Notification Simulation ----------
with tab4:
    st.subheader("Send Donor Notifications (Simulation)")
    st.caption("Simulated SMS/WhatsApp—no real messages sent.")

    c1, c2, c3 = st.columns(3)
    notif_name = c1.text_input("Recipient Name", value="Rohan Sharma")
    notif_phone = c2.text_input("Phone (E.164 / demo)", value="+91XXXXXXXXXX")
    channel = c3.selectbox("Channel", ["SMS", "WhatsApp", "Email"])

    default_msg = (
        "Hi {name}, you're now eligible to donate blood again. "
        "A Thalassemia patient nearby needs your blood type. "
        "Would you like to schedule a donation?"
    ).format(name=notif_name)

    message = st.text_area("Message", value=default_msg, height=120)

    if st.button("Simulate Send"):
        receipt = simulate_notification(notif_name, notif_phone, channel, message)
        st.success(f"✅ Notification queued (simulation).")
        st.json(receipt)

    with st.expander("How to integrate real SMS/WhatsApp later"):
        st.write(
            "- Use providers like Twilio/MSG91/WhatsApp Business API\n"
            "- Store donor opt-ins and respect DND/regulatory compliance\n"
            "- Queue messages via background jobs (Celery/RQ) for scalability"
        )

# ---------- Tab 5: e-RaktKosh Placeholder ----------
with tab5:
    st.subheader("e-RaktKosh Integration (Demo)")
    st.caption("Placeholder to demonstrate how the app would fetch data.")

    req_id = st.text_input("Enter Request/Case ID (demo)", value="ERK-CH-2025-0001")
    if st.button("Fetch Case (Demo)"):
        # Simulated response
        fake = {
            "request_id": req_id,
            "patient_name": "Maya",
            "blood_group": "B+",
            "urgency": "High",
            "hospital": "District Hospital, Chandrapur",
            "required_units": 2,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.success("Fetched (demo).")
        st.json(fake)

    with st.expander("Integration Notes"):
        st.write(
            "In production, we would:\n"
            "- Use official e-RaktKosh APIs (subject to access/approvals)\n"
            "- Map request -> nearest eligible donors\n"
            "- Log fulfillment & audit trails"
        )

st.divider()
st.markdown(
    "###### Privacy Note: This demo uses synthetic/sample data. Do not upload real personal health information (PHI)."
)
