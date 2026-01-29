import streamlit as st
import pandas as pd
import joblib
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Luminar Main Project",
    page_icon="🏏",
    layout="wide"
)

# ================= GLOBAL STYLES =================
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}
.title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    color: #0f172a;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #475569;
    margin-bottom: 30px;
}
.card {
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
}
.win {
    background: linear-gradient(135deg, #dcfce7, #ecfeff);
}
.lose {
    background: linear-gradient(135deg, #ffedd5, #fff7ed);
}
.conf-badge {
    display: inline-block;
    padding: 10px 20px;
    border-radius: 999px;
    background-color: #0ea5e9;
    color: white;
    font-weight: 700;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# ================= BASE DIRECTORY =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= LOAD RESOURCES =================
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "ra_model.pkl"))

@st.cache_resource
def load_encoder():
    return joblib.load(os.path.join(BASE_DIR, "encoder.pkl"))

@st.cache_resource
def load_feature_columns():
    return joblib.load(os.path.join(BASE_DIR, "model_columns.pkl"))

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "real_cric2.csv"))

model = load_model()
encoder = load_encoder()
feature_columns = load_feature_columns()
df = load_data()

# ================= FORM MAPPING =================
FORM_MAPPING = {
    "Poor": 40,
    "Average": 50,
    "Good": 60,
    "Very Good": 70,
    "Excellent": 80
}

# ================= SIDEBAR =================
st.sidebar.title("🏏 Cricket ML Project")
menu = st.sidebar.radio(
    "Navigation",
    ["Project Overview", "Dataset Preview", "Predict Match Winner"]
)

# ================= HEADER =================
st.markdown("<div class='title'>🏏  Cricket Pre Match Winner Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Pre-match ML system using Random Forest</div>", unsafe_allow_html=True)
st.divider()

# =================================================
# 📌 PROJECT OVERVIEW
# =================================================
if menu == "Project Overview":
    st.subheader("📌 About the Project")
    st.markdown("""
    - Predicts **match winner before the match starts**
    - Uses **Random Forest Classifier**
    - Categorical variables encoded using **OneHotEncoder**
    - Same preprocessing applied during deployment
    - Professional frontend built using **Streamlit**
    """)

# =================================================
# 📊 DATASET PREVIEW
# =================================================
elif menu == "Dataset Preview":
    st.subheader("📊 Dataset Preview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Features", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.head(10), use_container_width=True)

# =================================================
# 🔮 PREDICTION
# =================================================
else:
    st.subheader("🔮 Predict Match Winner")
    st.markdown("### Enter pre-match details")

    input_data = {}
    raw_features = df.drop("team1_win", axis=1)

    for col in raw_features.columns:

        if col == "toss_winner":
            choice = st.selectbox("Toss Winner", ["Team 1", "Team 2"])
            input_data[col] = 1 if choice == "Team 1" else 0

        elif col == "toss_decision_bat":
            choice = st.selectbox("Toss Decision", ["Bat", "Bowl"])
            input_data[col] = 1 if choice == "Bat" else 0

        elif col in ["team1_key_player_form", "team2_key_player_form"]:
            choice = st.selectbox(
                col.replace("_", " ").title(),
                FORM_MAPPING.keys()
            )
            input_data[col] = FORM_MAPPING[choice]

        elif raw_features[col].dtype in ["int64", "float64"]:
            input_data[col] = st.number_input(
                col,
                value=float(raw_features[col].mean())
            )

        else:
            input_data[col] = st.selectbox(
                col,
                sorted(raw_features[col].unique())
            )

    if st.button("🏏 Predict Winner"):
        input_df = pd.DataFrame([input_data])

        encoded_input = encoder.transform(input_df)
        encoded_df = pd.DataFrame(
            encoded_input,
            columns=encoder.get_feature_names_out()
        )

        encoded_df = encoded_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = model.predict(encoded_df)[0]
        confidence = model.predict_proba(encoded_df).max()
        confidence_pct = int(confidence * 100)

        st.markdown("## 🏆 Match Prediction Result")

        colA, colB = st.columns([2, 1])

        with colA:
            if prediction == 1:
                st.markdown(f"""
                <div class="card win">
                    <h1>🏆 TEAM 1 WILL WIN</h1>
                    <p>Model analysis favors Team 1</p>
                    <div class="conf-badge">{confidence_pct}% Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="card lose">
                    <h1>🏆 TEAM 2 WILL WIN</h1>
                    <p>Model analysis favors Team 2</p>
                    <div class="conf-badge">{confidence_pct}% Confidence</div>
                </div>
                """, unsafe_allow_html=True)

        with colB:
            st.markdown("### 🔍 Confidence Meter")
            st.progress(confidence_pct)
            st.metric(
                "Prediction Strength",
                f"{confidence_pct}%",
                "High" if confidence_pct > 65 else "Moderate"
            )

# ================= FOOTER =================
st.caption("")
