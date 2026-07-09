import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

model = joblib.load("model.pkl")

scaler = joblib.load("scaler.pkl")

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide"
)

st.markdown("""
<h1>🏭 Predictive Maintenance Dashboard</h1>

<div style='text-align:center;
font-size:18px;
color:lightgray;'>

AI4I 2020 Dataset • Random Forest Classifier • Industrial Machine Failure Prediction

</div>

""", unsafe_allow_html=True)


# ----------------------------------------------------
# Custom CSS
# ----------------------------------------------------

st.markdown("""
<style>

/* Main background */

.stApp {
    background-color: #0E1117;
}

/* Title */
h1 {
    color: #4CAF50;
    text-align: center;
    font-weight: bold;
}

/* Section headings */
h2, h3 {
    color: #00BFFF;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1E1E2F;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg,#2563eb,#7c3aed);
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size:18px;
    font-weight:bold;
    border:none;
}

.stButton > button:hover{
    background: linear-gradient(90deg,#1d4ed8,#6d28d9);
}

/* Metric cards */
[data-testid="metric-container"]{
    background-color:#202633;
    border-radius:12px;
    padding:15px;
    border:1px solid #404040;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.metric(
    "Dataset",
    "AI4I 2020 Predictive Maintenance"
)

st.sidebar.metric(
    "Algorithm",
    "Random Forest"
)

st.sidebar.metric(
    "Accuracy",
    "99.1%"
)

st.sidebar.write("### 👨‍💻 Developer")
st.sidebar.write("Sahil Ramzan Bhat")

st.sidebar.markdown("---")

st.sidebar.info(
    "This application predicts whether an industrial machine is likely to fail based on operating conditions."
)

# ----------------------------------------------------
# Main Title
# ----------------------------------------------------

st.markdown("---")

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        "99.1%"
    )

with col2:
    st.metric(
        "Algorithm",
        "Random Forest"
    )

with col3:
    st.metric(
        "Dataset",
        "AI4I 2020"
    )

with col4:
    st.metric(
        "Status",
        "Ready"
    )

st.markdown("---")

# ----------------------------------------------------
# Input Section
# ----------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["Low (L)", "Medium (M)", "High (H)"]
    )

    type_mapping = {

    "High (H)":0,

    "Low (L)":1,

    "Medium (M)":2

    }

    machine_type = type_mapping[machine_type]

    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=295.3,
        max_value=304.5,
        value=300.0,
        step=0.1
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=303.0,
        max_value=315.0,
        value=310.0,
        step=0.1
    )

with col2:

    rpm = st.number_input(
        "Rotational Speed (RPM)",
        min_value=1168.0,
        max_value=2886.0,
        value=1500.0
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=3.8,
        max_value=76.6,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear (minutes)",
        min_value=0.0,
        max_value=253.0,
        value=50.0,
        step=0.1
    )

st.markdown("---")

# ----------------------------------------------------
# Predict Button
# ----------------------------------------------------

predict_button = st.button(
    "🔍 Predict Machine Condition",
    use_container_width=True
)

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

if predict_button:

    sample = [[
        machine_type,
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear
    ]]

    # Prediction
    prediction = model.predict(sample)

    # Prediction Probability
    probability = model.predict_proba(sample)

    healthy_probability = probability[0][0] * 100
    failure_probability = probability[0][1] * 100

    st.markdown("## Prediction Result")

    if prediction[0] == 1:

        st.markdown("""
            <div style='background:#7f1d1d;
            padding:20px;
            border-radius:12px;
            text-align:center;'>

            <h2 style='color:white;'>

            🔴 MACHINE FAILURE PREDICTED

            </h2>

            </div>
            """, unsafe_allow_html=True)

    else:

        st.markdown("""
            <div style='background:#14532d;
            padding:20px;
            border-radius:12px;
            text-align:center;'>

            <h2 style='color:white;'>

            🟢 MACHINE IS OPERATING NORMALLY

            </h2>

            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Prediction Confidence")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Healthy",
            f"{healthy_probability:.2f}%"
        )

    with c2:

        st.metric(
            "Failure",
            f"{failure_probability:.2f}%"
        )

#----------------------------------------------------
# Feature Importance
#----------------------------------------------------

feature_importance = pd.DataFrame({

    "Feature":[

        "Machine Type",
        "Air Temperature",
        "Process Temperature",
        "Rotational Speed",
        "Torque",
        "Tool Wear"

    ],

    "Importance":model.feature_importances_

})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

st.markdown("---")

st.subheader("📊 Feature Importance")

fig = px.bar(

    feature_importance,

    x="Importance",

    y="Feature",

    orientation="h",

    color="Importance",

    color_continuous_scale="Blues"

)

fig.update_layout(

    height=450,

    template="plotly_dark",

    showlegend=False

)

st.plotly_chart(
    fig,
    use_container_width=True
)

#----------------------------------------------------
# About Section
#----------------------------------------------------

with st.expander("ℹ About this Model"):

    st.write("""
    This model predicts industrial machine failures
    using a Random Forest classifier trained on
    the AI4I 2020 Predictive Maintenance dataset.
    """)


st.markdown("---")

st.markdown(
"""
<div style='text-align:center'>

Developed by <b>Sahil Ramzan Bhat</b><br>

Predictive Maintenance using Machine Learning

</div>
""",
unsafe_allow_html=True
)
