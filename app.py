import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Load trained model and scaler
# --------------------------------------------------
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# --------------------------------------------------
# Title and introduction
# --------------------------------------------------
st.title("❤️ Cardiovascular Disease Prediction")

st.write(
    "This machine learning application predicts whether a patient "
    "is likely to have cardiovascular disease based on selected "
    "health and lifestyle information."
)

st.info(
    "The final prediction model is Support Vector Machine (SVM), "
    "which achieved 73.12% accuracy on the test dataset."
)

st.warning(
    "⚠️ This application is developed for educational purposes only "
    "and is not a medical diagnosis."
)


# --------------------------------------------------
# Project information
# --------------------------------------------------
with st.expander("📊 About the Project"):
    st.write(
        "This project uses the Cardiovascular Disease dataset and "
        "applies data preprocessing, exploratory data analysis, "
        "correlation analysis, and machine learning."
    )

    st.write("**Machine Learning Models Compared:**")

    st.write(
        "• Logistic Regression — 72.55%\n"
        "• K-Nearest Neighbors (KNN) — 69.07%\n"
        "• Decision Tree — 72.53%\n"
        "• Random Forest — 70.74%\n"
        "• Support Vector Machine (SVM) — 73.12%"
    )


# --------------------------------------------------
# Patient information
# --------------------------------------------------
st.header("🧑‍⚕️ Patient Information")

st.caption(
    "Enter the patient's information below and click the "
    "prediction button."
)


# Age
age_years = st.number_input(
    "Age (years)",
    min_value=30,
    max_value=65,
    value=50,
    step=1
)


# Gender
gender = st.selectbox(
    "Gender",
    options=[1, 2],
    index=0,
    format_func=lambda x: "Female" if x == 1 else "Male"
)


# Height and weight
col1, col2 = st.columns(2)

with col1:
    height = st.number_input(
        "Height (cm)",
        min_value=120.0,
        max_value=250.0,
        value=165.0,
        step=1.0
    )

with col2:
    weight = st.number_input(
        "Weight (kg)",
        min_value=28.0,
        max_value=200.0,
        value=65.0,
        step=1.0
    )


# Blood pressure
col3, col4 = st.columns(2)

with col3:
    ap_hi = st.number_input(
        "Systolic BP",
        min_value=80,
        max_value=240,
        value=120,
        step=1
    )

with col4:
    ap_lo = st.number_input(
        "Diastolic BP",
        min_value=40,
        max_value=150,
        value=80,
        step=1
    )


# Cholesterol
cholesterol = st.selectbox(
    "Cholesterol Level",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "Normal",
        2: "Above Normal",
        3: "Well Above Normal"
    }[x]
)


# Glucose
gluc = st.selectbox(
    "Glucose Level",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "Normal",
        2: "Above Normal",
        3: "Well Above Normal"
    }[x]
)


# Lifestyle information
col5, col6, col7 = st.columns(3)

with col5:
    smoke = st.selectbox(
        "Smoking",
        options=[0, 1],
        index=0,
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col6:
    alco = st.selectbox(
        "Alcohol",
        options=[0, 1],
        index=0,
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col7:
    active = st.selectbox(
        "Physical Activity",
        options=[0, 1],
        index=1,
        format_func=lambda x: "No" if x == 0 else "Yes"
    )


# --------------------------------------------------
# Calculate BMI
# --------------------------------------------------
bmi = weight / ((height / 100) ** 2)

st.metric(
    label="Calculated BMI",
    value=f"{bmi:.2f}"
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.divider()

if st.button(
    "🔍 Predict Cardiovascular Disease",
    use_container_width=True
):

    # Create input DataFrame
    new_patient = pd.DataFrame([[
        gender,
        height,
        weight,
        ap_hi,
        ap_lo,
        cholesterol,
        gluc,
        smoke,
        alco,
        active,
        age_years,
        round(bmi, 2)
    ]], columns=[
        "gender",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "age_years",
        "bmi"
    ])


    # Scale input
    new_patient_scaled = scaler.transform(new_patient)


    # Make prediction
    prediction = svm_model.predict(new_patient_scaled)


    # Display result
    st.subheader("🎯 Prediction Result")

    if prediction[0] == 1:
        st.error(
            "⚠️ Cardiovascular Disease Detected"
        )

    else:
        st.success(
            "✅ No Cardiovascular Disease Detected"
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Machine Learning Project | Final Model: SVM | Test Accuracy: 73.12%"
)

st.caption(
    "For educational purposes only. Not intended for medical diagnosis."
)