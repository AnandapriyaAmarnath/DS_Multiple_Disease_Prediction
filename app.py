import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Multiple Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Multiple Disease Prediction System")
st.write("Predict Diabetes, Heart Disease, and Parkinson's Disease using Machine Learning.")

# --------------------------------------------------
# Load datasets
# --------------------------------------------------

diabetes = pd.read_csv("datasets/diabetes.csv")
heart = pd.read_csv("datasets/heart.csv")
parkinsons = pd.read_csv("datasets/parkinsons.csv")

# --------------------------------------------------
# Data preprocessing
# --------------------------------------------------

# Heart dataset - fill missing numeric values with median
heart.fillna(heart.median(numeric_only=True), inplace=True)

# Fill categorical missing values with mode
for col in heart.select_dtypes(include="object").columns:
    heart[col] = heart[col].fillna(heart[col].mode()[0])

# Convert Heart target into binary
# 0 = No Disease
# 1 = Disease
heart["num"] = heart["num"].apply(lambda x: 0 if x == 0 else 1)

# Parkinson's dataset
# Remove name column if present
if "name" in parkinsons.columns:
    parkinsons.drop("name", axis=1, inplace=True)

# --------------------------------------------------
# Train Diabetes Model
# --------------------------------------------------

X_diabetes = diabetes.drop("Outcome", axis=1)
y_diabetes = diabetes["Outcome"]

diabetes_model = RandomForestClassifier(random_state=42)
diabetes_model.fit(X_diabetes, y_diabetes)

# --------------------------------------------------
# Train Heart Disease Model
# --------------------------------------------------

heart_features = heart.drop("num", axis=1)

# Remove ID column
if "id" in heart_features.columns:
    heart_features = heart_features.drop("id", axis=1)

# Convert categorical columns into numeric columns
heart_encoded = pd.get_dummies(heart_features, drop_first=True)

X_heart = heart_encoded
y_heart = heart["num"]

heart_model = RandomForestClassifier(random_state=42)
heart_model.fit(X_heart, y_heart)

# --------------------------------------------------
# Train Parkinson's Model
# --------------------------------------------------

X_parkinsons = parkinsons.drop("status", axis=1)
y_parkinsons = parkinsons["status"]

parkinsons_model = RandomForestClassifier(random_state=42)
parkinsons_model.fit(X_parkinsons, y_parkinsons)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Disease Prediction")

menu = st.sidebar.radio(
    "Select Disease",
    [
        "Home",
        "Diabetes Prediction",
        "Heart Disease Prediction",
        "Parkinson's Prediction"
    ]
)

# --------------------------------------------------
# Home
# --------------------------------------------------

if menu == "Home":

    st.header("Welcome to Multiple Disease Prediction")

    st.write(
        "This application uses Machine Learning models "
        "to predict three different diseases."
    )

    st.write("### Diseases covered:")

    st.write("🩸 Diabetes")
    st.write("❤️ Heart Disease")
    st.write("🧠 Parkinson's Disease")

    st.info(
        "This application is for educational purposes "
        "and should not be used as a medical diagnosis."
    )

# --------------------------------------------------
# Diabetes Prediction
# --------------------------------------------------

elif menu == "Diabetes Prediction":

    st.header("🩸 Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0,
            max_value=250,
            value=120
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0,
            max_value=200,
            value=70
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0,
            max_value=100,
            value=20
        )

    with col2:
        insulin = st.number_input(
            "Insulin",
            min_value=0,
            max_value=900,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    if st.button("Predict Diabetes"):

        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age
            ]],
            columns=X_diabetes.columns
        )

        prediction = diabetes_model.predict(input_data)

        if prediction[0] == 1:
            st.error("Prediction: Diabetes detected")
        else:
            st.success("Prediction: No Diabetes detected")

# --------------------------------------------------
# Heart Disease Prediction
# --------------------------------------------------

elif menu == "Heart Disease Prediction":

    st.header("❤️ Heart Disease Prediction")

    st.write("Enter the patient's information:")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        dataset = st.selectbox(
            "Dataset",
            heart["dataset"].unique().tolist()
        )

        cp = st.selectbox(
            "Chest Pain Type",
            heart["cp"].unique().tolist()
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=0.0,
            max_value=250.0,
            value=120.0
        )

        chol = st.number_input(
            "Cholesterol",
            min_value=0.0,
            max_value=700.0,
            value=200.0
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar",
            heart["fbs"].unique().tolist()
        )

        restecg = st.selectbox(
            "Resting ECG",
            heart["restecg"].unique().tolist()
        )

    with col2:

        thalch = st.number_input(
            "Maximum Heart Rate",
            min_value=0.0,
            max_value=250.0,
            value=150.0
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            heart["exang"].unique().tolist()
        )

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=0.0,
            max_value=10.0,
            value=1.0
        )

        slope = st.selectbox(
            "Slope",
            heart["slope"].unique().tolist()
        )

        ca = st.number_input(
            "CA",
            min_value=0.0,
            max_value=4.0,
            value=0.0
        )

        thal = st.selectbox(
            "Thal",
            heart["thal"].unique().tolist()
        )

    if st.button("Predict Heart Disease"):

        input_data = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "dataset": [dataset],
            "cp": [cp],
            "trestbps": [trestbps],
            "chol": [chol],
            "fbs": [fbs],
            "restecg": [restecg],
            "thalch": [thalch],
            "exang": [exang],
            "oldpeak": [oldpeak],
            "slope": [slope],
            "ca": [ca],
            "thal": [thal]
        })

        input_encoded = pd.get_dummies(
            input_data,
            drop_first=True
        )

        # Match training columns
        input_encoded = input_encoded.reindex(
            columns=heart_encoded.columns,
            fill_value=0
        )

        prediction = heart_model.predict(input_encoded)

        if prediction[0] == 1:
            st.error("Prediction: Heart Disease detected")
        else:
            st.success("Prediction: No Heart Disease detected")

# --------------------------------------------------
# Parkinson's Prediction
# --------------------------------------------------

elif menu == "Parkinson's Prediction":

    st.header("🧠 Parkinson's Disease Prediction")

    st.write("Enter the required voice measurement values:")

    input_values = []

    columns = X_parkinsons.columns

    for i, column in enumerate(columns):

        input_value = st.number_input(
            column,
            value=float(X_parkinsons[column].mean())
        )

        input_values.append(input_value)

    if st.button("Predict Parkinson's Disease"):

        input_data = pd.DataFrame(
            [input_values],
            columns=columns
        )

        prediction = parkinsons_model.predict(input_data)

        if prediction[0] == 1:
            st.error("Prediction: Parkinson's Disease detected")
        else:
            st.success("Prediction: No Parkinson's Disease detected")