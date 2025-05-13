import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load models
models = {
    "SVM": joblib.load("models/svm_model.pkl"),
    "KNN": joblib.load("models/knn_model.pkl"),
    "XGBoost": joblib.load("models/xgb_model.pkl")
}

# Preprocessing function (replicate your notebook logic)
def preprocess_input(df):
    # Encode categorical features (same as in your notebook)
    categorical_cols = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 
                       'family_history_with_overweight', 'CAEC', 'MTRANS']
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    
    # Normalize numerical features
    numerical_cols = ['Age', 'Height', 'Weight', 'FAF', 'TUE', 'FCVC', 'CH2O', 'NCP']
    scaler = MinMaxScaler()
    if all(col in df.columns for col in numerical_cols):
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

# Streamlit UI
st.title("Obesity Level Prediction")
st.write("Choose a model and input method:")

# Model selection
model_name = st.selectbox("Select Model", ["SVM", "KNN", "XGBoost"])
model = models[model_name]

# Input method (form or file)
input_method = st.radio("Input Method", ["Form", "File Upload"])

if input_method == "Form":
    # Manual input form
    st.subheader("Enter Data Manually")
    age = st.number_input("Age", min_value=0, max_value=100)
    height = st.number_input("Height (m)", min_value=0.0, max_value=3.0)
    weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    # Add other fields similarly...
    
    if st.button("Predict"):
        input_data = pd.DataFrame([[age, height, weight, gender]], 
                                columns=["Age", "Height", "Weight", "Gender"])
        input_processed = preprocess_input(input_data)
        prediction = model.predict(input_processed)
        st.success(f"Predicted Class: {prediction[0]}")

else:
    # File upload
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())
        
        if st.button("Predict Bulk"):
            df_processed = preprocess_input(df.copy())
            predictions = model.predict(df_processed)
            df["Predicted_Obesity_Class"] = predictions
            st.write("Results:", df)
            
            # Download results
            st.download_button(
                label="Download Predictions",
                data=df.to_csv(index=False),
                file_name="obesity_predictions.csv"
            )