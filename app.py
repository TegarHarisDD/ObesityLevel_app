import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

# Load models
models = {
    "SVM": joblib.load("models/svm_model.pkl"),
    "KNN": joblib.load("models/knn_model.pkl"),
    "XGBoost": joblib.load("models/xgb_model.pkl")
}

# Label encoders (if needed for categorical features)
# Note: Save and load encoders used during training if necessary.

def preprocess_input(input_df, is_bulk=False):
    """Preprocess input data to match training format."""
    # Encode categorical features (same as during training)
    categorical_cols = {
        'Gender': {'Female': 0, 'Male': 1},
        'family_history': {'No': 0, 'Yes': 1},
        'FAVC': {'No': 0, 'Yes': 1},
        'SMOKE': {'No': 0, 'Yes': 1},
        'SCC': {'No': 0, 'Yes': 1},
        'CALC': {'No': 0, 'Sometimes': 1, 'Frequently': 2},
        'CAEC': {'No': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
        'MTRANS': {'Public': 0, 'Walking': 1, 'Bike': 2, 'Motorbike': 3, 'Automobile': 4}
    }
    
    for col, mapping in categorical_cols.items():
        input_df[col] = input_df[col].map(mapping)
    
    # Scale numerical features
    scaler = MinMaxScaler()
    numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    input_df[numeric_cols] = scaler.fit_transform(input_df[numeric_cols])
    
    # Ensure column order matches training data
    expected_columns = [
        'Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC', 'FCVC', 
        'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'CAEC', 'MTRANS'
    ]
    return input_df[expected_columns]

def predict(model, data):
    """Run predictions and return results."""
    return model.predict(data)

# Streamlit UI
st.title("Obesity Level Prediction")
st.write("Choose a model and input method:")

# Model selection
model_name = st.selectbox("Select Model", list(models.keys()))

# Input method: Form or File Upload
input_method = st.radio("Input Method", ["Form", "File Upload"])

if input_method == "Form":
    st.header("Single Input Form")
    with st.form("input_form"):
        # Numerical Features
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7, step=0.01)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        fcvc = st.number_input("Frequency of veg consumption (1-3)", min_value=1, max_value=3, value=2)
        ncp = st.number_input("Number of main meals (1-4)", min_value=1, max_value=4, value=3)
        ch2o = st.number_input("Daily water intake (L) (1-3)", min_value=1, max_value=3, value=2)
        faf = st.number_input("Physical activity frequency (0-3)", min_value=0, max_value=3, value=1)
        tue = st.number_input("Screen time (hours/day) (0-2)", min_value=0, max_value=2, value=1)
        
        # Categorical Features (use radio/selectbox)
        gender = st.radio("Gender", ["Female", "Male"])
        family_history = st.radio("Family history of overweight", ["Yes", "No"])
        favc = st.radio("High-caloric food frequent", ["Yes", "No"])
        smoke = st.radio("Smoker?", ["Yes", "No"])
        scc = st.radio("Calories monitoring", ["Yes", "No"])
        calc = st.selectbox("Alcohol consumption", ["No", "Sometimes", "Frequently"])
        caec = st.selectbox("Food between meals", ["No", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Transportation used", ["Public", "Walking", "Bike", "Motorbike", "Automobile"])
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            input_data = pd.DataFrame([[
                gender, age, height, weight, family_history, favc, fcvc, 
                ncp, smoke, ch2o, scc, faf, tue, calc, caec, mtrans
            ]], columns=[
                'Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC', 'FCVC', 
                'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'CAEC', 'MTRANS'
            ])
            
            input_processed = preprocess_input(input_data)
            prediction = predict(models[model_name], input_processed)
            st.success(f"Predicted Class: {prediction[0]}")
else:
    st.header("Bulk File Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())
        
        if st.button("Predict Bulk"):
            df_processed = preprocess_input(df.copy(), is_bulk=True)
            predictions = predict(models[model_name], df_processed)
            df["Predicted_Obesity_Class"] = predictions
            st.write("Results:", df)
            
            # Download results
            st.download_button(
                label="Download Predictions",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="obesity_predictions.csv",
                mime="text/csv"
            )
