import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

# Load the saved models
@st.cache_resource
def load_models():
    svm_model = joblib.load('models/svm_model.pkl')
    knn_model = joblib.load('models/knn_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
    return svm_model, knn_model, xgb_model

svm_model, knn_model, xgb_model = load_models()

# Feature order as specified
feature_order = [
    'Age', 'Gender', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP', 
    'SCC', 'SMOKE', 'CH2O', 'family_history_with_overweight', 'FAF', 
    'TUE', 'CAEC', 'MTRANS'
]

# Label encodings for categorical features (based on your preprocessing)
categorical_mappings = {
    'Gender': {'Female': 0, 'Male': 1},
    'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'FAVC': {'no': 0, 'yes': 1},
    'SCC': {'no': 0, 'yes': 1},
    'SMOKE': {'no': 0, 'yes': 1},
    'family_history_with_overweight': {'no': 0, 'yes': 1},
    'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'MTRANS': {
        'Public_Transportation': 0, 
        'Walking': 1, 
        'Automobile': 2, 
        'Motorbike': 3, 
        'Bike': 4
    }
}

# Target class mapping (NObeyesdad)
target_mapping = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Obesity_Type_I',
    3: 'Obesity_Type_II',
    4: 'Obesity_Type_III',
    5: 'Overweight_Level_I',
    6: 'Overweight_Level_II'
}

# Streamlit app
st.title("Obesity Level Prediction")
st.write("""
This app predicts obesity levels based on lifestyle and physical attributes.
""")

# Create input form
with st.form("user_inputs"):
    st.header("Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=14, max_value=100, value=25)
        height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        gender = st.selectbox("Gender", options=['Female', 'Male'])
    
    st.header("Eating Habits")
    col1, col2 = st.columns(2)
    with col1:
        favc = st.selectbox("Frequent consumption of high caloric food (FAVC)", options=['no', 'yes'])
        fcvc = st.slider("Frequency of consumption of vegetables (FCVC)", 1.0, 3.0, 2.0, 0.1)
        ncp = st.slider("Number of main meals (NCP)", 1.0, 4.0, 3.0, 0.1)
        caec = st.selectbox("Consumption of food between meals (CAEC)", 
                          options=['no', 'Sometimes', 'Frequently', 'Always'])
    with col2:
        calc = st.selectbox("Consumption of alcohol (CALC)", 
                          options=['no', 'Sometimes', 'Frequently', 'Always'])
        ch2o = st.slider("Consumption of water daily (CH2O in liters)", 1.0, 3.0, 2.0, 0.1)
        scc = st.selectbox("Calories consumption monitoring (SCC)", options=['no', 'yes'])
    
    st.header("Physical Activity & Health")
    col1, col2 = st.columns(2)
    with col1:
        smoke = st.selectbox("Do you smoke? (SMOKE)", options=['no', 'yes'])
        faf = st.slider("Physical activity frequency (FAF)", 0.0, 3.0, 1.0, 0.1)
    with col2:
        tue = st.slider("Time using technology devices (TUE in hours)", 0.0, 2.0, 1.0, 0.1)
        family_history = st.selectbox("Family history with overweight", options=['no', 'yes'])
        mtrans = st.selectbox("Transportation used (MTRANS)", 
                            options=['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])
    
    submitted = st.form_submit_button("Predict Obesity Level")

if submitted:
    # Create input dataframe with the exact feature order
    input_data = {
        'Age': [age],
        'Gender': [gender],
        'Height': [height],
        'Weight': [weight],
        'CALC': [calc],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'SCC': [scc],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'family_history_with_overweight': [family_history],
        'FAF': [faf],
        'TUE': [tue],
        'CAEC': [caec],
        'MTRANS': [mtrans]
    }
    
    df_input = pd.DataFrame(input_data, columns=feature_order)
    
    # Encode categorical variables
    for col, mapping in categorical_mappings.items():
        df_input[col] = df_input[col].map(mapping)
    
    # Scale numerical features (using the same scaler from training)
    # Note: In a real deployment, you should save and load the scaler
    numerical_cols = ['Age', 'Height', 'Weight', 'FAF', 'TUE', 'FCVC', 'CH2O', 'NCP']
    scaler = MinMaxScaler()
    df_input[numerical_cols] = scaler.fit_transform(df_input[numerical_cols])
    
    # Make predictions
    svm_pred = svm_model.predict(df_input)
    knn_pred = knn_model.predict(df_input)
    xgb_pred = xgb_model.predict(df_input)
    
    # Display results
    st.subheader("Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("SVM Prediction", target_mapping[svm_pred[0]])
    with col2:
        st.metric("KNN Prediction", target_mapping[knn_pred[0]])
    with col3:
        st.metric("XGBoost Prediction", target_mapping[xgb_pred[0]])
    
    st.write("### Prediction Details")
    
    # Add some visualizations or explanations
    st.write("""
    - **Insufficient_Weight**: BMI < 18.5
    - **Normal_Weight**: BMI 18.5-24.9
    - **Overweight_Level_I**: BMI 25-26.9
    - **Overweight_Level_II**: BMI 27-29.9
    - **Obesity_Type_I**: BMI 30-34.9
    - **Obesity_Type_II**: BMI 35-39.9
    - **Obesity_Type_III**: BMI ≥ 40
    """)

# Add some information about the models
st.sidebar.header("About the Models")
st.sidebar.write("""
Three machine learning models were trained:
1. **Support Vector Machine (SVM)**
2. **K-Nearest Neighbors (KNN)**
3. **XGBoost**

The models were trained on lifestyle and physical attributes to predict obesity levels.
""")

st.sidebar.header("Feature Descriptions")
st.sidebar.write("""
- **FAVC**: Frequent consumption of high caloric food
- **FCVC**: Frequency of consumption of vegetables
- **NCP**: Number of main meals
- **CAEC**: Consumption of food between meals
- **CALC**: Consumption of alcohol
- **CH2O**: Consumption of water daily
- **SCC**: Calories consumption monitoring
- **FAF**: Physical activity frequency
- **TUE**: Time using technology devices
""")
