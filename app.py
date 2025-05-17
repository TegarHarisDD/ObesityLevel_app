import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load saved preprocessing objects and models
encoders = joblib.load('models/label_encoders.pkl')
scaler = joblib.load('models/minmax_scaler.pkl')
target_encoder = joblib.load('models/target_encoder.pkl')
svm_model = joblib.load('models/svm_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')

# Feature order (must match training data)
features = [
    'Age', 'Gender', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP',
    'SCC', 'SMOKE', 'CH2O', 'family_history_with_overweight', 'FAF', 'TUE',
    'CAEC', 'MTRANS'
]

st.title('Obesity Level Prediction')
st.write('Input the features below to predict obesity level using SVM, KNN, and XGBoost.')

# Input form
with st.form('input_form'):
    inputs = {}
    col1, col2 = st.columns(2)
    
    # Numerical Inputs
    with col1:
        inputs['Age'] = st.number_input('Age (years)', min_value=0.0, max_value=100.0, value=25.0)
        inputs['Height'] = st.number_input('Height (meters)', min_value=0.0, max_value=2.5, value=1.70)
        inputs['Weight'] = st.number_input('Weight (kg)', min_value=0.0, max_value=300.0, value=70.0)
        inputs['FCVC'] = st.number_input('FCVC (frequency)', min_value=0.0, max_value=3.0, value=2.0)
        inputs['NCP'] = st.number_input('NCP (meals)', min_value=0.0, max_value=4.0, value=3.0)
        inputs['CH2O'] = st.number_input('CH2O (liters)', min_value=0.0, max_value=3.0, value=2.0)
        inputs['FAF'] = st.number_input('FAF (activity)', min_value=0.0, max_value=3.0, value=1.0)
        inputs['TUE'] = st.number_input('TUE (hours)', min_value=0.0, max_value=2.0, value=1.0)
    
    # Categorical Inputs
    with col2:
        inputs['Gender'] = st.selectbox('Gender', encoders['Gender'].classes_)
        inputs['CALC'] = st.selectbox('CALC', encoders['CALC'].classes_)
        inputs['FAVC'] = st.selectbox('FAVC', encoders['FAVC'].classes_)
        inputs['SCC'] = st.selectbox('SCC', encoders['SCC'].classes_)
        inputs['SMOKE'] = st.selectbox('SMOKE', encoders['SMOKE'].classes_)
        inputs['family_history_with_overweight'] = st.selectbox('Family History', encoders['family_history_with_overweight'].classes_)
        inputs['CAEC'] = st.selectbox('CAEC', encoders['CAEC'].classes_)
        inputs['MTRANS'] = st.selectbox('MTRANS', encoders['MTRANS'].classes_)
    
    submitted = st.form_submit_button('Predict')

if submitted:
    # Create DataFrame
    input_df = pd.DataFrame([inputs], columns=features)
    
    # Encode categorical features
    for col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])
    
    # Scale numerical features
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Predict
    svm_pred = target_encoder.inverse_transform(svm_model.predict(input_df))[0]
    knn_pred = target_encoder.inverse_transform(knn_model.predict(input_df))[0]
    xgb_pred = target_encoder.inverse_transform(xgb_model.predict(input_df))[0]
    
    # Display results
    st.subheader('Predictions')
    st.write(f'SVM Prediction: **{svm_pred}**')
    st.write(f'KNN Prediction: **{knn_pred}**')
    st.write(f'XGBoost Prediction: **{xgb_pred}**')
