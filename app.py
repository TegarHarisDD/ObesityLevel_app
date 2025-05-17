import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load saved preprocessing objects and models
try:
    # Load saved preprocessing objects and models
    encoders = joblib.load('models/label_encoders.pkl')
    scaler = joblib.load('models/minmax_scaler.pkl')
    target_encoder = joblib.load('models/target_encoder.pkl')
    svm_model = joblib.load('models/svm_model.pkl')
    knn_model = joblib.load('models/knn_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Feature order (MUST match training order exactly)
features = [
    'Age', 'Gender', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP',
    'SCC', 'SMOKE', 'CH2O', 'family_history_with_overweight', 'FAF', 'TUE',
    'CAEC', 'MTRANS'
]

# Numerical columns that need scaling (must match training)
numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

st.title('Obesity Level Prediction')

with st.form('input_form'):
    inputs = {}
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['Age'] = st.number_input('Age (years)', min_value=0.0, max_value=100.0, value=25.0)
        inputs['Height'] = st.number_input('Height (meters)', min_value=0.0, max_value=2.5, value=1.70)
        inputs['Weight'] = st.number_input('Weight (kg)', min_value=0.0, max_value=300.0, value=70.0)
        inputs['FCVC'] = st.number_input('Frequency of veg consumption (1-3)', min_value=1.0, max_value=3.0, value=2.0)
        inputs['NCP'] = st.number_input('Number of main meals (1-4)', min_value=1.0, max_value=4.0, value=3.0)
        inputs['CH2O'] = st.number_input('Water consumption (1-3)', min_value=1.0, max_value=3.0, value=2.0)
        inputs['FAF'] = st.number_input('Physical activity (0-3)', min_value=0.0, max_value=3.0, value=1.0)
        inputs['TUE'] = st.number_input('Screen time (0-2)', min_value=0.0, max_value=2.0, value=1.0)
    
    with col2:
        inputs['Gender'] = st.selectbox('Gender', encoders['Gender'].classes_)
        inputs['CALC'] = st.selectbox('Alcohol consumption', encoders['CALC'].classes_)
        inputs['FAVC'] = st.selectbox('High caloric food', encoders['FAVC'].classes_)
        inputs['SCC'] = st.selectbox('Calories monitoring', encoders['SCC'].classes_)
        inputs['SMOKE'] = st.selectbox('Smoking', encoders['SMOKE'].classes_)
        inputs['family_history_with_overweight'] = st.selectbox('Family history', encoders['family_history_with_overweight'].classes_)
        inputs['CAEC'] = st.selectbox('Food between meals', encoders['CAEC'].classes_)
        inputs['MTRANS'] = st.selectbox('Transportation', encoders['MTRANS'].classes_)
    
    submitted = st.form_submit_button('Predict')

if submitted:
    try:
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([inputs])[features]
        
        # Encode categorical features
        for col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])
        
        # Scale numerical features - ensure we only scale the columns that were scaled during training
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # Predict
        svm_pred = target_encoder.inverse_transform(svm_model.predict(input_df))[0]
        knn_pred = target_encoder.inverse_transform(knn_model.predict(input_df))[0]
        xgb_pred = target_encoder.inverse_transform(xgb_model.predict(input_df))[0]
        
        # Display results
        st.success('Predictions:')
        st.markdown(f"- **SVM**: {svm_pred}")
        st.markdown(f"- **KNN**: {knn_pred}")
        st.markdown(f"- **XGBoost**: {xgb_pred}")
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.error("Please check that all input values are valid.")
