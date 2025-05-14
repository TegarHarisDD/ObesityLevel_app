import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load models
svm_model = joblib.load('models/svm_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')

# Load label encoder for target
le_target = LabelEncoder()
le_target.classes_ = np.load('classes.npy', allow_pickle=True)  # Save this during training

# Feature order (must match training)
feature_names = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'
]

# Helper for categorical encoding (should match training encoding)
def encode_input(input_dict):
    # Map categorical values to integers as in training
    cat_maps = {
        'Gender': {'Female': 0, 'Male': 1},
        'family_history_with_overweight': {'no': 0, 'yes': 1},
        'FAVC': {'no': 0, 'yes': 1},
        'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
        'SMOKE': {'no': 0, 'yes': 1},
        'SCC': {'no': 0, 'yes': 1},
        'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
        'MTRANS': {'Automobile': 0, 'Motorbike': 1, 'Bike': 2, 'Public_Transportation': 3, 'Walking': 4}
    }
    encoded = []
    for col in feature_names:
        if col in cat_maps:
            encoded.append(cat_maps[col][input_dict[col]])
        else:
            encoded.append(float(input_dict[col]))
    return np.array(encoded).reshape(1, -1)

st.title("Obesity Classification Prediction")

# Input fields
input_data = {}
input_data['Gender'] = st.selectbox('Gender', ['Female', 'Male'])
input_data['Age'] = st.number_input('Age', min_value=1, max_value=100, value=25)
input_data['Height'] = st.number_input('Height (in meters)', min_value=1.0, max_value=2.5, value=1.7, step=0.01)
input_data['Weight'] = st.number_input('Weight (in kg)', min_value=10.0, max_value=200.0, value=70.0, step=0.1)
input_data['family_history_with_overweight'] = st.selectbox('Family History with Overweight', ['no', 'yes'])
input_data['FAVC'] = st.selectbox('Frequent Consumption of High Caloric Food (FAVC)', ['no', 'yes'])
input_data['FCVC'] = st.slider('Frequency of Vegetable Consumption (FCVC)', 1.0, 3.0, 2.0, 0.1)
input_data['NCP'] = st.slider('Number of Main Meals (NCP)', 1.0, 4.0, 3.0, 0.1)
input_data['CAEC'] = st.selectbox('Consumption of Food Between Meals (CAEC)', ['no', 'Sometimes', 'Frequently', 'Always'])
input_data['SMOKE'] = st.selectbox('Do you smoke?', ['no', 'yes'])
input_data['CH2O'] = st.slider('Daily Water Intake (CH2O)', 1.0, 3.0, 2.0, 0.1)
input_data['SCC'] = st.selectbox('Calories Consumption Monitoring (SCC)', ['no', 'yes'])
input_data['FAF'] = st.slider('Physical Activity Frequency (FAF)', 0.0, 3.0, 1.0, 0.1)
input_data['TUE'] = st.slider('Time using Technology Devices (TUE)', 0.0, 2.0, 1.0, 0.1)
input_data['CALC'] = st.selectbox('Alcohol Consumption (CALC)', ['no', 'Sometimes', 'Frequently', 'Always'])
input_data['MTRANS'] = st.selectbox('Transportation used', ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'])

if st.button('Predict'):
    X_input = encode_input(input_data)
    pred_svm = le_target.inverse_transform(svm_model.predict(X_input))[0]
    pred_knn = le_target.inverse_transform(knn_model.predict(X_input))[0]
    pred_xgb = le_target.inverse_transform(xgb_model.predict(X_input))[0]

    st.success(f"SVM Prediction: {pred_svm}")
    st.success(f"KNN Prediction: {pred_knn}")
    st.success(f"XGBoost Prediction: {pred_xgb}")
