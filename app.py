import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load models
svm_model = joblib.load('models/svm_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')

# Feature order (must match training data)
FEATURE_ORDER = [
    'Age', 'Gender', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP',
    'SCC', 'SMOKE', 'CH2O', 'family_history_with_overweight', 'FAF', 'TUE',
    'CAEC', 'MTRANS'
]

# Label encoders for categorical features (recreate as in training)
def encode_input(data):
    # Categorical features and their possible values
    categorical_mapping = {
        'Gender': ['Female', 'Male'],
        'CALC': ['no', 'Sometimes', 'Frequently', 'Always'],
        'FAVC': ['no', 'yes'],
        'SCC': ['no', 'yes'],
        'SMOKE': ['no', 'yes'],
        'family_history_with_overweight': ['no', 'yes'],
        'CAEC': ['no', 'Sometimes', 'Frequently', 'Always'],
        'MTRANS': ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike']
    }
    
    encoded_data = data.copy()
    for col, categories in categorical_mapping.items():
        le = LabelEncoder()
        le.fit(categories)
        encoded_data[col] = le.transform([encoded_data[col]])[0]
    return encoded_data

# MinMaxScaler (recreate as in training)
def scale_input(data, numeric_cols):
    scaler = MinMaxScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

# Streamlit UI
st.title("Obesity Level Prediction")
st.write("Enter your details to predict obesity level using SVM, KNN, and XGBoost.")

# Input form
with st.form("user_input"):
    st.subheader("Personal Information")
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)

    st.subheader("Habits")
    calc = st.selectbox("Alcohol Consumption (CALC)", ["no", "Sometimes", "Frequently", "Always"])
    favc = st.selectbox("High-Calorie Food (FAVC)", ["no", "yes"])
    fcvc = st.number_input("Vegetable Intake (FCVC)", min_value=1, max_value=3, value=2)
    ncp = st.number_input("Meals per Day (NCP)", min_value=1, max_value=4, value=3)
    scc = st.selectbox("Calorie Monitoring (SCC)", ["no", "yes"])
    smoke = st.selectbox("Smoking (SMOKE)", ["no", "yes"])
    ch2o = st.number_input("Water Intake (CH2O)", min_value=1, max_value=3, value=2)
    family_history = st.selectbox("Family History (Overweight)", ["no", "yes"])
    faf = st.number_input("Physical Activity (FAF)", min_value=0, max_value=3, value=1)
    tue = st.number_input("Screen Time (TUE)", min_value=0, max_value=2, value=1)
    caec = st.selectbox("Snacking (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transportation (MTRANS)", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    # Create input DataFrame
    input_data = {
        'Age': age,
        'Gender': gender,
        'Height': height,
        'Weight': weight,
        'CALC': calc,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'SCC': scc,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'family_history_with_overweight': family_history,
        'FAF': faf,
        'TUE': tue,
        'CAEC': caec,
        'MTRANS': mtrans
    }
    
    # Convert to DataFrame and reorder columns
    input_df = pd.DataFrame([input_data])[FEATURE_ORDER]
    
    # Encode categorical features
    encoded_df = encode_input(input_df)
    
    # Scale numeric features (same cols as in training)
    numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    scaled_df = scale_input(encoded_df.copy(), numeric_cols)
    
    # Predict
    svm_pred = svm_model.predict(scaled_df)[0]
    knn_pred = knn_model.predict(scaled_df)[0]
    xgb_pred = xgb_model.predict(scaled_df)[0]
    
    # Map encoded predictions back to labels
    target_labels = [
        'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
        'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
    ]
    
    # Display results
    st.subheader("Prediction Results")
    st.write(f"- **SVM Prediction:** {target_labels[svm_pred]}")
    st.write(f"- **KNN Prediction:** {target_labels[knn_pred]}")
    st.write(f"- **XGBoost Prediction:** {target_labels[xgb_pred]}")
