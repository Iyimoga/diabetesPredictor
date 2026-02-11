import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .healthy {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .prediabetic {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .diabetic {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏥 Diabetes Predictor</h1>', unsafe_allow_html=True)

# Load or train model
@st.cache_resource
def load_model():
    """Load the trained model or train a new one if not available"""
    try:
        with open('diabetes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        st.warning("Model file not found. Please ensure the model is trained first.")
        return None

model = load_model()

# Sidebar for input
st.sidebar.header("Patient Information")
st.sidebar.markdown("Enter the patient's medical data below:")

# Input fields
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 1, 100, 45)

st.sidebar.subheader("Laboratory Values")
urea = st.sidebar.number_input("Urea (mmol/L)", 0.0, 20.0, 5.0, 0.1, help="Normal range: 2.5-7.1 mmol/L")
cr = st.sidebar.number_input("Creatinine (μmol/L)", 0, 200, 70, 1, help="Normal range: 60-110 μmol/L")
hba1c = st.sidebar.number_input("HbA1c (%)", 0.0, 15.0, 5.5, 0.1, help="Normal: <5.7%, Prediabetes: 5.7-6.4%, Diabetes: ≥6.5%")

st.sidebar.subheader("Lipid Profile")
chol = st.sidebar.number_input("Total Cholesterol (mmol/L)", 0.0, 15.0, 5.0, 0.1, help="Normal: <5.2 mmol/L")
tg = st.sidebar.number_input("Triglycerides (mmol/L)", 0.0, 10.0, 1.5, 0.1, help="Normal: <1.7 mmol/L")
hdl = st.sidebar.number_input("HDL (mmol/L)", 0.0, 5.0, 1.2, 0.1, help="Normal: >1.0 (M), >1.3 (F) mmol/L")
ldl = st.sidebar.number_input("LDL (mmol/L)", 0.0, 10.0, 3.0, 0.1, help="Normal: <3.4 mmol/L")
vldl = st.sidebar.number_input("VLDL (mmol/L)", 0.0, 5.0, 0.7, 0.1, help="Normal: 0.1-0.5 mmol/L")

st.sidebar.subheader("Physical Measurements")
bmi = st.sidebar.number_input("BMI (kg/m²)", 10.0, 60.0, 25.0, 0.1, help="Normal: 18.5-24.9")

# Create input dataframe
input_data = pd.DataFrame({
    'Gender': [1 if gender == "Male" else 0],
    'AGE': [age],
    'Urea': [urea],
    'Cr': [cr],
    'HbA1c': [hba1c],
    'Chol': [chol],
    'TG': [tg],
    'HDL': [hdl],
    'LDL': [ldl],
    'VLDL': [vldl],
    'BMI': [bmi]
})

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Patient Data Summary")
    
    # Display input in a nice format
    display_data = input_data.copy()
    display_data['Gender'] = gender
    display_data = display_data.T
    display_data.columns = ['Value']
    st.dataframe(display_data, use_container_width=True)

with col2:
    st.subheader("🎯 Risk Indicators")
    
    # Risk indicators based on HbA1c
    if hba1c < 5.7:
        st.success("✅ Normal HbA1c")
    elif hba1c < 6.5:
        st.warning("⚠️ Prediabetic HbA1c")
    else:
        st.error("🚨 Diabetic HbA1c")
    
    # BMI indicator
    if bmi < 18.5:
        st.info("📉 Underweight BMI")
    elif bmi < 25:
        st.success("✅ Normal BMI")
    elif bmi < 30:
        st.warning("⚠️ Overweight BMI")
    else:
        st.error("🚨 Obese BMI")

# Prediction button
st.markdown("---")
if st.button("🔍 Predict Diabetes Status", type="primary", use_container_width=True):
    if model is not None:
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        class_mapping = {0: 'Non-diabetic', 1: 'Prediabetic', 2: 'Diabetic'}
        result = class_mapping[prediction]
        
        st.markdown("---")
        st.subheader("📋 Prediction Results")
        
        # Display prediction with appropriate styling
        if prediction == 0:
            box_class = "healthy"
            emoji = "✅"
            color = "#28a745"
        elif prediction == 1:
            box_class = "prediabetic"
            emoji = "⚠️"
            color = "#ffc107"
        else:
            box_class = "diabetic"
            emoji = "🚨"
            color = "#dc3545"
        
        st.markdown(f'''
            <div class="prediction-box {box_class}">
                <h2 style="text-align: center;">{emoji} Prediction: {result.upper()}</h2>
            </div>
        ''', unsafe_allow_html=True)
        
        # Probability visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Class': ['Non-diabetic', 'Prediabetic', 'Diabetic'],
                'Probability': prediction_proba * 100
            })
            
            for i, row in prob_df.iterrows():
                st.metric(
                    label=row['Class'],
                    value=f"{row['Probability']:.2f}%"
                )
        
        with col2:
            st.subheader("Probability Distribution")
            fig = go.Figure(data=[
                go.Bar(
                    x=prob_df['Class'],
                    y=prob_df['Probability'],
                    marker_color=['#28a745', '#ffc107', '#dc3545'],
                    text=[f"{p:.1f}%" for p in prob_df['Probability']],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                yaxis_title="Probability (%)",
                xaxis_title="Class",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if prediction == 0:
            st.success("""
            **Great news!** The model predicts a low risk of diabetes. However:
            - Maintain a healthy lifestyle
            - Regular exercise (30+ minutes daily)
            - Balanced diet rich in vegetables and whole grains
            - Annual health check-ups
            """)
        elif prediction == 1:
            st.warning("""
            **Attention needed!** The model indicates prediabetes risk:
            - Consult with a healthcare provider
            - Monitor blood sugar levels regularly
            - Increase physical activity
            - Reduce sugar and refined carbohydrate intake
            - Consider weight management if BMI is high
            - Follow-up testing in 3-6 months
            """)
        else:
            st.error("""
            **Important!** The model indicates diabetes risk:
            - **Consult a doctor immediately** for proper diagnosis
            - Blood glucose monitoring may be necessary
            - Lifestyle modifications are essential
            - Medication may be required (doctor's decision)
            - Regular medical follow-ups
            - Diet and exercise plan with professional guidance
            """)
        
        st.info("⚠️ **Disclaimer:** This is a predictive model for educational purposes only. It does NOT replace professional medical diagnosis. Always consult with healthcare professionals for proper medical advice.")
    else:
        st.error("Model not loaded. Please check the model file.")

# Additional information
with st.expander("ℹ️ About This App"):
    st.markdown("""
    ### Diabetes Prediction System
    
    This application uses a **Decision Tree Classifier** trained on diabetes patient data to predict 
    the likelihood of diabetes based on various medical parameters.
    
    **Features Used:**
    - Gender
    - Age
    - Urea levels
    - Creatinine (Cr)
    - HbA1c (Glycated Hemoglobin)
    - Cholesterol profile (Total, TG, HDL, LDL, VLDL)
    - BMI (Body Mass Index)
    
    **Model Performance:**
    - Algorithm: Decision Tree (max_depth=5)
    - Training/Test Split: 70/30
    
    **Classification Categories:**
    1. **Non-diabetic**: No diabetes detected
    2. **Prediabetic**: At risk, requires monitoring
    3. **Diabetic**: Diabetes indicators present
    """)

with st.expander("📖 Understanding the Parameters"):
    st.markdown("""
    - **HbA1c**: Average blood sugar over 2-3 months. Key diabetes indicator.
    - **Urea & Creatinine**: Kidney function markers
    - **Cholesterol Profile**: Heart health and metabolic indicators
    - **BMI**: Weight status relative to height
    - **Lipid Profile**: TG, HDL, LDL, VLDL indicate fat metabolism
    """)