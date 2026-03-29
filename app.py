import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and preprocessing objects
@st.cache_resource
def load_model_and_processors():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, scaler, label_encoder

try:
    model, scaler, label_encoder = load_model_and_processors()
    model_loaded = True
except FileNotFoundError as e:
    st.error(f"❌ Error: {e}")
    st.error("Please ensure best_model.pkl, scaler.pkl, and label_encoder.pkl are in the same directory")
    model_loaded = False

if model_loaded:
    # Header
    st.title("🌾 Crop Recommendation System")
    st.write("---")
    st.markdown("""
    ### Welcome to the Smart Crop Recommendation Platform!
    This system uses machine learning to recommend the best crop based on soil quality 
    and environmental conditions.
    """)
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Enter Soil & Environmental Parameters")
        
        # Create input form with more user-friendly layout
        col_a, col_b = st.columns(2)
        
        with col_a:
            nitrogen = st.number_input(
                "Nitrogen (N) Content (kg/ha)",
                min_value=0,
                max_value=200,
                value=50,
                step=1,
                help="Amount of nitrogen in the soil"
            )
            
            phosphorus = st.number_input(
                "Phosphorus (P) Content (kg/ha)",
                min_value=0,
                max_value=150,
                value=30,
                step=1,
                help="Amount of phosphorus in the soil"
            )
            
            potassium = st.number_input(
                "Potassium (K) Content (kg/ha)",
                min_value=0,
                max_value=200,
                value=40,
                step=1,
                help="Amount of potassium in the soil"
            )
            
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=0.0,
                max_value=50.0,
                value=25.0,
                step=0.1,
                help="Average temperature in Celsius"
            )
        
        with col_b:
            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=0.1,
                help="Relative humidity percentage"
            )
            
            ph = st.number_input(
                "Soil pH",
                min_value=3.0,
                max_value=10.0,
                value=6.5,
                step=0.1,
                help="Soil pH level (3-10)"
            )
            
            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                max_value=500.0,
                value=150.0,
                step=0.1,
                help="Average annual rainfall in millimeters"
            )
    
    with col2:
        st.subheader("📊 Input Summary")
        summary_data = {
            "Parameter": ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH", "Rainfall"],
            "Value": [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall],
            "Unit": ["kg/ha", "kg/ha", "kg/ha", "°C", "%", "-", "mm"]
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Prediction Button
    st.write("---")
    
    if st.button("🔮 Get Crop Recommendation", key="predict_btn", use_container_width=True):
        # Prepare input data
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Decode the label
        recommended_crop = label_encoder.inverse_transform(prediction)[0]
        
        # Display results
        st.write("---")
        st.subheader("🎯 Prediction Result")
        
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            st.success(f"### Recommended Crop: **{recommended_crop.upper()}**")
            st.write(f"Based on the provided soil and environmental parameters, the model predicts that **{recommended_crop}** would be the best crop to grow.")
        
        with result_col2:
            st.info(f"🌱 Crop: {recommended_crop}")
    
    # Additional Information
    st.write("---")
    st.subheader("ℹ️ About This Model")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.write("""
        **Model Details:**
        - Algorithm: CatBoost Classifier
        - Accuracy: 99.77%
        - Features: 7 soil & environmental parameters
        """)
    
    with info_col2:
        st.write(f"""
        **Supported Crops ({len(label_encoder.classes_)} varieties):**
        {', '.join(sorted(label_encoder.classes_))}
        """)
    
    # Footer
    st.write("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9rem;'>
    💡 This recommendation system is powered by machine learning and analyzes soil nutrients and environmental conditions to suggest the most suitable crop for your farm.
    </div>
    """, unsafe_allow_html=True)
