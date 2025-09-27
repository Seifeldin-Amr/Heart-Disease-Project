import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('/workspaces/Heart-Disease-Project/models/final_model.pkl')
        return model
    except:
        st.error("Error loading the model. Please check if the model file exists.")
        return None

# Load and cache the dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('/workspaces/Heart-Disease-Project/data/heart_disease.csv')
        return data
    except:
        st.error("Error loading the dataset. Please check if the data file exists.")
        return None

# Feature preprocessing function
@st.cache_data
def get_training_scaler():
    """
    Create and fit a scaler using the training data to match the model's expectations
    """
    # Load the training data
    data = pd.read_csv('/workspaces/Heart-Disease-Project/data/heart_disease.csv')
    
    # Get the numeric features that need scaling
    numeric_features = ['age', 'chol', 'thalach']
    
    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(data[numeric_features])
    
    return scaler

def preprocess_features(features_dict):
    """
    Preprocess the input features to match the training data format
    """
    # Create a DataFrame from the input features
    feature_df = pd.DataFrame([features_dict])
    
    # Create one-hot encoded features to match training data
    # Based on the selected features: 'chol', 'age', 'thalach', 'ca_2.0', 'exang_1.0', 'cp_4.0', 'slope_2.0', 'thal_7.0', 'ca_1.0', 'ca_3.0'
    
    # Initialize all one-hot encoded features to 0
    feature_df['ca_1.0'] = 0
    feature_df['ca_2.0'] = 0  
    feature_df['ca_3.0'] = 0
    feature_df['exang_1.0'] = 0
    feature_df['cp_4.0'] = 0
    feature_df['slope_2.0'] = 0
    feature_df['thal_7.0'] = 0
    
    # Set the appropriate one-hot encoded values based on input
    if features_dict['ca'] == 1.0:
        feature_df['ca_1.0'] = 1
    elif features_dict['ca'] == 2.0:
        feature_df['ca_2.0'] = 1
    elif features_dict['ca'] == 3.0:
        feature_df['ca_3.0'] = 1
        
    if features_dict['exang'] == 1:
        feature_df['exang_1.0'] = 1
        
    if features_dict['cp'] == 4:
        feature_df['cp_4.0'] = 1
        
    if features_dict['slope'] == 2:
        feature_df['slope_2.0'] = 1
        
    if features_dict['thal'] == 7.0:
        feature_df['thal_7.0'] = 1
    
    # Scale the numeric features using the training data scaler
    scaler = get_training_scaler()
    numeric_cols = ['age', 'chol', 'thalach']
    feature_df[numeric_cols] = scaler.transform(feature_df[numeric_cols])
    
    # Select only the features that were used in training (in the correct order)
    selected_features = ['chol', 'age', 'thalach', 'ca_2.0', 'exang_1.0', 'cp_4.0', 'slope_2.0', 'thal_7.0', 'ca_1.0', 'ca_3.0']
    
    # Return the processed features in the correct order
    return feature_df[selected_features]

# Prediction function
def make_prediction(model, features):
    """
    Make prediction and return probabilities
    """
    try:
        # Preprocess features
        processed_features = preprocess_features(features)
        
        # Make prediction
        prediction = model.predict(processed_features)[0]
        prediction_proba = model.predict_proba(processed_features)[0]
        
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Main title and description
st.title("❤️ Heart Disease Prediction Application")
st.markdown("""
This application uses machine learning to predict the risk of heart disease based on various health parameters.
Please fill in your health information in the sidebar to get a prediction.
""")

# Load model and data
model = load_model()
data = load_data()

if model is None or data is None:
    st.stop()

# Sidebar for user inputs
st.sidebar.header("📋 Enter Your Health Information")
st.sidebar.markdown("Please provide the following information:")

# Input fields in sidebar
age = st.sidebar.slider("Age", min_value=20, max_value=100, value=50, help="Your age in years")

sex = st.sidebar.selectbox("Sex", 
                          options=[1, 0], 
                          format_func=lambda x: "Male" if x == 1 else "Female",
                          help="Biological sex")

cp = st.sidebar.selectbox("Chest Pain Type", 
                         options=[0, 1, 2, 3, 4],
                         format_func=lambda x: {
                             0: "Asymptomatic",
                             1: "Typical Angina", 
                             2: "Atypical Angina",
                             3: "Non-anginal Pain",
                             4: "Atypical Angina"
                         }.get(x, "Unknown"),
                         help="Type of chest pain experienced")

trestbps = st.sidebar.slider("Resting Blood Pressure (mmHg)", 
                            min_value=80, max_value=220, value=120,
                            help="Resting blood pressure in mmHg")

chol = st.sidebar.slider("Cholesterol (mg/dl)", 
                        min_value=100, max_value=600, value=200,
                        help="Serum cholesterol in mg/dl")

fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                          options=[0, 1],
                          format_func=lambda x: "Yes" if x == 1 else "No",
                          help="Fasting blood sugar > 120 mg/dl")

restecg = st.sidebar.selectbox("Resting ECG Results", 
                              options=[0, 1, 2],
                              format_func=lambda x: {
                                  0: "Normal",
                                  1: "ST-T Wave Abnormality",
                                  2: "Left Ventricular Hypertrophy"
                              }.get(x, "Unknown"),
                              help="Resting electrocardiographic results")

thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 
                           min_value=60, max_value=220, value=150,
                           help="Maximum heart rate achieved during exercise")

exang = st.sidebar.selectbox("Exercise Induced Angina", 
                            options=[0, 1],
                            format_func=lambda x: "Yes" if x == 1 else "No",
                            help="Exercise induced angina")

oldpeak = st.sidebar.slider("ST Depression", 
                           min_value=0.0, max_value=7.0, value=1.0, step=0.1,
                           help="ST depression induced by exercise relative to rest")

slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", 
                            options=[1, 2, 3],
                            format_func=lambda x: {
                                1: "Upsloping",
                                2: "Flat", 
                                3: "Downsloping"
                            }.get(x, "Unknown"),
                            help="Slope of the peak exercise ST segment")

ca = st.sidebar.slider("Number of Major Vessels", 
                      min_value=0.0, max_value=4.0, value=0.0, step=1.0,
                      help="Number of major vessels colored by fluoroscopy")

thal = st.sidebar.selectbox("Thalassemia", 
                           options=[3.0, 6.0, 7.0],
                           format_func=lambda x: {
                               3.0: "Normal",
                               6.0: "Fixed Defect",
                               7.0: "Reversible Defect"
                           }.get(x, "Unknown"),
                           help="Thalassemia test results")

# Create feature dictionary
features = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data Insights", "📈 Visualizations"])

with tab1:
    st.header("Heart Disease Risk Prediction")
    
    # Make prediction button
    if st.button("🔍 Get Prediction", type="primary", use_container_width=True):
        with st.spinner("Analyzing your health data..."):
            prediction, probabilities = make_prediction(model, features)
            
            if prediction is not None:
                # Display results
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    # Define severity levels
                    severity_labels = {
                        0: "✅ No Heart Disease",
                        1: "⚠️ Mild Heart Disease",
                        2: "🟡 Moderate Heart Disease",
                        3: "🔴 Severe Heart Disease",
                        4: "🆘 Very Severe Heart Disease"
                    }
                    
                    # Define risk colors
                    severity_colors = {
                        0: "green",
                        1: "yellow", 
                        2: "orange",
                        3: "red",
                        4: "darkred"
                    }
                    
                    # Convert prediction to regular int if it's a numpy type
                    prediction_int = int(prediction)
                    
                    prediction_label = severity_labels.get(prediction_int, "Unknown")
                    prediction_color = severity_colors.get(prediction_int, "gray")
                    
                    if prediction_int == 0:
                        st.success(prediction_label)
                        risk_level = "No Risk"
                    elif prediction_int == 1:
                        st.warning(prediction_label)
                        risk_level = "Low Risk"
                    elif prediction_int == 2:
                        st.warning(prediction_label)
                        risk_level = "Moderate Risk"
                    else:
                        st.error(prediction_label)
                        risk_level = "High Risk"
                    
                    # Show probability
                    risk_probability = probabilities[prediction_int] * 100
                    st.metric(
                        label=f"Confidence Level", 
                        value=f"{risk_probability:.1f}%"
                    )
                    
                    # Risk gauge chart - map severity to 0-100 scale
                    gauge_value = min((prediction_int / 4) * 100, 100)
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=gauge_value,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"Severity Level: {prediction_int}"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': prediction_color},
                            'steps': [
                                {'range': [0, 20], 'color': "lightgreen"},
                                {'range': [20, 40], 'color': "yellow"},
                                {'range': [40, 60], 'color': "orange"},
                                {'range': [60, 80], 'color': "red"},
                                {'range': [80, 100], 'color': "darkred"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 60
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=400)
                    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Display input summary
    st.header("📋 Your Input Summary")
    
    # Show the selected features used by the model
    with st.expander("ℹ️ Model Information", expanded=False):
        st.info("This model uses 10 carefully selected features from the original 13 features through feature selection techniques.")
        selected_features_list = ['chol', 'age', 'thalach', 'ca_2.0', 'exang_1.0', 'cp_4.0', 'slope_2.0', 'thal_7.0', 'ca_1.0', 'ca_3.0']
        st.write("**Selected Features:**")
        for i, feat in enumerate(selected_features_list, 1):
            if 'ca_' in feat:
                st.write(f"{i}. {feat} - Number of major vessels = {feat.split('_')[1]}")
            elif feat == 'exang_1.0':
                st.write(f"{i}. {feat} - Exercise induced angina = Yes")
            elif feat == 'cp_4.0':
                st.write(f"{i}. {feat} - Chest pain type = 4")
            elif feat == 'slope_2.0':
                st.write(f"{i}. {feat} - ST slope = 2 (Flat)")
            elif feat == 'thal_7.0':
                st.write(f"{i}. {feat} - Thalassemia = 7 (Reversible defect)")
            else:
                st.write(f"{i}. {feat} - {feat.title()}")
    
    input_df = pd.DataFrame([features])
    
    # Format the display
    display_df = input_df.copy()
    display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
    
    # Create formatted display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        st.write(f"**Age:** {age} years")
        st.write(f"**Sex:** {'Male' if sex == 1 else 'Female'}")
        st.write(f"**Chest Pain Type:** {cp}")
        st.write(f"**Resting Blood Pressure:** {trestbps} mmHg")
        st.write(f"**Cholesterol:** {chol} mg/dl")
        st.write(f"**Fasting Blood Sugar > 120:** {'Yes' if fbs == 1 else 'No'}")
    
    with col2:
        st.subheader("Cardiac Measurements")
        st.write(f"**Resting ECG:** {restecg}")
        st.write(f"**Max Heart Rate:** {thalach} bpm")
        st.write(f"**Exercise Angina:** {'Yes' if exang == 1 else 'No'}")
        st.write(f"**ST Depression:** {oldpeak}")
        st.write(f"**ST Slope:** {slope}")
        st.write(f"**Major Vessels:** {ca}")
        st.write(f"**Thalassemia:** {thal}")

with tab2:
    st.header("📊 Dataset Insights")
    
    if data is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", len(data))
        with col2:
            # No disease cases (target = 0)
            no_disease_count = len(data[data['target'] == 0])
            st.metric("No Disease Cases", no_disease_count)
        with col3:
            # Any disease cases (target > 0)
            disease_count = len(data[data['target'] > 0])
            st.metric("Disease Cases", disease_count)
        with col4:
            disease_rate = (disease_count / len(data)) * 100
            st.metric("Disease Rate", f"{disease_rate:.1f}%")
        
        # Target distribution
        st.subheader("Heart Disease Distribution")
        target_counts = data['target'].value_counts().sort_index()
        
        # Create labels for the different heart disease severity levels
        target_labels = {
            0: 'No Disease',
            1: 'Mild Disease',
            2: 'Moderate Disease',
            3: 'Severe Disease',
            4: 'Very Severe Disease'
        }
        
        fig_pie = px.pie(
            values=target_counts.values, 
            names=[target_labels[i] for i in target_counts.index], 
            title="Distribution of Heart Disease Severity in Dataset"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Age distribution by target
        st.subheader("Age Distribution by Heart Disease Severity")
        
        # Create a copy of data with readable labels
        data_viz = data.copy()
        target_labels = {
            0: 'No Disease',
            1: 'Mild Disease', 
            2: 'Moderate Disease',
            3: 'Severe Disease',
            4: 'Very Severe Disease'
        }
        data_viz['disease_level'] = data_viz['target'].map(target_labels)
        
        fig_age = px.histogram(
            data_viz, 
            x='age', 
            color='disease_level', 
            barmode='overlay',
            title="Age Distribution by Heart Disease Severity",
            labels={'disease_level': 'Disease Severity', 'age': 'Age', 'count': 'Count'},
            opacity=0.7
        )
        st.plotly_chart(fig_age, use_container_width=True)

with tab3:
    st.header("📈 Data Visualizations")
    
    if data is not None:
        # Correlation heatmap
        st.subheader("Feature Correlation Matrix")
        
        # Select numeric columns for correlation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        fig_heatmap.update_layout(height=600)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Box plots for key features
        st.subheader("Key Health Metrics Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cholesterol by heart disease severity
            fig_chol = px.box(
                data_viz, 
                x='disease_level', 
                y='chol',
                title="Cholesterol Levels by Disease Severity",
                labels={'disease_level': 'Disease Severity', 'chol': 'Cholesterol (mg/dl)'}
            )
            fig_chol.update_xaxes(tickangle=45)
            st.plotly_chart(fig_chol, use_container_width=True)
        
        with col2:
            # Blood pressure by heart disease severity
            fig_bp = px.box(
                data_viz, 
                x='disease_level', 
                y='trestbps',
                title="Blood Pressure by Disease Severity",
                labels={'disease_level': 'Disease Severity', 'trestbps': 'Blood Pressure (mmHg)'}
            )
            fig_bp.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bp, use_container_width=True)
        
        # Scatter plot: Age vs Max Heart Rate
        st.subheader("Age vs Maximum Heart Rate")
        fig_scatter = px.scatter(
            data_viz, 
            x='age', 
            y='thalach',
            color='disease_level',
            title="Age vs Maximum Heart Rate by Disease Severity",
            labels={
                'age': 'Age (years)', 
                'thalach': 'Maximum Heart Rate (bpm)',
                'disease_level': 'Disease Severity'
            }
        )
        fig_scatter.update_traces(marker=dict(size=8))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Chest pain type distribution
        st.subheader("Chest Pain Types Distribution")
        cp_counts = data['cp'].value_counts().sort_index()
        cp_labels = {0: 'Asymptomatic', 1: 'Typical Angina', 2: 'Atypical Angina', 
                    3: 'Non-anginal Pain', 4: 'Atypical Angina'}
        
        fig_cp = px.bar(
            x=[cp_labels.get(i, f'Type {i}') for i in cp_counts.index],
            y=cp_counts.values,
            title="Distribution of Chest Pain Types",
            labels={'x': 'Chest Pain Type', 'y': 'Count'}
        )
        st.plotly_chart(fig_cp, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>❤️ Heart Disease Prediction App | Built with Streamlit & Machine Learning</p>
    <p><small>⚠️ This app is for educational purposes only. Please consult healthcare professionals for medical advice.</small></p>
</div>
""", unsafe_allow_html=True)
