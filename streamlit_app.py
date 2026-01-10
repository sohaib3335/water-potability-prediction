"""
Water Potability Prediction System
Streamlit Web Application
MSc Computing - Independent Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             classification_report, precision_score, 
                             recall_score, f1_score, roc_curve, auc)

# Page Configuration
st.set_page_config(
    page_title="Water Potability Prediction",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .prediction-safe {
        background-color: #4CAF50;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
    }
    .prediction-unsafe {
        background-color: #f44336;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('water_potability.csv')
    # Handle missing values with median
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            df[column].fillna(df[column].median(), inplace=True)
    return df

@st.cache_resource
def train_models(df):
    # Prepare data
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, 
                                  random_state=42, eval_metric='logloss', scale_pos_weight=1.5),
        'LightGBM': LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, 
                                    random_state=42, verbose=-1, class_weight='balanced')
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        trained_models[name] = model
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'Predictions': y_pred,
            'Probabilities': y_proba
        }
    
    return trained_models, results, scaler, X_test, y_test, X.columns.tolist()

# Main App
def main():
    st.markdown('<h1 class="main-header">Water Potability Prediction System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/water.png", width=80)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                            ["Home", "Data Exploration", "Model Training", 
                             "Prediction", "Model Comparison"])
    
    # Load data
    try:
        df = load_data()
        trained_models, results, scaler, X_test, y_test, feature_names = train_models(df)
    except FileNotFoundError:
        st.error("Dataset file 'water_potability.csv' not found. Please ensure it's in the same directory.")
        return
    
    # Page: Home
    if page == "Home":
        st.markdown("### Welcome to the Water Potability Prediction System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### About This Project
            
            This application predicts whether water is **safe for human consumption** 
            based on various water quality parameters using Machine Learning algorithms.
            
            **Domain:** Environmental Science / Public Health
            
            **Problem Statement:**
            Access to safe drinking water is essential for health and is a basic human right. 
            This system helps predict water potability based on chemical properties.
            """)
            
        with col2:
            st.markdown("""
            #### Dataset Overview
            """)
            st.metric("Total Samples", len(df))
            st.metric("Features", len(df.columns) - 1)
            st.metric("Target Classes", "2 (Potable / Not Potable)")
        
        st.markdown("---")
        st.markdown("#### Features Description")
        
        features_info = pd.DataFrame({
            'Feature': ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                       'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'],
            'Description': [
                'Acidity/Alkalinity level (0-14)',
                'Capacity to precipitate soap (mg/L)',
                'Total dissolved solids (ppm)',
                'Amount of chloramines (ppm)',
                'Sulfate dissolved (mg/L)',
                'Electrical conductivity (μS/cm)',
                'Organic carbon amount (ppm)',
                'Trihalomethanes amount (μg/L)',
                'Cloudiness measure (NTU)'
            ]
        })
        st.table(features_info)
        
        # Quick Prediction Section on Home Page
        st.markdown("---")
        st.markdown("### Quick Prediction")
        st.markdown("Enter water quality parameters to predict if water is safe for consumption:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            home_ph = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1, key="home_ph")
            home_hardness = st.slider("Hardness (mg/L)", 50.0, 350.0, 200.0, 1.0, key="home_hardness")
            home_solids = st.slider("Solids (ppm)", 300.0, 60000.0, 20000.0, 100.0, key="home_solids")
        
        with col2:
            home_chloramines = st.slider("Chloramines (ppm)", 0.0, 15.0, 7.0, 0.1, key="home_chloramines")
            home_sulfate = st.slider("Sulfate (mg/L)", 100.0, 500.0, 300.0, 1.0, key="home_sulfate")
            home_conductivity = st.slider("Conductivity (μS/cm)", 100.0, 800.0, 400.0, 1.0, key="home_conductivity")
        
        with col3:
            home_organic_carbon = st.slider("Organic Carbon (ppm)", 2.0, 30.0, 14.0, 0.1, key="home_organic_carbon")
            home_trihalomethanes = st.slider("Trihalomethanes (μg/L)", 0.0, 130.0, 65.0, 1.0, key="home_trihalomethanes")
            home_turbidity = st.slider("Turbidity (NTU)", 1.0, 7.0, 4.0, 0.1, key="home_turbidity")
        
        # Model selection
        home_selected_model = st.selectbox("Select Model for Prediction", list(trained_models.keys()), key="home_model")
        
        if st.button("Predict Water Potability", type="primary", key="home_predict"):
            # Prepare input
            home_input_data = np.array([[home_ph, home_hardness, home_solids, home_chloramines, home_sulfate, 
                                   home_conductivity, home_organic_carbon, home_trihalomethanes, home_turbidity]])
            home_input_scaled = scaler.transform(home_input_data)
            
            # Make prediction
            home_model = trained_models[home_selected_model]
            home_prediction = home_model.predict(home_input_scaled)[0]
            home_probability = home_model.predict_proba(home_input_scaled)[0]
            
            st.markdown("---")
            st.markdown("### Prediction Result")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if home_prediction == 1:
                    st.markdown("""
                    <div class="prediction-safe">
                        POTABLE<br>
                        <small>Water is SAFE for consumption</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-unsafe">
                        NOT POTABLE<br>
                        <small>Water is NOT SAFE for consumption</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            with result_col2:
                st.markdown("#### Confidence Scores")
                st.progress(float(home_probability[0]), text=f"Not Potable: {home_probability[0]*100:.1f}%")
                st.progress(float(home_probability[1]), text=f"Potable: {home_probability[1]*100:.1f}%")
    
    # Page: Data Exploration
    elif page == "Data Exploration":
        st.markdown("### Data Exploration")
        
        tab1, tab2, tab3 = st.tabs(["Dataset", "Distributions", "Correlation"])
        
        with tab1:
            st.markdown("#### Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            st.markdown("#### Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
        
        with tab2:
            st.markdown("#### Target Variable Distribution")
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            colors = ['#e74c3c', '#2ecc71']
            labels = ['Not Potable (0)', 'Potable (1)']
            
            df['Potability'].value_counts().plot(kind='pie', ax=ax[0], colors=colors, 
                                                  autopct='%1.1f%%', labels=labels)
            ax[0].set_title('Target Distribution', fontweight='bold')
            ax[0].set_ylabel('')
            
            df['Potability'].value_counts().plot(kind='bar', ax=ax[1], color=colors)
            ax[1].set_title('Target Count', fontweight='bold')
            ax[1].set_xticklabels(labels, rotation=0)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("#### Feature Distributions")
            feature = st.selectbox("Select Feature", feature_names)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data=df, x=feature, hue='Potability', kde=True, 
                        palette=['#e74c3c', '#2ecc71'], ax=ax)
            ax.set_title(f'Distribution of {feature}', fontweight='bold')
            st.pyplot(fig)
        
        with tab3:
            st.markdown("#### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', linewidths=0.5, ax=ax)
            ax.set_title('Feature Correlation Heatmap', fontweight='bold')
            st.pyplot(fig)
    
    # Page: Model Training
    elif page == "Model Training":
        st.markdown("### Model Training Results")
        
        # Display metrics for each model
        for model_name, metrics in results.items():
            with st.expander(f"{model_name}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{metrics['Accuracy']*100:.2f}%")
                col2.metric("Precision", f"{metrics['Precision']:.4f}")
                col3.metric("Recall", f"{metrics['Recall']:.4f}")
                col4.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
        
        st.markdown("---")
        st.markdown("#### Confusion Matrices")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for idx, (name, metrics) in enumerate(results.items()):
            row, col = idx // 2, idx % 2
            cm = confusion_matrix(y_test, metrics['Predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col],
                       xticklabels=['Not Potable', 'Potable'],
                       yticklabels=['Not Potable', 'Potable'])
            axes[row, col].set_title(f'{name}', fontweight='bold')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Page: Prediction
    elif page == "Prediction":
        st.markdown("### Make a Prediction")
        
        st.markdown("Enter water quality parameters to predict if water is safe for consumption:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ph = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1)
            hardness = st.slider("Hardness (mg/L)", 50.0, 350.0, 200.0, 1.0)
            solids = st.slider("Solids (ppm)", 300.0, 60000.0, 20000.0, 100.0)
        
        with col2:
            chloramines = st.slider("Chloramines (ppm)", 0.0, 15.0, 7.0, 0.1)
            sulfate = st.slider("Sulfate (mg/L)", 100.0, 500.0, 300.0, 1.0)
            conductivity = st.slider("Conductivity (μS/cm)", 100.0, 800.0, 400.0, 1.0)
        
        with col3:
            organic_carbon = st.slider("Organic Carbon (ppm)", 2.0, 30.0, 14.0, 0.1)
            trihalomethanes = st.slider("Trihalomethanes (μg/L)", 0.0, 130.0, 65.0, 1.0)
            turbidity = st.slider("Turbidity (NTU)", 1.0, 7.0, 4.0, 0.1)
        
        # Model selection
        selected_model = st.selectbox("Select Model for Prediction", list(trained_models.keys()))
        
        if st.button("Predict Water Potability", type="primary"):
            # Prepare input
            input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                                   conductivity, organic_carbon, trihalomethanes, turbidity]])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            model = trained_models[selected_model]
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            st.markdown("---")
            st.markdown("### Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.markdown("""
                    <div class="prediction-safe">
                        POTABLE<br>
                        <small>Water is SAFE for consumption</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-unsafe">
                        NOT POTABLE<br>
                        <small>Water is NOT SAFE for consumption</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Confidence Scores")
                st.progress(float(probability[0]), text=f"Not Potable: {probability[0]*100:.1f}%")
                st.progress(float(probability[1]), text=f"Potable: {probability[1]*100:.1f}%")
    
    # Page: Model Comparison
    elif page == "Model Comparison":
        st.markdown("### Model Comparison")
        
        # Comparison table
        comparison_data = []
        for name, metrics in results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{metrics['Accuracy']*100:.2f}%",
                'Precision': f"{metrics['Precision']:.4f}",
                'Recall': f"{metrics['Recall']:.4f}",
                'F1-Score': f"{metrics['F1-Score']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        # Bar chart comparison
        st.markdown("#### Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        models_names = list(results.keys())
        accuracies = [results[m]['Accuracy'] * 100 for m in models_names]
        
        bars = ax.bar(models_names, accuracies, color=colors, edgecolor='black')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Model Accuracy Comparison', fontweight='bold')
        ax.set_ylim(0, 100)
        
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.1f}%', ha='center', fontsize=10)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # ROC Curves
        st.markdown("#### ROC Curves")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name, metrics in results.items():
            fpr, tpr, _ = roc_curve(y_test, metrics['Probabilities'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - Model Comparison', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Best model
        best_model = max(results.items(), key=lambda x: x[1]['Accuracy'])
        st.success(f"Best Model: {best_model[0]} with {best_model[1]['Accuracy']*100:.2f}% accuracy")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    **Water Potability Prediction System**
    
    MSc Computing - Independent Project
    
    Using Machine Learning to predict 
    water safety for consumption.
    """)

if __name__ == "__main__":
    main()
