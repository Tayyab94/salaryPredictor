# app.py - Clean version that only loads pre-trained model
import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables
model = None
scaler = None
features = []

def load_model_artifacts():
    """Load pre-trained model and artifacts"""
    global model, scaler, features
    
    try:
        # Load your trained model
        model = joblib.load('model/trained_model.pkl')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    try:
        # Load scaler
        scaler = joblib.load('model/scaler.pkl')
        print("✅ Scaler loaded successfully")
    except Exception as e:
        print(f"⚠️ Scaler not found: {e}")
        scaler = None
    
    try:
        # Load features list
        with open('model/features.txt', 'r') as f:
            features = [line.strip() for line in f.readlines()]
        print(f"✅ Features loaded: {len(features)} features")
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        return False
    
    return True

# Load all artifacts on startup
if not load_model_artifacts():
    print("❌ Failed to load model artifacts. Please check your model files.")
    exit(1)

def engineer_features(df):
    """Apply feature engineering (must match your training pipeline)"""
    
    df_engineered = df.copy()
    
    # Add engineered features that your model expects
    df_engineered[' '] = (df_engineered['equity_offered_pct'] > 0).astype(int)
    df_engineered['has_bonus'] = (df_engineered['bonus_pct'] > 0).astype(int)
    
    # Experience mapping
    exp_map = {'Entry': 1, 'Mid': 2, 'Senior': 3, 'Executive': 4}
    df_engineered['exp_level_encoded'] = df_engineered['experience_level'].map(exp_map).fillna(2)
    df_engineered['seniority_score'] = df_engineered['years_experience'] * df_engineered['exp_level_encoded']
    
    # Remote features
    df_engineered['remote_intensity'] = df_engineered['remote_ratio'] / 100
    company_size_map = {'S': 1, 'M': 2, 'L': 3}
    df_engineered['company_size_encoded'] = df_engineered['company_size'].map(company_size_map).fillna(2)
    df_engineered['remote_company_size_interaction'] = df_engineered['remote_ratio'] * df_engineered['company_size_encoded']
    
    # Education features
    education_map = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    df_engineered['education_level_encoded'] = df_engineered['education_level'].map(education_map).fillna(2)
    df_engineered['has_advanced_degree'] = (df_engineered['education_level'].isin(['Master', 'PhD'])).astype(int)
    
    # AI features
    df_engineered['ai_usage_intensity'] = df_engineered['ai_tools_hours_per_week'] / (df_engineered['weekly_hours'] + 1)
    
    # Comprehensive seniority
    df_engineered['comprehensive_seniority'] = (
        df_engineered['years_experience'] * 0.3 +
        df_engineered['exp_level_encoded'] * 0.25 +
        df_engineered['education_level_encoded'] * 0.15 +
        df_engineered['certifications_count'] * 0.1 +
        df_engineered['team_size'] * 0.1
    )
    
    # Additional features
    df_engineered['certs_per_experience_year'] = df_engineered['certifications_count'] / (df_engineered['years_experience'] + 1)
    df_engineered['job_switcher'] = (df_engineered['switched_jobs_last_year'] > 0).astype(int)
    df_engineered['upskilling_dedication'] = (df_engineered['upskilling_hours_per_month'] > 10).astype(int)
    
    # Categorical mappings
    categorical_mappings = {
        'job_category': {
            'Data_Scientist': 0, 'ML_Engineer': 1, 'AI_Engineer': 2, 
            'Analyst': 3, 'Manager': 4, 'Other': 5
        },
        'remote_category': {
            'Onsite': 0, 'Hybrid_Low': 1, 'Hybrid_High': 2, 'Fully_Remote': 3
        },
        'ai_usage_category': {
            'Non_User': 0, 'Light_User': 1, 'Moderate_User': 2, 'Heavy_User': 3
        }
    }
    
    for col, mapping in categorical_mappings.items():
        if col in df_engineered.columns:
            df_engineered[col] = df_engineered[col].map(mapping).fillna(0)
    
    # Encode other categoricals
    if 'employment_type' in df_engineered.columns:
        df_engineered['employment_type'] = df_engineered['employment_type'].map({
            'Full-time': 0, 'Part-time': 1, 'Contract': 2
        }).fillna(0)
    
    if 'primary_language' in df_engineered.columns:
        df_engineered['primary_language'] = df_engineered['primary_language'].map({
            'Python': 0, 'SQL': 1, 'R': 2, 'Java': 3, 'C++': 4
        }).fillna(0)
    
    if 'industry' in df_engineered.columns:
        df_engineered['industry'] = df_engineered['industry'].map({
            'Technology': 0, 'Finance': 1, 'Healthcare': 2, 'Education': 3, 'Retail': 4
        }).fillna(0)
    
    # Select only features the model expects
    available_features = [f for f in features if f in df_engineered.columns]
    df_engineered = df_engineered[available_features]
    
    # Add missing features
    for f in features:
        if f not in df_engineered.columns:
            df_engineered[f] = 0
    
    # Reorder columns to match training
    df_engineered = df_engineered[features]
    
    # Scale if scaler exists
    if scaler is not None:
        numerical_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
        try:
            df_engineered[numerical_cols] = scaler.transform(df_engineered[numerical_cols])
        except Exception as e:
            print(f"Warning: Scaling failed: {e}")
    
    return df_engineered

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'experience_level': request.form.get('experience_level', 'Mid'),
            'years_experience': float(request.form.get('years_experience', 3)),
            'company_size': request.form.get('company_size', 'M'),
            'company_location_full': request.form.get('company_location_full', 'United States'),
            'remote_ratio': float(request.form.get('remote_ratio', 50)),
            'education_level': request.form.get('education_level', 'Master'),
            'employment_type': request.form.get('employment_type', 'Full-time'),
            'industry': request.form.get('industry', 'Technology'),
            'primary_language': request.form.get('primary_language', 'Python'),
            'team_size': int(request.form.get('team_size', 5)),
            'certifications_count': int(request.form.get('certifications_count', 2)),
            'weekly_hours': float(request.form.get('weekly_hours', 40)),
            'ai_tools_hours_per_week': float(request.form.get('ai_tools_hours_per_week', 10)),
            'equity_offered_pct': float(request.form.get('equity_offered_pct', 10)),
            'bonus_pct': float(request.form.get('bonus_pct', 15)),
            'job_satisfaction_score': float(request.form.get('job_satisfaction_score', 7)),
            'interviews_to_offer': int(request.form.get('interviews_to_offer', 3)),
            'switched_jobs_last_year': int(request.form.get('switched_jobs_last_year', 0)),
            'upskilling_hours_per_month': float(request.form.get('upskilling_hours_per_month', 15)),
            'fears_ai_automation_score': float(request.form.get('fears_ai_automation_score', 4)),
            'employee_residence': request.form.get('employee_residence', 'US'),
            'manages_people': int(request.form.get('manages_people', 0)),
            'has_ml_in_title': int(request.form.get('has_ml_in_title', 0)),
            'job_category': request.form.get('job_category', 'Data_Scientist'),
            'remote_category': request.form.get('remote_category', 'Hybrid_High'),
            'ai_usage_category': request.form.get('ai_usage_category', 'Moderate_User')
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Apply feature engineering
        engineered_df = engineer_features(input_df)
        
        # Make prediction
        prediction = model.predict(engineered_df)[0]
        
        return jsonify({
            'success': True,
            'predicted_salary': round(prediction, 2),
            'predicted_salary_formatted': f"${round(prediction, 2):,.2f}",
            'features_used': len(features)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'features_count': len(features)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)