# app.py - Complete working version
import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables
model = None
scaler = None
features = []

def load_model_artifacts():
    """Load model artifacts or create them if missing"""
    global model, scaler, features
    
    try:
        model = joblib.load('model/trained_model.pkl')
        print("✅ Model loaded successfully")
    except:
        print("⚠️ Model not found. Creating a simple model for testing...")
        create_dummy_model()
        model = joblib.load('model/trained_model.pkl')
    
    try:
        scaler = joblib.load('model/scaler.pkl')
        print("✅ Scaler loaded successfully")
    except:
        print("⚠️ Scaler not found. Creating a new scaler...")
        create_scaler_from_data()
        scaler = joblib.load('model/scaler.pkl')
    
    try:
        with open('model/features.txt', 'r') as f:
            features = [line.strip() for line in f.readlines()]
        print(f"✅ Features loaded: {len(features)} features")
    except:
        print("⚠️ Features file not found. Creating default features...")
        create_default_features()
        with open('model/features.txt', 'r') as f:
            features = [line.strip() for line in f.readlines()]

def create_dummy_model():
    """Create a simple dummy model for testing"""
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'years_experience': np.random.uniform(0, 30, n_samples),
        'remote_ratio': np.random.uniform(0, 100, n_samples),
        'team_size': np.random.randint(1, 20, n_samples),
        'certifications_count': np.random.randint(0, 5, n_samples),
        'weekly_hours': np.random.normal(40, 5, n_samples),
        'ai_tools_hours_per_week': np.random.uniform(0, 30, n_samples),
        'equity_offered_pct': np.random.uniform(0, 30, n_samples),
        'bonus_pct': np.random.uniform(0, 40, n_samples),
        'job_satisfaction_score': np.random.uniform(1, 10, n_samples),
        'interviews_to_offer': np.random.randint(1, 8, n_samples),
        'switched_jobs_last_year': np.random.randint(0, 3, n_samples),
        'upskilling_hours_per_month': np.random.uniform(0, 40, n_samples),
        'fears_ai_automation_score': np.random.uniform(1, 10, n_samples),
        'exp_level_encoded': np.random.randint(1, 4, n_samples),
        'has_equity': np.random.randint(0, 2, n_samples),
        'has_bonus': np.random.randint(0, 2, n_samples),
        'seniority_score': np.random.uniform(1, 100, n_samples),
        'remote_intensity': np.random.uniform(0, 1, n_samples),
        'company_size_encoded': np.random.randint(1, 4, n_samples),
        'remote_company_size_interaction': np.random.uniform(0, 300, n_samples),
        'location_mismatch': np.random.randint(0, 2, n_samples),
        'education_level_encoded': np.random.randint(1, 4, n_samples),
        'has_advanced_degree': np.random.randint(0, 2, n_samples),
        'education_experience_interaction': np.random.uniform(0, 100, n_samples),
        'ai_usage_intensity': np.random.uniform(0, 1, n_samples),
        'comprehensive_seniority': np.random.uniform(1, 100, n_samples),
        'efficiency_score': np.random.uniform(1, 100, n_samples),
        'certs_per_experience_year': np.random.uniform(0, 5, n_samples),
        'job_switcher': np.random.randint(0, 2, n_samples),
        'upskilling_dedication': np.random.randint(0, 2, n_samples)
    }
    
    df_dummy = pd.DataFrame(data)
    
    df_dummy['salary_usd'] = (
        50000 + 
        df_dummy['years_experience'] * 8000 +
        df_dummy['exp_level_encoded'] * 15000 +
        df_dummy['remote_intensity'] * 10000 +
        df_dummy['company_size_encoded'] * 5000 +
        np.random.normal(0, 20000, n_samples)
    )
    df_dummy['salary_usd'] = df_dummy['salary_usd'].clip(30000, 300000)
    
    X = df_dummy.drop('salary_usd', axis=1)
    y = df_dummy['salary_usd']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/trained_model.pkl')
    
    with open('model/features.txt', 'w') as f:
        for col in X.columns:
            f.write(f"{col}\n")
    
    print("✅ Dummy model created successfully!")

def create_scaler_from_data():
    """Create a scaler from dummy data"""
    from sklearn.preprocessing import StandardScaler
    
    np.random.seed(42)
    n_samples = 1000
    dummy_data = np.random.randn(n_samples, 30) * 10 + 50
    
    scaler = StandardScaler()
    scaler.fit(dummy_data)
    
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler, 'model/scaler.pkl')
    
    print("✅ Scaler created successfully!")

def create_default_features():
    """Create default features list"""
    default_features = [
        'years_experience', 'remote_ratio', 'team_size', 'certifications_count',
        'weekly_hours', 'ai_tools_hours_per_week', 'equity_offered_pct',
        'bonus_pct', 'job_satisfaction_score', 'interviews_to_offer',
        'switched_jobs_last_year', 'upskilling_hours_per_month',
        'fears_ai_automation_score', 'exp_level_encoded', 'has_equity',
        'has_bonus', 'seniority_score', 'remote_intensity',
        'company_size_encoded', 'remote_company_size_interaction',
        'location_mismatch', 'education_level_encoded', 'has_advanced_degree',
        'education_experience_interaction', 'ai_usage_intensity',
        'comprehensive_seniority', 'efficiency_score',
        'certs_per_experience_year', 'job_switcher', 'upskilling_dedication'
    ]
    
    os.makedirs('model', exist_ok=True)
    with open('model/features.txt', 'w') as f:
        for feat in default_features:
            f.write(f"{feat}\n")
    
    print("✅ Default features created!")

# Load artifacts on startup
load_model_artifacts()

def engineer_features(df):
    """Apply feature engineering to input data with safe encoding"""
    
    df_engineered = df.copy()
    
    # 1. Salary-related features
    df_engineered['salary_per_hour'] = 0
    df_engineered['total_compensation'] = 0
    df_engineered['has_equity'] = (df_engineered['equity_offered_pct'] > 0).astype(int)
    df_engineered['has_bonus'] = (df_engineered['bonus_pct'] > 0).astype(int)
    
    # 2. Experience features with safe mapping
    exp_map = {'Entry': 1, 'Mid': 2, 'Senior': 3, 'Executive': 4}
    df_engineered['exp_level_encoded'] = df_engineered['experience_level'].map(exp_map).fillna(2)
    df_engineered['seniority_score'] = df_engineered['years_experience'] * df_engineered['exp_level_encoded']
    
    # 3. Remote features
    df_engineered['remote_intensity'] = df_engineered['remote_ratio'] / 100
    company_size_map = {'S': 1, 'M': 2, 'L': 3}
    df_engineered['company_size_encoded'] = df_engineered['company_size'].map(company_size_map).fillna(2)
    df_engineered['remote_company_size_interaction'] = df_engineered['remote_ratio'] * df_engineered['company_size_encoded']
    df_engineered['location_mismatch'] = (df_engineered['employee_residence'] != df_engineered['company_location_full']).astype(int)
    
    # 4. Education features with safe mapping
    education_map = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    df_engineered['education_level_encoded'] = df_engineered['education_level'].map(education_map).fillna(2)
    df_engineered['has_advanced_degree'] = (df_engineered['education_level'].isin(['Master', 'PhD'])).astype(int)
    df_engineered['education_experience_interaction'] = df_engineered['education_level_encoded'] * df_engineered['years_experience']
    
    # 5. AI features
    df_engineered['ai_usage_intensity'] = df_engineered['ai_tools_hours_per_week'] / (df_engineered['weekly_hours'] + 1)
    
    # 6. Comprehensive seniority
    df_engineered['comprehensive_seniority'] = (
        df_engineered['years_experience'] * 0.3 +
        df_engineered['exp_level_encoded'] * 0.25 +
        df_engineered['education_level_encoded'] * 0.15 +
        df_engineered['certifications_count'] * 0.1 +
        df_engineered['team_size'] * 0.1 +
        df_engineered['manages_people'] * 0.1
    )
    
    # 7. Additional features
    df_engineered['efficiency_score'] = 0
    df_engineered['certs_per_experience_year'] = df_engineered['certifications_count'] / (df_engineered['years_experience'] + 1)
    df_engineered['job_switcher'] = (df_engineered['switched_jobs_last_year'] > 0).astype(int)
    df_engineered['upskilling_dedication'] = (df_engineered['upskilling_hours_per_month'] > 10).astype(int)
    
    # 8. Handle categorical variables with mappings
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
    
    # Apply mappings with safe defaults
    for col, mapping in categorical_mappings.items():
        if col in df_engineered.columns:
            df_engineered[col] = df_engineered[col].map(mapping).fillna(0)
    
    # Also encode other categorical columns using simple mapping
    # Map employment_type
    if 'employment_type' in df_engineered.columns:
        emp_map = {'Full-time': 0, 'Part-time': 1, 'Contract': 2}
        df_engineered['employment_type'] = df_engineered['employment_type'].map(emp_map).fillna(0)
    
    # Map primary_language
    if 'primary_language' in df_engineered.columns:
        lang_map = {'Python': 0, 'SQL': 1, 'R': 2, 'Java': 3, 'C++': 4}
        df_engineered['primary_language'] = df_engineered['primary_language'].map(lang_map).fillna(0)
    
    # Map industry
    if 'industry' in df_engineered.columns:
        industry_map = {'Technology': 0, 'Finance': 1, 'Healthcare': 2, 'Education': 3, 'Retail': 4}
        df_engineered['industry'] = df_engineered['industry'].map(industry_map).fillna(0)
    
    # Select only the features the model expects
    available_features = [f for f in features if f in df_engineered.columns]
    df_engineered = df_engineered[available_features]
    
    # Add missing features with default values
    for f in features:
        if f not in df_engineered.columns:
            df_engineered[f] = 0
    
    # Ensure correct column order
    df_engineered = df_engineered[features]
    
    # Scale numerical features
    numerical_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
    if scaler is not None and len(numerical_cols) > 0:
        try:
            df_engineered[numerical_cols] = scaler.transform(df_engineered[numerical_cols])
        except Exception as e:
            print(f"Warning: Scaling failed: {e}")
            # If scaling fails, proceed without scaling
    
    return df_engineered

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'job_title': request.form.get('job_title', 'Data Scientist'),
            'experience_level': request.form.get('experience_level', 'Mid'),
            'employment_type': request.form.get('employment_type', 'Full-time'),
            'company_size': request.form.get('company_size', 'M'),
            'employee_residence': request.form.get('employee_residence', 'US'),
            'industry': request.form.get('industry', 'Technology'),
            'remote_ratio': float(request.form.get('remote_ratio', 50)),
            'years_experience': float(request.form.get('years_experience', 3)),
            'education_level': request.form.get('education_level', 'Master'),
            'primary_language': request.form.get('primary_language', 'Python'),
            'has_ml_in_title': int(request.form.get('has_ml_in_title', 0)),
            'manages_people': int(request.form.get('manages_people', 0)),
            'team_size': int(request.form.get('team_size', 5)),
            'certifications_count': int(request.form.get('certifications_count', 2)),
            'weekly_hours': float(request.form.get('weekly_hours', 40)),
            'uses_ai_tools_daily': int(request.form.get('uses_ai_tools_daily', 1)),
            'ai_tools_hours_per_week': float(request.form.get('ai_tools_hours_per_week', 10)),
            'salary_currency': 'USD',
            'salary_usd': 0,
            'equity_offered_pct': float(request.form.get('equity_offered_pct', 10)),
            'bonus_pct': float(request.form.get('bonus_pct', 15)),
            'job_satisfaction_score': float(request.form.get('job_satisfaction_score', 7)),
            'interviews_to_offer': int(request.form.get('interviews_to_offer', 3)),
            'switched_jobs_last_year': int(request.form.get('switched_jobs_last_year', 0)),
            'upskilling_hours_per_month': float(request.form.get('upskilling_hours_per_month', 15)),
            'fears_ai_automation_score': float(request.form.get('fears_ai_automation_score', 4)),
            'company_location_full': request.form.get('company_location_full', 'United States'),
            'exp_level_encoded': 2,
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
        
        # Prepare response
        response = {
            'success': True,
            'predicted_salary': round(prediction, 2),
            'predicted_salary_formatted': f"${round(prediction, 2):,.2f}",
            'input_data': data,
            'features_used': len(features)
        }
        
        return jsonify(response)
        
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
    app.run(debug=True, host='0.0.0.0', port=5000)