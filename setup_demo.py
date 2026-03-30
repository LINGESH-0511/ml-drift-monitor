import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("="*60)
print("CredGuard Demo Data Generator")
print("="*60)

# Create demo_data directory
os.makedirs('demo_data', exist_ok=True)

# Generate baseline data
print("\n📊 Generating baseline data...")
np.random.seed(42)
n_samples = 5000

baseline_data = pd.DataFrame({
    'age': np.random.normal(35, 10, n_samples).clip(18, 80).astype(int),
    'income': np.random.normal(55000, 20000, n_samples).clip(20000, 200000).astype(int),
    'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], 
                                        n_samples, p=[0.6, 0.15, 0.15, 0.1]),
    'credit_score': np.random.normal(680, 50, n_samples).clip(300, 850).astype(int),
    'debt_to_income_ratio': np.random.uniform(0, 0.5, n_samples).round(3),
    'loan_amount': np.random.uniform(1000, 50000, n_samples).astype(int)
})

# Generate target (default probability)
default_prob = 1 / (1 + np.exp(-(
    -3 + 
    0.02 * ((baseline_data['credit_score'] - 680) / 100) -  
    0.00001 * ((baseline_data['income'] - 55000) / 10000) +   
    2 * baseline_data['debt_to_income_ratio']
)))
baseline_data['default'] = (np.random.random(n_samples) < default_prob).astype(int)

baseline_data.to_csv('demo_data/baseline_demo.csv', index=False)
print(f"✓ Baseline data saved: {len(baseline_data)} samples")

# Generate drifted data
print("\n📈 Generating drifted data (with income and credit score shift)...")
np.random.seed(43)
n_drifted = 2000

drifted_data = pd.DataFrame({
    'age': np.random.normal(35, 10, n_drifted).clip(18, 80).astype(int),
    'income': np.random.normal(45000, 15000, n_drifted).clip(20000, 200000).astype(int),
    'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], 
                                        n_drifted, p=[0.5, 0.2, 0.2, 0.1]),
    'credit_score': np.random.normal(650, 55, n_drifted).clip(300, 850).astype(int),
    'debt_to_income_ratio': np.random.uniform(0.1, 0.6, n_drifted).round(3),
    'loan_amount': np.random.uniform(1000, 50000, n_drifted).astype(int)
})

default_prob_drifted = 1 / (1 + np.exp(-(
    -2.5 + 
    0.02 * ((drifted_data['credit_score'] - 650) / 100) -  
    0.00001 * ((drifted_data['income'] - 45000) / 10000) +   
    2 * drifted_data['debt_to_income_ratio']
)))
drifted_data['default'] = (np.random.random(n_drifted) < default_prob_drifted).astype(int)

drifted_data.to_csv('demo_data/current_demo_drift.csv', index=False)
print(f"✓ Drifted data saved: {len(drifted_data)} samples")

# Train model
print("\n🤖 Training demo model...")
feature_cols = ['age', 'income', 'employment_type', 'credit_score', 'debt_to_income_ratio', 'loan_amount']
X = pd.get_dummies(baseline_data[feature_cols], drop_first=True)
y = baseline_data['default']

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

joblib.dump(model, 'demo_data/demo_model.pkl')
print("✓ Model saved: demo_model.pkl")

print("\n" + "="*60)
print("✅ Demo setup complete!")
print("\n📁 Files created in 'demo_data' folder:")
print("   • baseline_demo.csv (5000 samples)")
print("   • current_demo_drift.csv (2000 samples)")
print("   • demo_model.pkl (RandomForest model)")
print("\n🚀 To run CredGuard:")
print("   1. Run: streamlit run app.py")
print("   2. Upload these three files")
print("   3. Enter 'default' as target column")
print("   4. Click 'Start Analysis'")
print("="*60)
