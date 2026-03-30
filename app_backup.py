import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_validator import DataValidator
from modules.drift_detector import DriftDetector
from modules.model_handler import ModelHandler
from modules.performance_eval import PerformanceEvaluator
from modules.report_generator import ReportGenerator

st.set_page_config(page_title="CredGuard - ML Model Drift Monitor", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .drift-high { background-color: #ff4757; padding: 0.5rem; border-radius: 5px; color: white; font-weight: bold; text-align: center; }
    .drift-medium { background-color: #ffa502; padding: 0.5rem; border-radius: 5px; color: white; font-weight: bold; text-align: center; }
    .drift-low { background-color: #26de81; padding: 0.5rem; border-radius: 5px; color: white; font-weight: bold; text-align: center; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 0.5rem; }
    .recommendation-box { background: #f0f2f6; padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #667eea; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🛡️ CredGuard</h1><h3>ML Model Drift Monitoring System</h3><p>Detect silent model failures before they impact your business</p></div>', unsafe_allow_html=True)

modules = {
    'validator': DataValidator(),
    'drift_detector': DriftDetector(),
    'model_handler': ModelHandler(),
    'evaluator': PerformanceEvaluator(),
    'report_generator': ReportGenerator()
}

with st.sidebar:
    st.header("📁 Upload Files")
    baseline_file = st.file_uploader("Baseline Dataset (Training Data)", type=['csv'])
    current_file = st.file_uploader("Current Dataset (Production Data)", type=['csv'])
    model_file = st.file_uploader("Trained Model", type=['pkl', 'joblib'])
    target_col = st.text_input("Target Column (Optional)", help="Column name for performance evaluation")
    analyze_button = st.button("🚀 Start Analysis", type="primary", use_container_width=True)

if analyze_button:
    if not baseline_file or not current_file or not model_file:
        st.error("❌ Please upload all required files")
    else:
        with st.spinner("🔍 Analyzing your data..."):
            try:
                baseline_df = pd.read_csv(baseline_file)
                current_df = pd.read_csv(current_file)
                
                st.subheader("📊 Data Validation")
                validation_results = modules['validator'].full_validation(baseline_df, current_df, target_col if target_col else None)
                
                if validation_results['success']:
                    st.success("✅ Data validation passed!")
                else:
                    st.error("❌ Data validation failed!")
                    for error in validation_results['errors']:
                        st.error(f"- {error}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Baseline Samples", f"{baseline_df.shape[0]:,}")
                with col2:
                    st.metric("Current Samples", f"{current_df.shape[0]:,}")
                
                st.subheader("🤖 Model Loading")
                success, message = modules['model_handler'].load_model(model_file)
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
                    st.stop()
                
                st.subheader("📈 Drift Detection")
                feature_cols = [col for col in baseline_df.columns if col != target_col]
                drift_results = modules['drift_detector'].detect_drift(baseline_df, current_df, feature_cols)
                
                overall_drift = drift_results['overall_drift']
                drift_class = f"drift-{overall_drift.lower()}"
                st.markdown(f'<div class="{drift_class}">Overall Drift: {overall_drift}</div>', unsafe_allow_html=True)
                
                drift_table = modules['drift_detector'].get_drift_report_table(drift_results)
                st.dataframe(drift_table, use_container_width=True)
                
                st.subheader("🎯 Model Performance")
                baseline_predictions = modules['model_handler'].predict(baseline_df[feature_cols])
                current_predictions = modules['model_handler'].predict(current_df[feature_cols])
                
                baseline_target = baseline_df[target_col].values if target_col and target_col in baseline_df.columns else None
                current_target = current_df[target_col].values if target_col and target_col in current_df.columns else None
                
                performance_results = modules['evaluator'].compare_performance(
                    baseline_predictions, current_predictions, baseline_target, current_target
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(values=list(performance_results['baseline']['prediction_distribution'].values()),
                                 names=list(performance_results['baseline']['prediction_distribution'].keys()),
                                 title="Baseline Predictions")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.pie(values=list(performance_results['current']['prediction_distribution'].values()),
                                 names=list(performance_results['current']['prediction_distribution'].keys()),
                                 title="Current Predictions")
                    st.plotly_chart(fig, use_container_width=True)
                
                if baseline_target is not None:
                    metrics_df = pd.DataFrame()
                    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                        baseline_val = performance_results['baseline']['metrics'][metric]
                        current_val = performance_results['current']['metrics'][metric]
                        metrics_df[metric] = [baseline_val, current_val, current_val - baseline_val]
                    metrics_df.index = ['Baseline', 'Current', 'Change']
                    st.dataframe(metrics_df.round(4), use_container_width=True)
                
                st.subheader("💡 Recommendations")
                recommendations = []
                if drift_results['overall_drift'] == 'HIGH':
                    recommendations.append("🔴 **URGENT:** Severe data drift detected. Retrain model immediately.")
                elif drift_results['overall_drift'] == 'MEDIUM':
                    recommendations.append("🟡 **WARNING:** Moderate data drift detected. Plan model refresh within 1-2 weeks.")
                else:
                    recommendations.append("🟢 **STABLE:** No significant drift detected. Continue monitoring.")
                
                if performance_results.get('health_score', {}).get('status') == 'DEGRADED':
                    recommendations.append("⚠️ Model performance is degrading. Consider retraining.")
                
                for rec in recommendations:
                    st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)
                
                st.subheader("📄 Export Reports")
                user_inputs = {
                    'baseline_filename': baseline_file.name,
                    'current_filename': current_file.name,
                    'model_filename': model_file.name,
                    'target_column': target_col if target_col else None
                }
                
                report = modules['report_generator'].generate_report(validation_results, drift_results, performance_results, user_inputs)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Download JSON Report"):
                        json_path = modules['report_generator'].export_json()
                        with open(json_path, 'rb') as f:
                            st.download_button("Click to Download", f, file_name=os.path.basename(json_path), mime="application/json")
                with col2:
                    if st.button("📋 Download CSV Summary"):
                        csv_path = modules['report_generator'].export_csv_summary()
                        with open(csv_path, 'rb') as f:
                            st.download_button("Click to Download", f, file_name=os.path.basename(csv_path), mime="text/csv")
                
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'><p>CredGuard - Protecting ML Models from Silent Failures</p></div>", unsafe_allow_html=True)
