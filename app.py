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

# Custom CSS for better visibility
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Drift indicators */
    .drift-high { 
        background-color: #dc3545; 
        padding: 1rem; 
        border-radius: 10px; 
        color: white; 
        font-weight: bold; 
        text-align: center; 
        font-size: 1.3rem;
        margin: 1rem 0;
    }
    .drift-medium { 
        background-color: #ffc107; 
        padding: 1rem; 
        border-radius: 10px; 
        color: #2c3e50; 
        font-weight: bold; 
        text-align: center;
        font-size: 1.3rem;
        margin: 1rem 0;
    }
    .drift-low { 
        background-color: #28a745; 
        padding: 1rem; 
        border-radius: 10px; 
        color: white; 
        font-weight: bold; 
        text-align: center;
        font-size: 1.3rem;
        margin: 1rem 0;
    }
    
    /* Recommendation boxes */
    .recommendation-critical {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        font-size: 1rem;
        border-left: 5px solid #ff6b6b;
    }
    .recommendation-warning {
        background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #2c3e50;
        font-size: 1rem;
        border-left: 5px solid #ffa502;
    }
    .recommendation-info {
        background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        font-size: 1rem;
        border-left: 5px solid #26de81;
    }
    
    /* Performance card */
    .performance-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🛡️ CredGuard</h1><h3>ML Model Drift Monitoring System</h3><p>Detect silent model failures before they impact your business</p></div>', unsafe_allow_html=True)

# Initialize modules
modules = {
    'validator': DataValidator(),
    'drift_detector': DriftDetector(),
    'model_handler': ModelHandler(),
    'evaluator': PerformanceEvaluator(),
    'report_generator': ReportGenerator()
}

# Sidebar for file uploads
with st.sidebar:
    st.header("📁 Upload Files")
    baseline_file = st.file_uploader("Baseline Dataset (Training Data)", type=['csv'])
    current_file = st.file_uploader("Current Dataset (Production Data)", type=['csv'])
    model_file = st.file_uploader("Trained Model", type=['pkl', 'joblib'])
    target_col = st.text_input("Target Column (Optional)", help="e.g., 'default', 'churn'")
    analyze_button = st.button("🚀 Start Analysis", type="primary", use_container_width=True)

# Analysis results
if analyze_button:
    if not baseline_file or not current_file or not model_file:
        st.error("❌ Please upload all required files")
    else:
        with st.spinner("🔍 Analyzing your data..."):
            try:
                # Load data
                baseline_df = pd.read_csv(baseline_file)
                current_df = pd.read_csv(current_file)
                
                # Data Validation
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
                
                # Load Model
                st.subheader("🤖 Model Loading")
                success, message = modules['model_handler'].load_model(model_file)
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
                    st.stop()
                
                # Drift Detection
                st.subheader("📈 Drift Detection")
                feature_cols = [col for col in baseline_df.columns if col != target_col]
                drift_results = modules['drift_detector'].detect_drift(baseline_df, current_df, feature_cols)
                
                # Display overall drift
                overall_drift = drift_results['overall_drift']
                if overall_drift == 'HIGH':
                    st.markdown('<div class="drift-high">🔴 SEVERE DRIFT DETECTED - Immediate Action Required</div>', unsafe_allow_html=True)
                elif overall_drift == 'MEDIUM':
                    st.markdown('<div class="drift-medium">🟡 MODERATE DRIFT DETECTED - Plan Model Refresh</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="drift-low">🟢 LOW DRIFT - Model Stable</div>', unsafe_allow_html=True)
                
                # Display drift table
                drift_table = modules['drift_detector'].get_drift_report_table(drift_results)
                st.dataframe(drift_table, use_container_width=True)
                
                # Model Performance Section
                st.subheader("🎯 Model Performance Analysis")
                
                # Make predictions
                baseline_predictions = modules['model_handler'].predict(baseline_df[feature_cols])
                current_predictions = modules['model_handler'].predict(current_df[feature_cols])
                
                # Convert to numpy arrays if needed
                if baseline_predictions is not None:
                    baseline_predictions = np.array(baseline_predictions)
                    current_predictions = np.array(current_predictions)
                    
                    # Calculate prediction distributions safely
                    baseline_unique, baseline_counts = np.unique(baseline_predictions, return_counts=True)
                    current_unique, current_counts = np.unique(current_predictions, return_counts=True)
                    
                    baseline_dist = baseline_counts / len(baseline_predictions)
                    current_dist = current_counts / len(current_predictions)
                    
                    # Create columns for charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Training Data Predictions")
                        fig1 = px.pie(
                            values=baseline_dist,
                            names=[f"Class {int(x)}" for x in baseline_unique],
                            title="Predictions on Training Data",
                            color_discrete_sequence=['#667eea', '#764ba2', '#f39c12', '#e74c3c'],
                            hole=0.3
                        )
                        fig1.update_layout(
                            template="simple_white",
                            height=450,
                            showlegend=True,
                            font=dict(size=14)
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Calculate default rate (assuming class 1 is positive/default)
                        default_rate_baseline = (baseline_predictions == 1).sum() / len(baseline_predictions) * 100
                        st.info(f"📈 **Predicted Default Rate:** {default_rate_baseline:.1f}%")
                        st.caption(f"Total predictions: {len(baseline_predictions):,}")
                    
                    with col2:
                        st.markdown("### 📊 Production Data Predictions")
                        fig2 = px.pie(
                            values=current_dist,
                            names=[f"Class {int(x)}" for x in current_unique],
                            title="Predictions on Production Data",
                            color_discrete_sequence=['#667eea', '#764ba2', '#f39c12', '#e74c3c'],
                            hole=0.3
                        )
                        fig2.update_layout(
                            template="simple_white",
                            height=450,
                            showlegend=True,
                            font=dict(size=14)
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Calculate default rate for production
                        default_rate_current = (current_predictions == 1).sum() / len(current_predictions) * 100
                        st.info(f"📈 **Predicted Default Rate:** {default_rate_current:.1f}%")
                        st.caption(f"Total predictions: {len(current_predictions):,}")
                    
                    # Show prediction shift
                    rate_change = default_rate_current - default_rate_baseline
                    st.markdown("---")
                    if abs(rate_change) > 10:
                        st.warning(f"⚠️ **Significant Prediction Shift:** Default rate changed by {rate_change:+.1f}%")
                    elif abs(rate_change) > 5:
                        st.info(f"📊 **Moderate Prediction Shift:** Default rate changed by {rate_change:+.1f}%")
                    else:
                        st.success(f"✅ **Stable Predictions:** Default rate changed by {rate_change:+.1f}%")
                
                # Performance metrics if target available
                if target_col and target_col in baseline_df.columns and target_col in current_df.columns:
                    st.markdown("---")
                    st.markdown("### 📈 Performance Metrics Comparison")
                    
                    baseline_target = baseline_df[target_col].values
                    current_target = current_df[target_col].values
                    
                    performance_results = modules['evaluator'].compare_performance(
                        baseline_predictions, current_predictions, baseline_target, current_target
                    )
                    
                    if 'metrics' in performance_results['baseline'] and 'metrics' in performance_results['current']:
                        # Create metrics dataframe
                        metrics_list = []
                        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                            if metric in performance_results['baseline']['metrics']:
                                baseline_val = performance_results['baseline']['metrics'][metric]
                                current_val = performance_results['current']['metrics'][metric]
                                change = current_val - baseline_val
                                change_pct = (change / baseline_val) * 100 if baseline_val > 0 else 0
                                
                                # Determine trend indicator
                                if change > 0.05:
                                    trend = "🟢 Improving"
                                elif change < -0.05:
                                    trend = "🔴 Degrading"
                                else:
                                    trend = "🟡 Stable"
                                
                                metrics_list.append({
                                    'Metric': metric.upper(),
                                    'Training Data': f"{baseline_val:.3f}",
                                    'Production Data': f"{current_val:.3f}",
                                    'Change': f"{change:+.3f} ({change_pct:+.1f}%)",
                                    'Status': trend
                                })
                        
                        if metrics_list:
                            metrics_df = pd.DataFrame(metrics_list)
                            st.dataframe(metrics_df, use_container_width=True)
                            
                            # Show performance summary
                            if change < -0.05:
                                st.error("⚠️ **Model Performance is Degrading!** Consider retraining immediately.")
                            elif change < -0.02:
                                st.warning("📉 **Performance is Declining.** Plan for model refresh soon.")
                            else:
                                st.success("✅ **Model Performance is Stable.** Continue monitoring.")
                
                # Actionable Recommendations
                st.subheader("💡 Actionable Recommendations")
                
                # Generate clear recommendations based on drift
                if overall_drift == 'HIGH':
                    st.markdown("""
                    <div class="recommendation-critical">
                        <strong>🚨 URGENT ACTION REQUIRED</strong><br><br>
                        • <strong>Immediate Retraining Needed:</strong> Your model is experiencing severe data drift<br>
                        • <strong>Action Plan:</strong> Collect recent production data and retrain model within 48 hours<br>
                        • <strong>Impact:</strong> Model accuracy and reliability are at risk<br>
                        • <strong>Root Cause:</strong> Significant changes detected in key features
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # List high drift features
                    high_drift_features = [f for f, r in drift_results['feature_results'].items() 
                                         if r.get('overall_severity') == 'HIGH']
                    if high_drift_features:
                        st.markdown(f"""
                        <div class="recommendation-warning">
                            <strong>📊 Features with Critical Drift</strong><br><br>
                            • High drift detected in: {', '.join(high_drift_features)}<br>
                            • These features have changed significantly from training data<br>
                            • Investigate: Economic changes, data collection issues, or business process changes
                        </div>
                        """, unsafe_allow_html=True)
                
                elif overall_drift == 'MEDIUM':
                    st.markdown("""
                    <div class="recommendation-warning">
                        <strong>⚠️ PLAN MODEL REFRESH WITHIN 2 WEEKS</strong><br><br>
                        • <strong>Recommended Action:</strong> Schedule model retraining in the next 1-2 weeks<br>
                        • <strong>Monitoring:</strong> Track drift trends weekly to catch accelerating changes<br>
                        • <strong>Data Collection:</strong> Start collecting more recent training data<br>
                        • <strong>Preventive:</strong> Early action prevents severe degradation
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # List medium drift features
                    medium_drift_features = [f for f, r in drift_results['feature_results'].items() 
                                           if r.get('overall_severity') == 'MEDIUM']
                    if medium_drift_features:
                        st.markdown(f"""
                        <div class="recommendation-info">
                            <strong>📊 Features Showing Moderate Drift</strong><br><br>
                            • Moderate drift in: {', '.join(medium_drift_features)}<br>
                            • These features are starting to shift - ideal time to plan model update<br>
                            • Review feature importance to understand impact
                        </div>
                        """, unsafe_allow_html=True)
                
                else:
                    st.markdown("""
                    <div class="recommendation-info">
                        <strong>✅ MODEL IS STABLE</strong><br><br>
                        • <strong>Status:</strong> No significant drift detected - model is performing well<br>
                        • <strong>Action:</strong> Continue regular monitoring (weekly or monthly)<br>
                        • <strong>Confidence:</strong> Current model is safe to use in production<br>
                        • <strong>Best Practice:</strong> Schedule regular reviews every 2-3 months
                    </div>
                    """, unsafe_allow_html=True)
                
                # Performance-based recommendations if metrics available
                if target_col and target_col in baseline_df.columns and 'metrics_list' in locals():
                    if change < -0.05:
                        st.markdown("""
                        <div class="recommendation-critical">
                            <strong>📉 SEVERE PERFORMANCE DEGRADATION DETECTED</strong><br><br>
                            • <strong>Immediate Action:</strong> Model accuracy has dropped significantly<br>
                            • <strong>Business Impact:</strong> Incorrect predictions may affect decisions<br>
                            • <strong>Solution:</strong> Retrain model with recent data and validate before deployment<br>
                            • <strong>Rollback Plan:</strong> Consider using previous model version if available
                        </div>
                        """, unsafe_allow_html=True)
                    elif change < -0.02:
                        st.markdown("""
                        <div class="recommendation-warning">
                            <strong>📉 PERFORMANCE DECLINING</strong><br><br>
                            • <strong>Action Required:</strong> Model metrics are showing degradation<br>
                            • <strong>Timeline:</strong> Plan retraining within the next month<br>
                            • <strong>Investigation:</strong> Review if business conditions or data quality have changed<br>
                            • <strong>Monitoring:</strong> Increase monitoring frequency to bi-weekly
                        </div>
                        """, unsafe_allow_html=True)
                
                # Summary
                st.markdown("---")
                st.success("✅ **Analysis Complete!** Use the recommendations above to maintain your model's performance.")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; padding: 1rem;'><p>CredGuard - Protecting ML Models from Silent Failures | Monitor. Detect. Act.</p></div>", unsafe_allow_html=True)
