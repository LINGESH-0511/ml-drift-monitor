from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_validator import DataValidator
from modules.drift_detector import DriftDetector
from modules.model_handler import ModelHandler
from modules.performance_eval import PerformanceEvaluator

app = Flask(__name__)
CORS(app)

# Initialize modules
validator = DataValidator()
drift_detector = DriftDetector()
model_handler = ModelHandler()
evaluator = PerformanceEvaluator()

# Helper function to convert numpy types to Python native types
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'message': 'CredGuard API is running',
        'endpoints': ['/api/health', '/api/analyze']
    })

# Test endpoint
@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({
        'message': 'API is working!', 
        'timestamp': pd.Timestamp.now().isoformat()
    })

# Main analysis endpoint
@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        print("="*60)
        print("📊 Received analysis request")
        print("="*60)
        
        # Get files
        baseline_file = request.files.get('baseline')
        current_file = request.files.get('current')
        model_file = request.files.get('model')
        target_column = request.form.get('target_column', '')
        
        print(f"Files received: baseline={baseline_file is not None}, current={current_file is not None}, model={model_file is not None}")
        print(f"Target column: {target_column}")
        
        if not baseline_file or not current_file or not model_file:
            return jsonify({'error': 'Missing files'}), 400
        
        # Load data
        print("📂 Loading data...")
        baseline_df = pd.read_csv(baseline_file)
        current_df = pd.read_csv(current_file)
        print(f"Baseline shape: {baseline_df.shape}")
        print(f"Current shape: {current_df.shape}")
        print(f"Baseline columns: {list(baseline_df.columns)}")
        
        # Load model
        print("🤖 Loading model...")
        success, msg = model_handler.load_model(model_file)
        if not success:
            print(f"❌ Model loading failed: {msg}")
            return jsonify({'error': msg}), 400
        print("✅ Model loaded successfully")
        
        # Drift detection
        print("📈 Detecting drift...")
        feature_cols = [col for col in baseline_df.columns if col != target_column]
        print(f"Feature columns: {feature_cols}")
        
        drift_results = drift_detector.detect_drift(baseline_df, current_df, feature_cols)
        print(f"Overall drift: {drift_results['overall_drift']}")
        print(f"Drift summary: {drift_results['summary']}")
        
        # Make predictions
        print("🎯 Making predictions...")
        try:
            baseline_predictions = model_handler.predict(baseline_df[feature_cols])
            current_predictions = model_handler.predict(current_df[feature_cols])
            
            # Convert predictions to serializable format
            if baseline_predictions is not None:
                baseline_predictions = convert_to_serializable(baseline_predictions)
            if current_predictions is not None:
                current_predictions = convert_to_serializable(current_predictions)
                
            print(f"Predictions made successfully")
        except Exception as e:
            print(f"Prediction error: {e}")
            baseline_predictions = None
            current_predictions = None
        
        # Performance evaluation if target provided
        performance_results = {}
        if target_column and target_column in baseline_df.columns:
            try:
                print(f"📊 Evaluating performance with target: {target_column}")
                baseline_target = baseline_df[target_column].values
                current_target = current_df[target_column].values
                
                performance_results = evaluator.compare_performance(
                    baseline_predictions, current_predictions, 
                    baseline_target, current_target
                )
                print("Performance evaluation complete")
            except Exception as e:
                print(f"Performance evaluation error: {e}")
                performance_results = {'error': str(e)}
        
        # Convert all results to serializable format
        print("🔄 Converting results to JSON-serializable format...")
        
        # Convert feature results
        feature_results = {}
        for k, v in drift_results['feature_results'].items():
            feature_results[k] = {
                'type': convert_to_serializable(v.get('type')),
                'overall_severity': convert_to_serializable(v.get('overall_severity')),
                'psi': convert_to_serializable(v.get('psi')),
                'change_percent': convert_to_serializable(v.get('change_percent')),
                'chi_square': convert_to_serializable(v.get('chi_square')),
                'baseline_mean': convert_to_serializable(v.get('baseline_mean')),
                'current_mean': convert_to_serializable(v.get('current_mean'))
            }
        
        # Convert summary
        summary = {
            'total_features': convert_to_serializable(drift_results['summary'].get('total_features', 0)),
            'high_drift_features': convert_to_serializable(drift_results['summary'].get('high_drift_features', 0)),
            'medium_drift_features': convert_to_serializable(drift_results['summary'].get('medium_drift_features', 0)),
            'low_drift_features': convert_to_serializable(drift_results['summary'].get('low_drift_features', 0))
        }
        
        # Prepare response
        response = {
            'drift_analysis': {
                'overall_drift': convert_to_serializable(drift_results.get('overall_drift', 'UNKNOWN')),
                'summary': summary,
                'feature_results': feature_results
            },
            'performance_analysis': convert_to_serializable(performance_results),
            'recommendations': {
                'overall_drift': convert_to_serializable(drift_results.get('overall_drift', 'UNKNOWN')),
                'high_drift_features': summary['high_drift_features'],
                'medium_drift_features': summary['medium_drift_features'],
                'low_drift_features': summary['low_drift_features']
            }
        }
        
        print("✅ Analysis complete!")
        print("="*60)
        
        # Return the response
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting CredGuard Backend API...")
    print("📍 API running at http://localhost:5000")
    print("📍 Health check: http://localhost:5000/api/health")
    print("📍 Test endpoint: http://localhost:5000/api/test")
    print("="*60)
    app.run(debug=True, port=5000, host='0.0.0.0')
