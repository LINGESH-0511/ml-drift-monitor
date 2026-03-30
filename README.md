# 🛡️ CredGuard - ML Model Drift Monitoring System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.2.0-blue.svg)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## 📊 Overview

**CredGuard** is a production-ready ML model monitoring system that detects data drift and prevents silent model failures. It helps data scientists and ML engineers identify when their models start degrading due to changing data patterns before they impact business decisions.

### 🎯 Why CredGuard?

ML models in production suffer from **silent failures** because the data they encounter changes over time (data drift), while the model remains static. This leads to:
- 📉 Degraded model performance
- 💰 Financial losses from incorrect predictions
- ⚖️ Compliance and regulatory issues
- 🚨 No visible errors until significant business impact occurs

**CredGuard catches these issues early!**

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔄 **Universal Compatibility** | Works with ANY ML model (.pkl, .joblib) and ANY CSV dataset |
| 📊 **Data Validation** | Automatic column matching, data type checking, missing value detection |
| 📈 **Drift Detection** | PSI, Chi-square, and KS tests for numerical & categorical features |
| 🎯 **Model Prediction Analysis** | Compare predictions between baseline and new data |
| 📉 **Performance Evaluation** | Accuracy, precision, recall, F1 score comparison |
| 🎨 **Interactive Dashboard** | Modern, professional UI with real-time visualizations |
| 📄 **Exportable Reports** | JSON and CSV formats for documentation |
| 🚀 **Real-time Analysis** | Fast processing with instant results |


