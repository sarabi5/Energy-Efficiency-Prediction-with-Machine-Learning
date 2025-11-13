# notebooks/ML_models.py
# Title: Energy Efficiency - Traditional ML Models
# Description: Train and evaluate Linear, Ridge, Random Forest models
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.preprocessing import load_data, preprocess_features
from src.models import train_linear_models, train_random_forest
# from src.evaluation import evaluate_models # No longer needed
from src.utils import plot_feature_importance # Import plotter

# --- MLflow Imports ---
import mlflow
import mlflow.sklearn
# ----------------------

# Set the experiment name. This will create a new experiment in the MLflow UI.
mlflow.set_experiment("Energy Efficiency - ML")

# Load dataset
df = load_data("data/energy_efficiency_data.csv")

# Split features and targets
X, y = preprocess_features(df)
# Get feature names BEFORE scaling and splitting
feature_names = X.columns.tolist() 

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train Linear Regression ---
with mlflow.start_run(run_name="Linear Regression"):
    print("Training Linear Regression...")
    results = train_linear_models(X_train_scaled, y_train, X_test_scaled, y_test, model_type='linear')
    
    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("r2_score", results["r2_score"])
    mlflow.sklearn.log_model(results["model"], "model")
    
    print(f"Linear Regression R²: {results['r2_score']:.4f}")

# --- Train Ridge Regression ---
with mlflow.start_run(run_name="Ridge Regression (GridSearch)"):
    print("Training Ridge Regression...")
    results = train_linear_models(X_train_scaled, y_train, X_test_scaled, y_test, model_type='ridge')
    
    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "Ridge")
    mlflow.log_params(results["best_params"]) # Logs the 'alpha'
    mlflow.log_metric("r2_score", results["r2_score"])
    mlflow.sklearn.log_model(results["model"], "model")

    print(f"Ridge Regression R²: {results['r2_score']:.4f}")
    print(f"Best Ridge Params: {results['best_params']}")

# --- Train Random Forest ---
# Note: RF works better on unscaled features, but we log both
with mlflow.start_run(run_name="Random Forest (GridSearch)"):
    print("Training Random Forest...")
    # Using unscaled data for RF as in original code
    results = train_random_forest(X_train, y_train, X_test, y_test, feature_names=feature_names)
    
    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_params(results["best_params"]) # Logs n_estimators, max_depth, etc.
    mlflow.log_metric("r2_score", results["r2_score"])
    mlflow.sklearn.log_model(results["model"], "model")

    print(f"Random Forest R²: {results['r2_score']:.4f}")
    print(f"Best RF Params: {results['best_params']}")

    # --- Log Feature Importance Plot as an Artifact ---
    plot_path = "reports/rf_feature_importance.png"
    plot_feature_importance(
        results['feature_names'], 
        results['feature_importances'],
        save_path=plot_path
    )
    # Log the plot file
    mlflow.log_artifact(plot_path)

print("\n--- Model training complete! Check the MLflow UI at http://127.0.0.1:5000 ---")