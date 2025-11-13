# notebooks/DL_models.py
# Title: Energy Efficiency - Deep Learning Model
# Description: Build and train multi-output neural network

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.preprocessing import load_data, preprocess_features
from src.models import build_multitarget_model
from src.evaluation import evaluate_deep_learning
import tensorflow as tf

# --- MLflow Imports ---
import mlflow
import mlflow.tensorflow
# ----------------------

# Set the experiment name
mlflow.set_experiment("Energy Efficiency - DL")

# --- Enable MLflow Autologging for TensorFlow ---
# This will automatically log model parameters, metrics per epoch,
# and the final trained model.
mlflow.tensorflow.autolog()
# ------------------------------------------------

# Load dataset
df = load_data("data/energy_efficiency_data.csv")

# Split features and targets
X, y = preprocess_features(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Start the MLflow Run ---
# Autolog will log to this run
with mlflow.start_run(run_name="Multi-Target NN"):
    
    # Log key parameters that autolog might miss
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # Build model
    model = build_multitarget_model(X_train_scaled.shape[1])
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
    ]
    
    # Train model
    print("Training Deep Learning model...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=16,
        callbacks=callbacks,
        verbose=0 # Set to 1 or 2 if you want to see progress
    )

    # Evaluate (and create/save the loss plot)
    eval_results = evaluate_deep_learning(model, history, X_test_scaled, y_test)
    
    # Manually log the final R2 scores and the loss plot
    mlflow.log_metrics({
        "final_heating_r2": eval_results["heating_load_r2"],
        "final_cooling_r2": eval_results["cooling_load_r2"]
    })
    mlflow.log_artifact(eval_results["loss_plot_path"])
    
    print("\n--- DL model training complete! Check the MLflow UI. ---")