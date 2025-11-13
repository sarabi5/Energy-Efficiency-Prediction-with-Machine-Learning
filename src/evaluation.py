# src/evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score # Use sklearn's r2_score

def evaluate_models(results_dict):
    """
    This function is no longer needed.
    The MLflow UI will be used to compare models.
    """
    pass

def evaluate_deep_learning(model, history, X_test, y_test, plot_path="reports/dl_loss.png"):
    preds = model.predict(X_test)
    
    # Calculate R2 scores
    r2_heat = r2_score(y_test['Heating_Load'], preds[:, 0])
    r2_cool = r2_score(y_test['Cooling_Load'], preds[:, 1])
    
    print(f"Heating Load R²: {r2_heat:.4f}")
    print(f"Cooling Load R²: {r2_cool:.4f}")
    
    # Plot and save training history
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Deep Learning Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(plot_path)
    plt.close()
    
    # Return metrics and plot path to be logged
    return {
        "heating_load_r2": r2_heat,
        "cooling_load_r2": r2_cool,
        "loss_plot_path": plot_path
    }
