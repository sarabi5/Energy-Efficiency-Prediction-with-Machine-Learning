# src/utils.py
# import joblib - NO LONGER NEEDED
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# def save_model(model, path):
#     """This function is no longer needed.
#     Use mlflow.sklearn.log_model() or mlflow.tensorflow.log_model()
#     """
#     pass

def plot_feature_importance(features, importances, save_path, title="Feature Importance"):
    """Barplot of feature importance, saved to a file."""
    df = pd.DataFrame({'Feature': features, 'Importance': importances})
    df = df.sort_values(by='Importance', ascending=False) # Changed to descending
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=df)
    plt.title(title)
    plt.tight_layout()
    
    # Save the figure instead of showing it
    plt.savefig(save_path)
    plt.close() # Close the plot to free up memory
