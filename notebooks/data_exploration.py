# Title: Energy Efficiency - Data Exploration
# Description: Explore dataset, visualize distributions and correlations

import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import load_data

# Load dataset
df = load_data("data/energy_efficiency_data.csv")

# Quick overview
print(df.head())
print(df.info())
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Feature distributions
df.hist(bins=30, figsize=(15,10), color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()