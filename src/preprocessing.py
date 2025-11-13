import pandas as pd

def load_data(path):
    """Load CSV dataset"""
    df = pd.read_csv(path)
    df.columns = [
        "Relative_Compactness", "Surface_Area", "Wall_Area", "Roof_Area",
        "Overall_Height", "Orientation", "Glazing_Area", "Glazing_Area_Distribution",
        "Heating_Load", "Cooling_Load"
    ]
    return df

def preprocess_features(df):
    """Split features and targets"""
    X = df.drop(['Heating_Load', 'Cooling_Load'], axis=1)
    y = df[['Heating_Load', 'Cooling_Load']]
    return X, y
