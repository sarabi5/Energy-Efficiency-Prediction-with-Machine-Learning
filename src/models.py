# src/models.py
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# ---------------- Traditional ML ----------------
def train_linear_models(X_train, y_train, X_test, y_test, model_type='linear'):
    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        
        return {
            "model": model,
            "best_params": {},  # No hyperparameters for basic linear regression
            "r2_score": score
        }
        
    elif model_type == 'ridge':
        params = {'alpha': [0.01, 0.1, 1, 10, 100]}
        grid = GridSearchCV(Ridge(), params, cv=5, scoring='r2')
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_score = grid.best_score_ # Or use r2_score on X_test if preferred
        
        return {
            "model": best_model,
            "best_params": best_params,
            "r2_score": best_score
        }

def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    params = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
    rf = GridSearchCV(RandomForestRegressor(random_state=42), params, cv=5, scoring='r2', n_jobs=-1)
    rf.fit(X_train, y_train)
    
    best_model = rf.best_estimator_
    best_params = rf.best_params_
    
    # Evaluate on the test set
    preds = best_model.predict(X_test)
    score = r2_score(y_test, preds)
    
    # Get feature importances
    importances = best_model.feature_importances_
    
    return {
        "model": best_model,
        "best_params": best_params,
        "r2_score": score,
        "feature_names": feature_names,
        "feature_importances": importances
    }

# ---------------- Deep Learning ----------------
def build_multitarget_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(2)  # Two outputs
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model
