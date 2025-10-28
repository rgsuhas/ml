import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

def create_features(df):
    """Create new features for better prediction"""
    df = df.copy()
    
    # Create price per km feature
    df['price_per_km'] = df['selling_price'] / df['km_driven']
    
    # Create power per engine size feature
    df['power_per_engine'] = df['max_power'] / df['engine']
    
    # Create age-related features
    df['km_per_year'] = df['km_driven'] / df['vehicle_age']
    
    # Create interaction features
    df['age_power'] = df['vehicle_age'] * df['max_power']
    df['mileage_engine'] = df['mileage'] * df['engine']
    
    return df

def train_advanced_models(X, y):
    """Train multiple advanced models with hyperparameter tuning"""
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models with tuned hyperparameters
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        min_child_weight=2,
        random_state=42
    )
    
    # Train models
    models = {
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'XGBoost': xgb_model
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"\n{name} Results:")
        print(f"Training R² Score: {train_score:.4f}")
        print(f"Testing R² Score: {test_score:.4f}")
        print(f"RMSE: {rmse:.2f}")
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return models, scaler

def predict_price(models, scaler, sample_input):
    """Make ensemble prediction using all models"""
    # Scale the input
    sample_scaled = scaler.transform(sample_input)
    
    # Get predictions from all models
    predictions = {}
    for name, model in models.items():
        pred = model.predict(sample_scaled)[0]
        predictions[name] = pred
    
    # Calculate ensemble prediction (weighted average)
    weights = {
        'Random Forest': 0.4,
        'Gradient Boosting': 0.3,
        'XGBoost': 0.3
    }
    
    ensemble_pred = sum(pred * weights[name] for name, pred in predictions.items())
    
    # Print individual and ensemble predictions
    print("\nModel Predictions:")
    for name, pred in predictions.items():
        print(f"{name}: ₹{pred:,.2f}")
    print(f"\nEnsemble Prediction: ₹{ensemble_pred:,.2f}")
    
    return ensemble_pred

# Example usage:
if __name__ == "__main__":
    # Load your data and prepare features
    # df = pd.read_csv('your_car_data.csv')
    # df = create_features(df)
    
    # Prepare your features (X) and target (y)
    # X = df[['vehicle_age', 'km_driven', ...]]
    # y = df['selling_price']
    
    # Train models
    # models, scaler = train_advanced_models(X, y)
    
    # Make prediction for a sample
    sample_input = {
        'vehicle_age': 9,
        'km_driven': 120000,
        'mileage': 19.7,
        'engine': 796,
        'max_power': 46.3,
        'brand_Maruti': 1,
        'model_Swift': 0,
        'seller_type_Individual': 1,
        'fuel_type_Petrol': 1,
        'transmission_type_Manual': 1
    }
    # predict_price(models, scaler, pd.DataFrame([sample_input]))
