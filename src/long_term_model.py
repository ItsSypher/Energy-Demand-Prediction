import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from external_data import get_external_data

class LongTermModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            n_jobs=-1,
            random_state=42
        )
        self.features = [
            'month', 'group_id', 
            'lag_12',
            'rolling_mean_12_3', # Mean of last 3 months from 1 year ago
            'temperature', 'wind_speed', 'wind_power_production' # External features
        ]

    def train(self, df_train, df_val=None):
        print("Training LongTermModel...")
        print(f"Train shape: {df_train.shape}")
        if df_train.empty:
            raise ValueError("Training data is empty!")
            
        X_train = df_train[self.features]
        y_train = df_train['consumption']
        
        eval_set = []
        if df_val is not None:
            X_val = df_val[self.features]
            y_val = df_val['consumption']
            eval_set = [(X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=10
        )

    def save(self, path):
        self.model.save_model(path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.model.load_model(path)
        print(f"Model loaded from {path}")

    def predict(self, df_features):
        X = df_features[self.features]
        return self.model.predict(X)

def prepare_monthly_data(df_cons, df_groups):
    print("Aggregating to monthly data...")
    # Melt to long format
    # df_cons is raw, so measured_at is a column.
    df_long = df_cons.melt(id_vars=['measured_at'], var_name='group_id', value_name='consumption')
    df_long['group_id'] = df_long['group_id'].astype(int)
    
    # Resample to monthly sum
    df_long['measured_at'] = pd.to_datetime(df_long['measured_at'], utc=True)
    df_monthly = df_long.set_index('measured_at').groupby('group_id').resample('MS')['consumption'].sum().reset_index()
    
    # Merge group features
    df_monthly = df_monthly.merge(df_groups, on='group_id', how='left')
    
    # --- External Data Integration ---
    print("Fetching and merging external data for training...")
    start_date = df_monthly['measured_at'].min()
    end_date = df_monthly['measured_at'].max() + pd.Timedelta(days=31) # Ensure coverage
    
    df_ext = get_external_data(start_date, end_date)
    
    # Aggregate external data to monthly mean
    df_ext['measured_at'] = pd.to_datetime(df_ext['measured_at'], utc=True)
    df_ext_monthly = df_ext.set_index('measured_at').resample('MS').mean().reset_index()
    
    # Merge
    df_monthly = df_monthly.merge(df_ext_monthly, on='measured_at', how='left')
    
    # Fill NaNs (e.g. if external data slightly shorter)
    df_monthly = df_monthly.ffill().bfill()
    # ---------------------------------
    
    return df_monthly

def create_monthly_features(df_monthly):
    print("Creating monthly features...")
    df_monthly['month'] = df_monthly['measured_at'].dt.month
    df_monthly['year'] = df_monthly['measured_at'].dt.year
    
    # Lags (12 months = 1 year)
    df_monthly = df_monthly.sort_values(['group_id', 'measured_at'])
    
    df_monthly['lag_12'] = df_monthly.groupby('group_id')['consumption'].shift(12)
    
    # Rolling mean of lag_12 (e.g. mean of same quarter last year)
    df_monthly['rolling_mean_12_3'] = df_monthly.groupby('group_id')['lag_12'].transform(lambda x: x.rolling(window=3).mean())
    
    # Drop NaNs
    # We need at least 12 months history + window 3 = 14 months?
    # shift(12) makes first 12 NaN.
    # rolling(3) on lag_12 makes first 12+2 = 14 NaN.
    df_model = df_monthly.dropna(subset=['lag_12', 'rolling_mean_12_3'])
    
    return df_model, df_monthly # Return full df_monthly for history reference

def generate_long_term_forecast(model, df_monthly, df_groups):
    print("Generating long-term forecast...")
    
    # Create future dataframe
    future_dates = pd.date_range(start='2024-10-01', end='2025-09-01', freq='MS', tz='UTC')
    group_ids = df_groups['group_id'].unique()
    
    # --- Fetch Future External Data ---
    print("Fetching future external data...")
    # We need data for the forecast period. 
    # get_external_data will fetch real data if available (e.g. up to now) or simulate/forecast if future.
    # Since we are forecasting 2024-10 to 2025-09, most of this is future relative to training data (2021-2023/24).
    # But wait, we are in Nov 2025 (simulated time)? No, user time is Nov 2025.
    # The challenge context is likely predicting for "future" from a past point, or we have actuals up to Sep 2024.
    # Let's assume we have access to "real" external data for the forecast period (perfect foresight or forecast).
    # Our fetch script fetched up to Oct 2025. So we have it.
    
    df_ext_future = get_external_data(future_dates.min(), future_dates.max() + pd.Timedelta(days=31))
    df_ext_future_monthly = df_ext_future.set_index('measured_at').resample('MS').mean().reset_index()
    # ----------------------------------
    
    # Optimized approach:
    # Since lag_12 and rolling_mean_12_3 depend only on history (for a 12-month forecast horizon),
    # we can pre-calculate features for all future dates at once.
    
    future_df_list = []
    for gid in group_ids:
        g_hist = df_monthly[df_monthly['group_id'] == gid].set_index('measured_at').sort_index()
        
        # Create future index
        g_future = pd.DataFrame({'measured_at': future_dates, 'group_id': gid})
        g_future = g_future.merge(df_ext_future_monthly, on='measured_at', how='left')
        
        # We need to append future rows to history to calculate lags using pandas shift/rolling
        # But we only need the features for the future rows.
        
        # Combine hist and future (empty consumption)
        g_combined = pd.concat([g_hist.reset_index(), g_future], ignore_index=True)
        g_combined['measured_at'] = pd.to_datetime(g_combined['measured_at'], utc=True)
        g_combined = g_combined.sort_values('measured_at').set_index('measured_at')
        
        # Calculate features
        g_combined['lag_12'] = g_combined['consumption'].shift(12)
        g_combined['rolling_mean_12_3'] = g_combined['lag_12'].rolling(window=3).mean()
        g_combined['month'] = g_combined.index.month
        
        # Filter for future dates
        # Ensure index name is set so reset_index creates 'measured_at' column
        g_combined.index.name = 'measured_at'
        
        # Use reindex to avoid KeyError if some dates are missing (though they shouldn't be)
        # and to ensure we get a DataFrame with the index we expect.
        g_future_features = g_combined.reindex(future_dates)
        g_future_features.index.name = 'measured_at' # Reindex might lose name
        g_future_features = g_future_features.reset_index()
        
        future_df_list.append(g_future_features)
        
    df_future_features = pd.concat(future_df_list, ignore_index=True)
    
    # Fill NaNs if any (e.g. first few lags if history is short)
    df_future_features = df_future_features.fillna(0) # Or better imputation
    
    # Debug: Check columns
    print(f"Forecast features columns: {df_future_features.columns.tolist()}")
    
    # Predict
    predictions = model.predict(df_future_features)
    df_future_features['consumption'] = predictions
    
    # Format output
    # Ensure measured_at is present
    if 'measured_at' not in df_future_features.columns:
        # Fallback if it's named 'index' or something else
        if 'index' in df_future_features.columns:
            df_future_features = df_future_features.rename(columns={'index': 'measured_at'})
            
    df_forecast = df_future_features[['measured_at', 'group_id', 'consumption']]
    
    return df_forecast

if __name__ == "__main__":
    from data_loader import load_data
    
    data = load_data()
    df_cons = data['training_consumption']
    df_groups = data['groups']
    
    # Prepare
    df_monthly_all = prepare_monthly_data(df_cons, df_groups)
    df_features, _ = create_monthly_features(df_monthly_all)
    
    # Train/Val Split
    # Train: < 2024, Val: 2024 (Jan-Sep)
    # Or just last 12 months as val
    val_start = '2023-10-01'
    train_mask = df_features['measured_at'] < val_start
    val_mask = df_features['measured_at'] >= val_start
    
    df_train = df_features[train_mask]
    df_val = df_features[val_mask]
    
    print(f"Train size: {len(df_train)}, Val size: {len(df_val)}")
    
    model = LongTermModel()
    model.train(df_train, df_val)
    
    # Save model
    import os
    os.makedirs("models", exist_ok=True)
    model.save("models/long_term_model.json")
    
    # Evaluate
    preds = model.predict(df_val)
    mape = mean_absolute_percentage_error(df_val['consumption'], preds)
    print(f"Validation MAPE: {mape:.4f}")
    
    # Save Validation Predictions
    df_val_save = df_val.copy()
    df_val_save['prediction'] = preds
    import os
    os.makedirs("predictions/monthly", exist_ok=True)
    df_val_save.to_csv("predictions/monthly/validation.csv")
    print("Saved validation predictions to predictions/monthly/validation.csv")
    
    # Forecast
    forecast_df = generate_long_term_forecast(model, df_monthly_all, df_groups)
    print(forecast_df.head())
    
    import os
    os.makedirs("predictions/monthly", exist_ok=True)
    forecast_df.to_csv("predictions/monthly/raw_forecast.csv")
