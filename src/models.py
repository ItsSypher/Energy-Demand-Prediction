import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
import holidays

class ShortTermModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:absoluteerror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            n_jobs=-1,
            random_state=42
        )
        self.features = [
            'hour', 'dayofweek', 'month', 'dayofyear', 'is_holiday',
            'eur_per_mwh',
            'lag_168', 'lag_336', 'rolling_mean_168_24',
            'group_id',
            # New External Features
            'temperature', 'wind_speed', 'solar_radiation', 'wind_power_production'
        ]

    def train(self, df_train, df_val=None):
        """
        Trains the XGBoost model.
        """
        print("Training ShortTermModel...")
        # Remove duplicates from features just in case
        self.features = list(dict.fromkeys(self.features))
        
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
        """
        Generates predictions.
        """
        X = df_features[self.features]
        return self.model.predict(X)

def train_and_evaluate_short_term(df_features):
    """
    Splits data, trains model, and prints metrics.
    """
    # Split data: Train < Sept 2024, Val = Sept 2024
    # Training data ends at 2024-09-30 23:00:00
    
    val_start_date = '2024-09-01'
    
    train_mask = df_features['measured_at'] < val_start_date
    val_mask = df_features['measured_at'] >= val_start_date
    
    df_train = df_features[train_mask]
    df_val = df_features[val_mask]
    
    print(f"Train set size: {len(df_train)}")
    print(f"Val set size: {len(df_val)}")
    
    model = ShortTermModel()
    model.train(df_train, df_val)
    
    # Save model
    import os
    os.makedirs("models", exist_ok=True)
    model.save("models/short_term_model.json")
    
    # Evaluate
    preds = model.predict(df_val)
    mape = mean_absolute_percentage_error(df_val['consumption'], preds)
    print(f"Validation MAPE: {mape:.4f}")
    
    # Save Validation Predictions
    df_val_save = df_val.copy()
    df_val_save['prediction'] = preds
    import os
    os.makedirs("predictions/hourly", exist_ok=True)
    df_val_save.to_csv("predictions/hourly/validation.csv")
    print("Saved validation predictions to predictions/hourly/validation.csv")
    
    return model

def generate_short_term_forecast(model, df_cons, df_prices, df_groups):
    """
    Generates the 48h forecast for Oct 1-2, 2024.
    """
    print("Generating short-term forecast...")
    
    # 1. Create future timestamps
    future_dates = pd.date_range(start='2024-10-01 00:00:00+00:00', end='2024-10-02 23:00:00+00:00', freq='h')
    
    # 2. Create future dataframe structure
    # We need a row for each group for each future timestamp
    group_ids = df_groups['group_id'].unique()
    
    future_rows = []
    for ts in future_dates:
        for gid in group_ids:
            future_rows.append({'measured_at': ts, 'group_id': gid})
            
    df_future = pd.DataFrame(future_rows)
    
    # 3. Append to historical data to calculate lags
    # We need enough history to calculate lag_168 and lag_336
    # Last timestamp in training is 2024-09-30 23:00
    # We need at least 336 hours of history
    
    # Prepare historical data (long format)
    df_hist_long = df_cons.reset_index().melt(id_vars=['measured_at'], var_name='group_id', value_name='consumption')
    df_hist_long['group_id'] = df_hist_long['group_id'].astype(int)
    
    # Filter to last 3 weeks to save memory, but enough for lags
    start_hist = pd.Timestamp('2024-09-01').tz_localize('UTC')
    df_hist_long = df_hist_long[df_hist_long['measured_at'] >= start_hist]
    
    # Concatenate
    df_combined = pd.concat([df_hist_long, df_future], ignore_index=True)
    
    # 4. Merge Prices
    df_combined = df_combined.merge(df_prices, on='measured_at', how='left')
    
    # Impute missing prices for Oct 2
    mask_missing_price = df_combined['eur_per_mwh'].isnull()
    if mask_missing_price.sum() > 0:
        print(f"Imputing {mask_missing_price.sum()} missing prices for forecast period...")
        # Get unique timestamps for Oct 2
        oct2_mask = (df_combined['measured_at'] >= '2024-10-02 00:00:00+00:00') & (df_combined['measured_at'] <= '2024-10-02 23:00:00+00:00')
        oct2_hours = df_combined.loc[oct2_mask, 'measured_at'].unique()
        for ts in oct2_hours:
            ts_prev = pd.Timestamp(ts) - pd.Timedelta(hours=24)
            try:
                price = df_prices.loc[ts_prev, 'eur_per_mwh']
                df_combined.loc[df_combined['measured_at'] == ts, 'eur_per_mwh'] = price
            except KeyError:
                pass

    # 5. Merge Group Features
    df_combined = df_combined.merge(df_groups, on='group_id', how='left')
    
    # 6. Feature Engineering (Time + Lags + External)
    df_combined['hour'] = df_combined['measured_at'].dt.hour
    df_combined['dayofweek'] = df_combined['measured_at'].dt.dayofweek
    df_combined['month'] = df_combined['measured_at'].dt.month
    df_combined['dayofyear'] = df_combined['measured_at'].dt.dayofyear
    
    fi_holidays = holidays.Finland()
    df_combined['is_holiday'] = df_combined['measured_at'].apply(lambda x: x in fi_holidays).astype(int)
    
    # --- External Data Integration ---
    from external_data import get_external_data
    print("Generating external data for forecast...")
    start_date = df_combined['measured_at'].min()
    end_date = df_combined['measured_at'].max()
    df_ext = get_external_data(start_date, end_date)
    df_combined = df_combined.merge(df_ext, on='measured_at', how='left')
    
    # Fill NaNs
    df_combined['temperature'] = df_combined['temperature'].ffill().bfill()
    df_combined['wind_speed'] = df_combined['wind_speed'].ffill().bfill()
    df_combined['solar_radiation'] = df_combined['solar_radiation'].fillna(0)
    df_combined['wind_power_production'] = df_combined['wind_power_production'].ffill()
    # ---------------------------------
    
    # Lags
    print("Calculating lags for forecast...")
    df_combined = df_combined.sort_values(['group_id', 'measured_at'])
    df_combined['lag_168'] = df_combined.groupby('group_id')['consumption'].shift(168)
    df_combined['lag_336'] = df_combined.groupby('group_id')['consumption'].shift(336)
    df_combined['rolling_mean_168_24'] = df_combined.groupby('group_id')['lag_168'].transform(lambda x: x.rolling(window=24).mean())
    
    # 7. Filter to Forecast Period
    df_forecast = df_combined[df_combined['measured_at'] >= '2024-10-01 00:00:00+00:00'].copy()
    
    # 8. Predict
    print("Predicting...")
    preds = model.predict(df_forecast)
    df_forecast['prediction'] = preds
    
    # 9. Format for Output (Pivot)
    df_pivot = df_forecast.pivot(index='measured_at', columns='group_id', values='prediction')
    
    return df_pivot

if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import preprocess_data, create_features
    
    data = load_data()
    df_cons, df_prices, df_groups = preprocess_data(data)
    df_features = create_features(df_cons, df_prices, df_groups)
    
    model = train_and_evaluate_short_term(df_features)
    
    # Generate Forecast
    forecast_df = generate_short_term_forecast(model, df_cons, df_prices, df_groups)
    print(forecast_df.head())
    
    # Save to CSV (temporary)
    import os
    os.makedirs("predictions/hourly", exist_ok=True)
    forecast_df.to_csv("predictions/hourly/raw_forecast.csv")
