import pandas as pd
import numpy as np
import holidays

def preprocess_data(data):
    """
    Preprocesses the raw data dictionaries.
    - Converts timestamps to datetime.
    - Handles missing values in prices.
    - Merges consumption and prices.
    """
    print("Preprocessing data...")
    
    # 1. Process Prices
    df_prices = data['training_prices'].copy()
    df_prices['measured_at'] = pd.to_datetime(df_prices['measured_at'])
    df_prices = df_prices.sort_values('measured_at').set_index('measured_at')
    
    # Handle missing prices (interpolate)
    if df_prices['eur_per_mwh'].isnull().sum() > 0:
        print(f"Imputing {df_prices['eur_per_mwh'].isnull().sum()} missing price values...")
        df_prices['eur_per_mwh'] = df_prices['eur_per_mwh'].interpolate(method='linear')
        
    # 2. Process Consumption
    df_cons = data['training_consumption'].copy()
    df_cons['measured_at'] = pd.to_datetime(df_cons['measured_at'])
    df_cons = df_cons.sort_values('measured_at').set_index('measured_at')
    
    # 3. Process Groups (if needed for static features)
    df_groups = data['groups'].copy()
    
    return df_cons, df_prices, df_groups

def create_features(df_cons, df_prices, df_groups):
    """
    Creates features for the short-term model.
    Returns a DataFrame suitable for training.
    """
    print("Creating features...")
    
    # We need to transform the wide consumption format (one column per group) 
    # into a long format (timestamp, group_id, consumption) for training a single model 
    # (or we can train one model per group, but a global model with group features might be better/faster).
    # Given 112 groups, training 112 models is feasible but maybe less robust if data is noisy.
    # Let's try a global model approach first.
    
    # Melt consumption to long format
    df_long = df_cons.reset_index().melt(id_vars=['measured_at'], var_name='group_id', value_name='consumption')
    df_long['group_id'] = df_long['group_id'].astype(int) # Ensure group_id is int
    
    # Merge prices
    df_long = df_long.merge(df_prices, on='measured_at', how='left')
    
    # Merge group static features
    # df_groups has columns like 'id', 'Macro Region', etc.
    # We need to encode these categorical features.
    # For now, let's just merge them.
    df_groups = df_groups.rename(columns={'id': 'group_id'})
    df_long = df_long.merge(df_groups, on='group_id', how='left')
    
    # Time features
    df_long['hour'] = df_long['measured_at'].dt.hour
    df_long['dayofweek'] = df_long['measured_at'].dt.dayofweek
    df_long['month'] = df_long['measured_at'].dt.month
    df_long['dayofyear'] = df_long['measured_at'].dt.dayofyear
    
    # Holidays (Finland)
    fi_holidays = holidays.Finland()
    df_long['is_holiday'] = df_long['measured_at'].apply(lambda x: x in fi_holidays).astype(int)
    
    # Lag features
    # Since we are predicting 48h ahead, we can't use immediate lags (t-1) for the whole horizon 
    # without recursive forecasting.
    # However, for training, we can create them.
    # A robust strategy for 48h forecast without recursion is to use lags >= 48h.
    # Or use lags like 168h (1 week) which is always available.
    
    # Add holidays
    fi_holidays = holidays.Finland()
    df_long['is_holiday'] = df_long['measured_at'].dt.date.apply(lambda x: x in fi_holidays).astype(int)
    
    # --- External Data Integration ---
    from external_data import get_external_data
    print("Fetching/Generating external data...")
    start_date = df_long['measured_at'].min()
    end_date = df_long['measured_at'].max()
    
    # Ensure we cover the full range including potential gaps if any, or just use min/max
    df_ext = get_external_data(start_date, end_date)
    
    # Merge
    df_long = df_long.merge(df_ext, on='measured_at', how='left')
    
    # Fill NaNs if any (e.g. if external data generation had gaps, though it shouldn't)
    df_long['temperature'] = df_long['temperature'].ffill().bfill()
    df_long['wind_speed'] = df_long['wind_speed'].ffill().bfill()
    df_long['solar_radiation'] = df_long['solar_radiation'].fillna(0)
    df_long['wind_power_production'] = df_long['wind_power_production'].ffill()
    # ---------------------------------

    # Create lag features (only for hourly data)
    # We need to be careful with group_id.
    # Sort by group and time
    df_long = df_long.sort_values(['group_id', 'measured_at'])
    
    print("Creating lag features...")
    # Lag 1 week (168 hours)
    df_long['lag_168'] = df_long.groupby('group_id')['consumption'].shift(168)
    
    # Lag 2 weeks (336 hours)
    df_long['lag_336'] = df_long.groupby('group_id')['consumption'].shift(336)
    
    # Rolling mean of lag_168 (e.g. average of same time over last few days? No, rolling over time)
    # Rolling mean of last 24h (shifted by 168 to be available)
    # We want the trend from last week.
    df_long['rolling_mean_168_24'] = df_long.groupby('group_id')['lag_168'].transform(lambda x: x.rolling(window=24).mean())
    
    # Drop rows with NaNs created by lags
    df_long = df_long.dropna(subset=['lag_168'])
    
    return df_long

if __name__ == "__main__":
    from data_loader import load_data
    data = load_data()
    df_cons, df_prices, df_groups = preprocess_data(data)
    df_features = create_features(df_cons, df_prices, df_groups)
    print(df_features.head())
    print(df_features.info())
