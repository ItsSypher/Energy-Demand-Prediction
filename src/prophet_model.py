import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from external_data import get_external_data
from data_loader import load_data
import matplotlib.pyplot as plt
import os

def prepare_prophet_data(df_cons, df_groups):
    print("Preparing data for Prophet...")
    
    # Melt to long format
    df_long = df_cons.melt(id_vars=['measured_at'], var_name='group_id', value_name='consumption')
    df_long['group_id'] = df_long['group_id'].astype(int)
    df_long['measured_at'] = pd.to_datetime(df_long['measured_at'], utc=True)
    
    # Resample to monthly sum (Prophet works best with daily/sub-daily, but monthly is fine)
    # We need to set 'ds' to the start of the month
    df_monthly = df_long.set_index('measured_at').groupby('group_id').resample('MS')['consumption'].sum().reset_index()
    
    # Rename for Prophet
    df_monthly = df_monthly.rename(columns={'measured_at': 'ds', 'consumption': 'y'})
    
    # Remove timezone from ds (Prophet prefers naive or consistent tz, usually naive is safer if we strip it)
    df_monthly['ds'] = df_monthly['ds'].dt.tz_localize(None)
    
    return df_monthly

def get_external_regressors(start_date, end_date):
    print("Fetching external regressors...")
    # Fetch hourly
    df_ext = get_external_data(start_date, end_date)
    df_ext['measured_at'] = pd.to_datetime(df_ext['measured_at'], utc=True)
    
    # Resample to monthly mean
    df_ext_monthly = df_ext.set_index('measured_at').resample('MS').mean().reset_index()
    
    # Rename
    df_ext_monthly = df_ext_monthly.rename(columns={'measured_at': 'ds'})
    df_ext_monthly['ds'] = df_ext_monthly['ds'].dt.tz_localize(None)
    
    return df_ext_monthly

    # Combine all forecasts
    df_final = pd.concat(all_forecasts, ignore_index=True)
    
    # Filter for forecast period (Oct 2024 - Sep 2025)
    forecast_start = pd.Timestamp('2024-10-01')
    df_submission = df_final[df_final['ds'] >= forecast_start].copy()
    
    # Rename for consistency
    df_submission = df_submission.rename(columns={'ds': 'measured_at', 'yhat': 'consumption'})
    
    # Save raw forecast
    os.makedirs('predictions/monthly', exist_ok=True)
    df_submission.to_csv('predictions/monthly/prophet_raw.csv', index=False)
    print("Saved Prophet predictions to predictions/monthly/prophet_raw.csv")
    
    return df_submission

def train_predict_prophet():
    from prophet.serialize import model_to_json
    
    # Load data
    data = load_data('Dataset/20251111_JUNCTION_training.xlsx')
    df_groups = data['groups']
    df_cons = data['training_consumption']
    
    df_all = prepare_prophet_data(df_cons, df_groups)
    
    # Get external data range
    start_date = df_all['ds'].min().tz_localize('UTC')
    end_date = df_all['ds'].max().tz_localize('UTC') + pd.Timedelta(days=400)
    
    df_ext = get_external_regressors(start_date, end_date)
    
    group_ids = df_all['group_id'].unique()
    all_forecasts = []
    
    os.makedirs('models/prophet', exist_ok=True)
    
    print(f"Training Prophet models for {len(group_ids)} groups...")
    
    for i, gid in enumerate(group_ids):
        try:
            # Filter group data
            df_group = df_all[df_all['group_id'] == gid].copy()
            
            # Merge external regressors
            df_group = df_group.merge(df_ext, on='ds', how='left')
            # Fix deprecated fillna
            df_group = df_group.ffill().bfill()
            
            # Initialize Prophet
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            
            # Add regressors only if they are not constant
            for reg in ['temperature', 'wind_speed', 'wind_power_production']:
                if df_group[reg].nunique() > 1:
                    model.add_regressor(reg)
                else:
                    print(f"Skipping constant regressor {reg} for group {gid}")
            
            # Fit
            model.fit(df_group)
            
            # Save model
            with open(f'models/prophet/group_{gid}.json', 'w') as fout:
                fout.write(model_to_json(model))
            
            # Make future dataframe
            future = model.make_future_dataframe(periods=12, freq='MS')
            
            # Add external regressors to future
            future = future.merge(df_ext, on='ds', how='left')
            future = future.ffill().bfill()
            
            # Predict
            forecast = model.predict(future)
            
            # Extract relevant columns
            forecast['group_id'] = gid
            forecast_cols = ['ds', 'group_id', 'yhat', 'yhat_lower', 'yhat_upper']
            all_forecasts.append(forecast[forecast_cols])
            
        except Exception as e:
            print(f"Error training group {gid}: {e}")
            continue
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(group_ids)} groups")
            
    # Combine all forecasts
    if not all_forecasts:
        print("No forecasts generated!")
        return None
        
    df_final = pd.concat(all_forecasts, ignore_index=True)
    
    # Filter for forecast period (Oct 2024 - Sep 2025)
    forecast_start = pd.Timestamp('2024-10-01')
    df_submission = df_final[df_final['ds'] >= forecast_start].copy()
    
    # Rename for consistency
    df_submission = df_submission.rename(columns={'ds': 'measured_at', 'yhat': 'consumption'})
    
    # Save raw forecast
    os.makedirs('predictions/monthly', exist_ok=True)
    df_submission.to_csv('predictions/monthly/prophet_raw.csv', index=False)
    print("Saved Prophet predictions to predictions/monthly/prophet_raw.csv")
    
    return df_submission

def validate_prophet():
    print("\n--- Validating Prophet Model ---")
    # Load data
    data = load_data('Dataset/20251111_JUNCTION_training.xlsx')
    df_groups = data['groups']
    df_cons = data['training_consumption']
    
    df_all = prepare_prophet_data(df_cons, df_groups)
    
    # Split date (Last 12 months for validation)
    dates = df_all['ds'].unique()
    dates = np.sort(dates)
    split_date = dates[-12]
    print(f"Validation Split Date: {split_date}")
    
    # External data for validation period
    start_date = df_all['ds'].min().tz_localize('UTC')
    end_date = df_all['ds'].max().tz_localize('UTC')
    df_ext = get_external_regressors(start_date, end_date)
    
    group_ids = df_all['group_id'].unique()
    val_preds = []
    
    for i, gid in enumerate(group_ids):
        df_group = df_all[df_all['group_id'] == gid].copy()
        df_group = df_group.merge(df_ext, on='ds', how='left')
        df_group = df_group.fillna(method='ffill').fillna(method='bfill')
        
        # Split
        train = df_group[df_group['ds'] < split_date]
        val = df_group[df_group['ds'] >= split_date]
        
        if train.empty or val.empty:
            continue
            
        # Train
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.add_regressor('temperature')
        model.add_regressor('wind_speed')
        model.add_regressor('wind_power_production')
        
        # Suppress output
        import logging
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
        
        model.fit(train)
        
        # Predict on Val
        forecast = model.predict(val)
        
        # Store
        val_res = val[['ds', 'group_id', 'y']].copy()
        val_res['prediction'] = forecast['yhat'].values
        val_preds.append(val_res)
        
        if (i+1) % 20 == 0:
            print(f"Validated {i+1}/{len(group_ids)} groups")
            
    df_val_all = pd.concat(val_preds, ignore_index=True)
    df_val_all = df_val_all.rename(columns={'ds': 'measured_at', 'y': 'consumption'})
    
    os.makedirs('predictions/monthly', exist_ok=True)
    df_val_all.to_csv('predictions/monthly/prophet_validation.csv', index=False)
    print("Saved Prophet validation predictions to predictions/monthly/prophet_validation.csv")
    
    # Quick metric check
    mape = mean_absolute_percentage_error(df_val_all['consumption'], df_val_all['prediction'])
    print(f"Prophet Validation MAPE: {mape:.4f}")

if __name__ == "__main__":
    train_predict_prophet()
    validate_prophet()
