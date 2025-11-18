import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

def calculate_mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

def evaluate_short_term():
    print("\n--- Short-Term Evaluation (Hourly) ---")
    try:
        # Load validation predictions (saved by models.py)
        df_val = pd.read_csv('predictions/hourly/validation.csv')
        df_val['measured_at'] = pd.to_datetime(df_val['measured_at'], utc=True)
        
        # Calculate Model MAPE
        mape_model = calculate_mape(df_val['consumption'], df_val['prediction'])
        print(f"Model MAPE: {mape_model:.4f} ({mape_model*100:.2f}%)")
        
        # Calculate Naive Baseline (Persistence: 168h lag - same hour last week)
        # We need to reconstruct the lag from the dataset or assume we can get it.
        # Since validation.csv might not have the lag column, we might need to load original data.
        # However, let's check if we can load the full dataset to get the lag.
        
        from data_loader import load_data
        print("Loading full data for baseline calculation...")
        data = load_data('Dataset/20251111_JUNCTION_training.xlsx')
        df_groups = data['groups']
        df_cons = data['training_consumption']
        
        # Melt and process
        df_long = df_cons.melt(id_vars=['measured_at'], var_name='group_id', value_name='consumption')
        df_long['measured_at'] = pd.to_datetime(df_long['measured_at'], utc=True)
        df_long['group_id'] = df_long['group_id'].astype(int)
        df_long = df_long.sort_values(['group_id', 'measured_at'])
        
        # Create Naive Prediction (Lag 168)
        df_long['naive_168'] = df_long.groupby('group_id')['consumption'].shift(168)
        
        # Merge naive prediction into validation set
        df_val_naive = df_val.merge(df_long[['measured_at', 'group_id', 'naive_168']], on=['measured_at', 'group_id'], how='left')
        
        # Drop NaNs (start of series)
        df_val_naive = df_val_naive.dropna(subset=['naive_168'])
        
        mape_naive = calculate_mape(df_val_naive['consumption'], df_val_naive['naive_168'])
        print(f"Naive Baseline (168h Persistence) MAPE: {mape_naive:.4f} ({mape_naive*100:.2f}%)")
        
        # Calculate FVA
        fva = (mape_naive - mape_model) / mape_naive
        print(f"Forecast Value Added (FVA): {fva:.4f} ({fva*100:.2f}%)")
        
        if fva > 0:
            print("SUCCESS: Model outperforms baseline.")
        else:
            print("WARNING: Model underperforms baseline.")
            
    except FileNotFoundError:
        print("Validation predictions not found. Run models.py first.")
    except Exception as e:
        print(f"Error in short-term evaluation: {e}")

def evaluate_long_term():
    print("\n--- Long-Term Evaluation (Monthly) ---")
    # We need to generate validation predictions for long-term model first if not saved.
    # Or we can simulate it here.
    # Let's assume we want to evaluate on the last 12 months of training data (holdout).
    
    from long_term_model import prepare_monthly_data, create_monthly_features, LongTermModel
    from data_loader import load_data
    
    print("Loading data...")
    data = load_data('Dataset/20251111_JUNCTION_training.xlsx')
    df_groups = data['groups']
    df_cons = data['training_consumption']
    df_monthly = prepare_monthly_data(df_cons, df_groups)
    df_features, _ = create_monthly_features(df_monthly)
    
    # Split Train/Val (Last 12 months as validation)
    # Dates are monthly start.
    dates = df_features['measured_at'].unique()
    dates = np.sort(dates)
    split_date = dates[-12] # Last 12 months
    
    print(f"Splitting data at {split_date}")
    
    train_mask = df_features['measured_at'] < split_date
    val_mask = df_features['measured_at'] >= split_date
    
    df_train = df_features[train_mask]
    df_val = df_features[val_mask]
    
    # Train Model
    model = LongTermModel()
    model.train(df_train, df_val)
    
    # Predict
    preds = model.predict(df_val)
    
    # Calculate Model MAPE
    mape_model = calculate_mape(df_val['consumption'], preds)
    print(f"Model MAPE: {mape_model:.4f} ({mape_model*100:.2f}%)")
    
    # Naive Baseline (Seasonal Persistence: Lag 12)
    # We already have lag_12 feature!
    mape_naive = calculate_mape(df_val['consumption'], df_val['lag_12'])
    print(f"Naive Baseline (12m Persistence) MAPE: {mape_naive:.4f} ({mape_naive*100:.2f}%)")
    
    # Calculate FVA
    fva = (mape_naive - mape_model) / mape_naive
    print(f"Forecast Value Added (FVA): {fva:.4f} ({fva*100:.2f}%)")
    
    if fva > 0:
        print("SUCCESS: Model outperforms baseline.")
    else:
        print("WARNING: Model underperforms baseline.")

if __name__ == "__main__":
    evaluate_short_term()
    evaluate_long_term()
