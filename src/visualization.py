import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_forecasts():
    print("Generating plots...")
    os.makedirs("plots", exist_ok=True)
    
    # Set style
    sns.set_theme(style="darkgrid")
    
    # 1. Load Forecasts
    try:
        df_hourly = pd.read_csv("predictions/hourly/raw_forecast.csv", parse_dates=['measured_at'])
        df_monthly = pd.read_csv("predictions/monthly/raw_forecast.csv", parse_dates=['measured_at'])
    except FileNotFoundError:
        print("Forecast files not found. Please run models first.")
        return

    # 2. Plot Hourly Forecast (Aggregate)
    plt.figure(figsize=(15, 6))
    hourly_agg = df_hourly.groupby('measured_at').sum(numeric_only=True).sum(axis=1)
    sns.lineplot(x=hourly_agg.index, y=hourly_agg.values)
    plt.title("Total Hourly Forecast (Aggregated across all groups)")
    plt.xlabel("Time")
    plt.ylabel("Total Consumption (FWH)")
    plt.tight_layout()
    plt.savefig("plots/hourly_forecast_aggregated.png")
    print("Saved plots/hourly_forecast_aggregated.png")
    
    # 3. Plot Monthly Forecast (Aggregate)
    plt.figure(figsize=(15, 6))
    monthly_agg = df_monthly.groupby('measured_at').sum(numeric_only=True).sum(axis=1)
    sns.lineplot(x=monthly_agg.index, y=monthly_agg.values, marker='o')
    plt.title("Total Monthly Forecast (Aggregated across all groups)")
    plt.xlabel("Month")
    plt.ylabel("Total Consumption (FWH)")
    plt.tight_layout()
    plt.savefig("plots/monthly_forecast_aggregated.png")
    print("Saved plots/monthly_forecast_aggregated.png")
    
    # 4. Plot Sample Groups (Hourly)
    # Pick 3 random groups
    sample_groups = df_hourly.columns[1:4] # Skip measured_at
    
    plt.figure(figsize=(15, 8))
    for gid in sample_groups:
        # Need to pivot back or just extract column
        # The raw forecast is pivoted (measured_at, group_1, group_2...)
        sns.lineplot(x=df_hourly['measured_at'], y=df_hourly[gid], label=f"Group {gid}")
        
    plt.title("Hourly Forecast for Sample Groups")
    plt.xlabel("Time")
    plt.ylabel("Consumption (FWH)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/hourly_forecast_samples.png")
    print("Saved plots/hourly_forecast_samples.png")

    # 5. Plot Sample Groups (Monthly)
    plt.figure(figsize=(15, 8))
    for gid in sample_groups:
        sns.lineplot(x=df_monthly['measured_at'], y=df_monthly[gid], label=f"Group {gid}", marker='o')
        
    plt.title("Monthly Forecast for Sample Groups")
    plt.xlabel("Month")
    plt.ylabel("Consumption (FWH)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/monthly_forecast_samples.png")
    print("Saved plots/monthly_forecast_samples.png")

def plot_validation_results():
    print("Generating validation plots...")
    
    # 1. Load Validation Data
    try:
        df_val_hourly = pd.read_csv("predictions/hourly/validation.csv", parse_dates=['measured_at'])
        df_val_monthly = pd.read_csv("predictions/monthly/validation.csv", parse_dates=['measured_at'])
    except FileNotFoundError:
        print("Validation files not found. Please run models first.")
        return

    # 2. Hourly Validation Plot (Aggregate)
    plt.figure(figsize=(15, 6))
    # Group by time to get total consumption
    val_hourly_agg = df_val_hourly.groupby('measured_at')[['consumption', 'prediction']].sum()
    
    sns.lineplot(data=val_hourly_agg, x=val_hourly_agg.index, y='consumption', label='Actual')
    sns.lineplot(data=val_hourly_agg, x=val_hourly_agg.index, y='prediction', label='Predicted', linestyle='--')
    
    plt.title("Short-Term Model Validation (Aggregated)")
    plt.xlabel("Time")
    plt.ylabel("Total Consumption (FWH)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/validation_hourly_aggregated.png")
    print("Saved plots/validation_hourly_aggregated.png")
    
    # 3. Monthly Validation Plot (Aggregate)
    plt.figure(figsize=(15, 6))
    val_monthly_agg = df_val_monthly.groupby('measured_at')[['consumption', 'prediction']].sum()
    
    sns.lineplot(data=val_monthly_agg, x=val_monthly_agg.index, y='consumption', label='Actual', marker='o')
    sns.lineplot(data=val_monthly_agg, x=val_monthly_agg.index, y='prediction', label='Predicted', marker='x', linestyle='--')
    
    plt.title("Long-Term Model Validation (Aggregated)")
    plt.xlabel("Month")
    plt.ylabel("Total Consumption (FWH)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/validation_monthly_aggregated.png")
    print("Saved plots/validation_monthly_aggregated.png")
    
    # 4. Sample Group Validation (Hourly)
    # Pick a group that exists in validation
    sample_gid = df_val_hourly['group_id'].unique()[0]
    sample_data = df_val_hourly[df_val_hourly['group_id'] == sample_gid]
    
    plt.figure(figsize=(15, 6))
    sns.lineplot(x=sample_data['measured_at'], y=sample_data['consumption'], label='Actual')
    sns.lineplot(x=sample_data['measured_at'], y=sample_data['prediction'], label='Predicted', linestyle='--')
    plt.title(f"Short-Term Validation for Group {sample_gid}")
    plt.xlabel("Time")
    plt.ylabel("Consumption (FWH)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/validation_hourly_sample.png")
    print("Saved plots/validation_hourly_sample.png")

if __name__ == "__main__":
    plot_forecasts()
    plot_validation_results()
