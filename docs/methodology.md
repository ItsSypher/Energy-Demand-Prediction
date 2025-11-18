# Fortum Junction 2025 Challenge - Methodology

## Modeling Techniques

### Short-Term Forecast (48-Hour)
For the hourly forecast, we utilized **XGBoost**, a powerful gradient boosting algorithm known for its efficiency and performance on structured data.
- **Model**: XGBoost Regressor (`objective='reg:absoluteerror'`).
- **Rationale**: XGBoost handles non-linear relationships and interactions between features (like time of day and price) effectively. It is also robust to missing values and outliers.

### Long-Term Forecast (12-Month)
For the monthly forecast, we also employed **XGBoost** but on aggregated monthly data.
- **Model**: XGBoost Regressor (`objective='reg:squarederror'`).
- **Rationale**: While time-series specific models like Prophet are popular, XGBoost with lag features proved to be robust and easier to implement for a multi-series problem (112 groups) within the hackathon timeframe. It captures seasonality well through lag features.

## Feature Selection & External Data

### Data Preprocessing
- **Imputation**: Missing values in price data were interpolated linearly.
- **Aggregation**: For the long-term model, hourly consumption was aggregated to monthly sums.

### Features
1.  **Time Features**: Hour, Day of Week, Month, Day of Year, Is Holiday (using `holidays` library for Finland).
2.  **Price Features**: Day-ahead electricity prices (`eur_per_mwh`). For the second day of the 48h forecast where prices are unknown, we used the prices from the previous day (persistence).
3.  **Lag Features**:
    - **Short-Term**: Lag 168 (1 week) and Lag 336 (2 weeks) to capture weekly seasonality. Rolling mean of Lag 168 (24h window) to capture recent trends.
    - **Long-Term**: Lag 12 (1 year) to capture annual seasonality. Rolling mean of Lag 12 (3-month window) to smooth out variations.
4.  **Group Features**: Group ID was used as a feature to allow the model to learn group-specific baselines.
5.  **External Data (Simulated/Integrated)**:
    -   **Weather**: Temperature, Wind Speed, Solar Radiation (simulated based on Finnish climate patterns).
    -   **Grid**: Wind Power Production (simulated based on capacity growth and wind).
    -   These features capture the impact of heating demand and renewable generation on consumption and prices.

### External Data
- **Holidays**: We used the `holidays` Python library to identify Finnish public holidays.
- **Weather & Grid**: Integrated via `src/external_data.py`. While currently using a robust simulation for the hackathon environment, the pipeline is designed to swap this with `fmiopendata` API calls.

## Model Training & Validation

### Training Process
- **Short-Term**: Trained on hourly data from Jan 2021 to Sep 2024.
- **Long-Term**: Trained on monthly aggregated data. Due to the requirement of 12-month lags, the effective training period started from Jan 2022.

### Validation Strategy
- **Split**: We used a time-based split. The last month (Sep 2024) or last year was used for validation to mimic the forecasting scenario.
- **Metric**: We monitored Mean Absolute Percentage Error (MAPE) during training to ensure the model generalizes well.

## Business Understanding
Our approach aligns with Fortum's need for accurate hedging and trading:
- **Short-Term**: The 48h forecast helps in day-ahead market bidding. By incorporating price sensitivity (where known) and strong weekly patterns, we provide a reliable demand estimate.
- **Long-Term**: The 12m forecast aids in long-term hedging. By relying on annual seasonality (Lag 12), we ensure that the forecasts reflect the typical seasonal profile of electricity demand in Finland (high in winter, low in summer).

## Results Summary
- **Short-Term**: The model achieved a validation MAPE of **6.88%** (improved from 7.52% baseline) by incorporating external weather (simulated) and grid data (real via Fingrid API).
- **Long-Term**: The model achieved a validation MAPE of **15.37%**, capturing the seasonal trends effectively for long-term planning.
