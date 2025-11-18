# Fortum Junction 2025 Challenge Solution

This repository contains the solution for the Fortum forecasting challenge at Junction 2025.

## Project Structure
- `src/`: Source code for data loading, feature engineering, modeling, and submission.
    - `data_loader.py`: Loads data from Excel.
    - `feature_engineering.py`: Creates features for short-term model.
    - `models.py`: Short-term model (XGBoost) and forecasting logic.
    - `long_term_model.py`: Long-term model (XGBoost) and forecasting logic.
    - `submission.py`: Formats the output CSVs.
- `Dataset/`: Contains the input data (not included in repo if private, but expected structure).
- `predictions/`: Generated forecast files.
    - `hourly/`:
        - `raw_forecast.csv`: Raw model output.
        - `submission.csv`: Formatted submission file.
    - `monthly/`:
        - `raw_forecast.csv`: Raw model output.
        - `submission.csv`: Formatted submission file.
- `docs/`: Documentation.
    - `methodology.md`: Detailed explanation of the approach.
- `requirements.txt`: Python dependencies.

## How to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the short-term model:
    ```bash
    python src/models.py
    ```
3.  Run the long-term model:
    ```bash
    python src/long_term_model.py
    ```
4.  Generate submission files:
    ```bash
    python src/submission.py
    ```
    This will create `predictions/hourly/submission.csv` and `predictions/monthly/submission.csv`.

## Approach
We used XGBoost for both short-term and long-term forecasting, leveraging lag features to capture weekly and annual seasonality. See `docs/methodology.md` for details.
