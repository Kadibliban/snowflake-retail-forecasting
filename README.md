# Snowflake Retail Sales Forecasting

An end-to-end retail forecasting portfolio project combining Snowflake SQL, reproducible synthetic data, scikit-learn model evaluation, recursive forecasting, and an interactive Streamlit dashboard.

> **Demo-data notice:** The default dataset is synthetic. It demonstrates the engineering workflow, not the performance or sales patterns of a real retailer.

## What this project demonstrates

- Generates 730 reproducible daily sales observations with trend, weekly and annual seasonality, and noise.
- Validates date continuity and data quality instead of silently filling gaps.
- Builds leakage-aware lag, rolling-average, and calendar features.
- Uses a chronological 80/20 holdout and compares a Random Forest with a naive 7-day seasonal baseline.
- Reports MAE, RMSE, and WAPE when its denominator is valid.
- Produces recursive forecasts for up to 14 days and persists model/evaluation artifacts locally.
- Supports a local synthetic workflow without a Snowflake account, with optional Snowflake reads and loads.
- Provides dashboard controls to focus the historical chart, inspect a selected holdout window, and choose the forecast horizon.

## Dashboard

The dashboard is organized into four focused views:

1. **Sales history** — choose how many of the latest days to display.
2. **Model evaluation** — compare actuals, model predictions, and the naive baseline; inspect and score a selected holdout window.
3. **Forecast** — choose 1–14 forecast days and review the chart, total, daily average, and table.
4. **Data quality** — review date gaps, duplicate dates, and model run details.

The sidebar lets you select synthetic CSV or Snowflake data (Snowflake appears when all required connection settings are present) and refresh cached data.

## Quick start (Windows PowerShell)

Use Python 3.12 for the Snowflake connector's broadest Windows compatibility.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.generate_data
python -m src.train
streamlit run app.py
```

Open the local Streamlit URL printed in the terminal. Run the tests with:

```powershell
python -m pytest -q
```

Generated local data and model outputs are ignored by Git; the commands above recreate them.

## Workflow and architecture

```text
Synthetic CSV ─┐
               ├─> validation ─> feature engineering ─> chronological split
Snowflake ─────┘                                          │
                                                         ├─> Random Forest vs. naive baseline
                                                         ├─> holdout predictions + metrics
                                                         └─> recursive 14-day forecast
                                                                  │
                                                                  └─> Streamlit dashboard
```

| Area | Implementation |
| --- | --- |
| Warehouse | Snowflake database `RETAIL_ML`, schema `ANALYTICS`, table `SALES_RAW` |
| Data | `src/generate_data.py`, `src/load_data.py` |
| Features and validation | `src/features.py` |
| Training and evaluation | `src/train.py` |
| Forecasting | `src/predict.py` |
| Dashboard | `app.py` |
| SQL examples | `sql/setup.sql`, `sql/sales_analysis.sql` |
| Tests | `tests/` |

## Snowflake setup (optional)

1. Copy `.env.example` to `.env` and add your Snowflake account identifier, username, password, warehouse, database, and schema. Never commit `.env`.
2. Review `sql/setup.sql`. Object creation may require an administrator or another appropriately privileged role; the script includes role privilege examples. Do not assume your application role can create a database or warehouse.
3. Run the reviewed SQL in a Snowflake worksheet. The warehouse is XSMALL with auto-suspend after 60 seconds, auto-resume enabled, and initially suspended.
4. Generate the demo CSV and explicitly load it when ready:

   ```powershell
   python -m src.generate_data
   python -m src.load_data
   ```

   The loader does not replace existing table rows by default. To explicitly replace table contents, pass `--replace`.
5. Set the dashboard's data source to **Snowflake**. To train and forecast from Snowflake, run:

   ```powershell
   python -m src.train --data-source snowflake
   ```

The application uses the configured warehouse/database/schema for connections and reads from the explicitly qualified `RETAIL_ML.ANALYTICS.SALES_RAW` table. Snowflake access requires suitable warehouse usage, database/schema usage, and table `SELECT` privileges; uploading additionally requires `INSERT`. See the SQL setup script for example grants.

## Evaluation notes

The test set is the latest 20% of the chronological observations and is not used to fit the model. The dashboard also reports the naive baseline because a more complex model is not automatically a better model. Recursive forecasts feed earlier predictions into later steps, so errors may accumulate. Walk-forward validation and validation against real retailer data are future improvements; forecasts should not be used for business decisions without that validation.

## Repository hygiene

`.gitignore` excludes credentials, virtual environments, Python caches, and generated CSV/model/output files. Keep `.env` private, and only add screenshots that do not reveal account identifiers or other sensitive information.
