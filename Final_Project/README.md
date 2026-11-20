link:     https://india-air-quality-rspm.streamlit.app/

# 🌫️ India Air Quality — RSPM (PM10) Prediction

Regression project that predicts **RSPM** (Respirable Suspended Particulate Matter, µg/m³) from location, area type, date and other pollutants (SO₂, NO₂), using the CPCB *India Air Quality* dataset (1987–2015).

**Live app:** _[add Streamlit link]_

## Structure
```
Final_Project/
├── dataset/data.csv
├── notebooks/india_air_quality.ipynb
├── models/rspm_model.pkl      # preprocessing pipeline + tuned XGBoost
├── app.py                     # Streamlit app
├── app_metadata.json          # states / cities / station info for the app
├── requirements.txt
└── README.md
```

## Workflow
1. **Data understanding** – 435K rows, 13 columns; target = `rspm`; task = regression.
2. **Cleaning** – read with `latin-1`; dropped `pm2_5` (~98% missing), `spm` (~54% missing), `sampling_date`, `stn_code`; removed duplicates and rows with missing target; unified `type` into Residential / Industrial / Sensitive; clipped target outliers at the 99th percentile.
3. **EDA** – RSPM is right-skewed; northern states (Delhi, UP, Punjab) are the most polluted; strong seasonality (winter high, monsoon low).
4. **Feature engineering** – `year` and `month` from `date`.
5. **Preprocessing** – `ColumnTransformer` inside a `Pipeline`: median imputation + `StandardScaler` for numeric, most-frequent imputation + `OrdinalEncoder` for categorical. Fitted on training data only (no leakage).
6. **Models** – Linear, Ridge, Lasso, ElasticNet, Decision Tree, Gradient Boosting, XGBoost. SVR and AdaBoost excluded (training cost / poor fit).
7. **Tuning** – `GridSearchCV` on XGBoost (16 combinations, 5-fold CV on a 100K-row training sample). Best: `learning_rate=0.05, max_depth=10, min_child_weight=5, n_estimators=500`, CV R² = 0.674.

## Results (test set, 77K rows)
| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.157 | 46.14 | 60.34 |
| Ridge | 0.157 | 46.14 | 60.34 |
| Lasso | 0.156 | 46.28 | 60.37 |
| ElasticNet | 0.146 | 46.91 | 60.75 |
| Decision Tree | 0.452 | 30.64 | 48.67 |
| Gradient Boosting | 0.463 | 35.38 | 48.16 |
| XGBoost (default) | 0.658 | 26.52 | 38.41 |
| **XGBoost (tuned)** | **0.711** | **23.47** | **35.32** |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
share.streamlit.io → **Create app** → repo `NTI-Machine-Learning`, branch `main`, main file `Final_Project/app.py` → **Deploy**.
