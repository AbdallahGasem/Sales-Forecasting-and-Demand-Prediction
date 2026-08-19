# Sales Forecasting and Demand Prediction

An end-to-end data science project that predicts the sales value of a retail order from its attributes. It covers exploratory analysis of a Superstore sales dataset, feature engineering, a comparison of five regression models, and a Flask web application that serves predictions from the selected model.

Graduation project for the Digital Egypt Pioneers Initiative (DEPI), Ministry of Communications and Information Technology. Supervised by Eng. Mahmoud Talaat.

## Overview

Retailers that misjudge demand end up either holding stock they cannot sell or running out of what customers want. The goal here is a model accurate enough to support inventory, staffing, and marketing decisions, packaged behind an interface a non-technical user can operate.

The pipeline runs: clean the raw transactions, engineer time and categorical features, train and compare regressors, then export the best model as a pickle that a Flask app loads at startup.

## Dataset

Superstore retail data from Kaggle (`stores_sales_forecasting.csv`), covering four years of US orders from 2014 to 2017. The file is the furniture slice of the Superstore set, so every row falls under one product category with sub-categories Bookcases, Chairs, Furnishings, and Tables.

The raw file carries 18 columns: Row ID, Order ID, Order Date, Ship Date, Ship Mode, Customer ID, Customer Name, Segment, Country, City, State, Postal Code, Region, Product ID, Category, Sub-Category, Product Name, and Sales, plus Quantity, Discount, and Profit.

After dropping identifiers and removing Sales outliers by the 1.5 IQR rule, 1,957 rows and 30 features remain, split 1,565 for training and 392 for testing.

## Structure

```
├── Notebooks/
│   ├── Sales-Forecasting-Notebook.ipynb                EDA and modelling, first pass
│   ├── Sales-Forecasting-Notebook-NewData.ipynb        Full pipeline: cleaning, EDA, all five models
│   └── SalesForecasting-AggregatedData-Products.ipynb  Alternative framing: monthly sales per product,
│                                                       split by year (2014-2016 train, 2017 eval)
└── Deployment/FINAL/
    ├── app.py                    Flask app: form input, preprocessing, prediction
    ├── Model_Training.ipynb      Trains the final model and writes model.pkl and columns.pkl
    ├── encoded_NewData.csv       Cleaned and encoded dataset used for training
    ├── templates/index.html      Prediction form
    └── static/css/style.css
```

## Pipeline

### Cleaning

Identifier and constant columns are dropped (Row ID, Order ID, Customer ID, Customer Name, Product ID, Product Name, Postal Code, Country). Postal Code goes because City, State, and Region already carry the geography; Country goes because every row is US. Sales outliers outside the 1.5 IQR fence are removed.

### Feature engineering

| Step | Detail |
| --- | --- |
| Date parsing | Order Date and Ship Date converted to datetime |
| Shipping duration | Ship Date minus Order Date, in days |
| Temporal features | Order year, month, day, and weekday extracted, then the raw date columns dropped |
| One-hot encoding | Category, Sub-Category, Segment, Ship Mode, Region, with the first level dropped |
| Label encoding | State mapped to an integer code |
| City grouping | Top 10 cities by frequency kept, everything else collapsed to "Other", then one-hot encoded |

### Models

Five regressors were trained on the same split and scored on RMSLE, RMSE, MSE, and MAE.

## Results

Hold-out set (80/20 split, `random_state=42`):

| Model | RMSLE | RMSE | MSE | MAE | Train R² | Test R² |
| --- | --- | --- | --- | --- | --- | --- |
| Linear Regression | 1.00 | 179.20 | 32,111.44 | 129.65 | 0.479 | 0.427 |
| Decision Tree | 0.62 | 165.25 | 27,307.89 | 101.26 | 1.000 | 0.513 |
| Random Forest | 0.54 | 127.42 | 16,235.72 | 79.87 | 0.884 | 0.710 |
| XGBoost | 0.55 | 128.84 | 16,599.59 | 80.31 | 0.971 | 0.704 |
| XGBoost (log target) | 0.48 | 132.20 | 17,477.38 | 77.32 | 0.971 | 0.874 |

Five-fold cross-validation on the full dataset:

| Model | RMSLE | RMSE | MSE | MAE |
| --- | --- | --- | --- | --- |
| Linear Regression | 0.999 | 178.697 | 31,932.487 | 130.277 |
| Decision Tree | 0.619 | 158.045 | 24,978.293 | 94.348 |
| Random Forest | 0.515 | 119.429 | 14,263.239 | 75.771 |
| XGBoost | 0.552 | 116.891 | 13,663.398 | 73.344 |
| XGBoost (log target) | 0.464 | 118.420 | 14,023.196 | 69.469 |

Linear regression underfits badly; the single decision tree memorizes the training set (train R² of exactly 1.0) and generalizes poorly. The ensembles close most of that gap.

### Final model

XGBoost trained on a `log1p`-transformed target, with predictions inverted through `expm1`:

```python
XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=6)
```

Sales is right-skewed with a long tail of large orders, so the log transform stabilizes variance and keeps a handful of extreme values from dominating the loss. It gives the lowest RMSLE and MAE of the five, both on the hold-out set and under cross-validation.

## Web application

The Flask app takes 12 inputs, applies the same feature engineering used in training, aligns the resulting columns against the saved training column list, and displays the predicted sales value.

Inputs: Order Date, Ship Date, Ship Mode, Segment, City, State, Region, Category, Sub-Category, Quantity, Discount, Profit. Dates use `MM/DD/YYYY`.

### Running it

```bash
pip install flask pandas numpy scikit-learn xgboost
cd Deployment/FINAL
```

Run `Model_Training.ipynb` to produce `model.pkl` and `columns.pkl`, then:

```bash
python app.py
```

The app serves on `0.0.0.0:5000`. Note that `app.py` currently loads the pickles from absolute Windows paths; change them to relative paths before running elsewhere.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, Jupyter, Flask, HTML and CSS.

## Known limitations

- `app.py` hardcodes absolute Windows paths to `model.pkl` and `columns.pkl`, so the app will not start on another machine without edits. The notebooks read the dataset the same way.
- The app fits a fresh `LabelEncoder` on the single submitted row, which always yields `State_Encoded = 0`. The encoder fitted during training needs to be pickled and reused, otherwise State is effectively ignored at prediction time.
- Profit is used as an input feature, but profit is not known before a sale occurs, which limits the model's use as a true forward-looking forecast.
- The test R² of 0.874 for the log-target model is computed in log space, while the other models' R² values are in the original scale, so the two are not directly comparable. On raw-scale RMSE, plain XGBoost is marginally better.
- The target is the sales value of an individual order line, not an aggregated weekly or monthly series. Lag and rolling-average features are identified as candidates in the notebooks but are not part of the final pipeline.
- Training data covers the furniture category only, so predictions should not be extended to other product lines without retraining.
- The form field is named `Disscount` and values are read positionally, so the input order in `index.html` must stay aligned with the column list in `app.py`.

## Team

Abdallah Gasem, Mazen Karam, Abdallah Sayed, Youssef Ali, Micheal Emad, Mostafa Ahmed
