# ⚡ Building Electricity Consumption Prediction (ASHRAE Dataset)

A complete end-to-end Machine Learning project that predicts the electricity consumption (`meter_reading`) of buildings using the **ASHRAE – Great Energy Predictor III** dataset.

The project includes full preprocessing, advanced modeling, an ensemble voting system, and a **Streamlit** web application for interactive predictions.

---

## ✨ Features

This project offers a comprehensive solution for energy prediction:

* **Building Information Input:**
    * Building ID, Site ID
    * Primary Use (e.g., Education, Office, Lodging)
    * Year Built, Floor Count
    * Square Feet
* **Weather Conditions Input:**
    * Air Temperature, Dew Temperature
    * Cloud Coverage & Precip Depth
    * Sea Level Pressure
    * Wind Speed & Wind Direction
* **Time Features:** Date and specific hour (capturing daily cycles).
* **Fully Automated Pipeline:**
    * Missing data handling & Imputation.
    * Encoding (OneHot) & Feature Scaling (StandardScaler).
    * Log transformations for skewed data.
    * Non-linear feature expansion using Splines.

---

## 🔬 Machine Learning Models Explained

The prediction engine is based on a **Weighted Voting Regressor** composed of three optimized models.

### 1. ExtraTrees Regressor
* **Type:** Tree-based ensemble learning.
* **Pros:** Handles non-linearity extremely well, robust to noise, and efficient on large datasets.

### 2. Gradient Boosting Regressor
* **Type:** Sequential tree building to minimize error.
* **Pros:** High accuracy on tabular data.
* **Optimization:** Hyperparameters tuned via `RandomizedSearchCV`.

### 3. Spline Regression (SplineTransformer + Ridge)
* **Type:** Linear regression on non-linear features.
* **Technique:** Uses B-splines to model smooth non-linear relationships (ideal for weather data) combined with Ridge regularization to prevent overfitting.

### 4. Weighted Voting Regressor (The Final Model)
This model combines predictions from the three base learners. The influence of each model is determined by its validation performance:

$$weight = \frac{1}{RMSE_{model}}$$

> **Result:** Models with better accuracy have a higher impact on the final prediction, providing a stable and robust output.

---

## 📊 Preprocessing Pipeline

The pipeline is automatically applied before training and inference using `ColumnTransformer`:

### Custom Transformers
* **`HourExtractor`:** Extracts the hour from the timestamp to capture daily energy usage cycles.
* **`LogTransformer`:** Stabilizes highly skewed numerical features (e.g., `square_feet`).

### Standard Transformations
* **OneHotEncoder:** For categorical features (e.g., `primary_use`).
* **StandardScaler:** For numerical features.
* **SplineTransformer:** Generates smooth non-linear features.
* **Imputation:** Fills missing values to ensure data integrity.

---

## 🗂 Folder Structure

```text
electricity_prediction_ml/
│
├── data/
│   ├── train.csv
│   ├── building_metadata.csv
│   └── weather_train.csv
│
├── models/
│   └── model.pkl              # Final trained pipeline (Preproc + Ensemble)
│
├── notebooks/
│   ├── AED.ipynb              # Exploratory Data Analysis
│   └── Model.ipynb            # Prototype and experiments
│
├── electricity_prediction_ml/
│   ├── custom_transformers.py # Classes: LogTransformer, HourExtractor
│   ├── train_model.py         # Final training script
│   ├── evaluation.py          # Independent test evaluation
│
├── app/
│   └── streamlit_app.py       # Streamlit prediction app
│
└── README.md
