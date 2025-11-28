⚡ Building Electricity Consumption Prediction (ASHRAE Dataset)

A complete end-to-end Machine Learning project that predicts the electricity consumption (meter_reading) of buildings using the ASHRAE – Great Energy Predictor III dataset.
The project includes full preprocessing, advanced modeling, an ensemble voting system, and a Streamlit web application for interactive predictions.

✨ Features

Upload or enter building information:

Building ID, Site ID

Primary Use

Year Built, Floor Count

Square Feet

Enter weather conditions:

Air Temperature, Dew Temperature

Cloud Coverage

Sea Level Pressure

Wind Speed & Wind Direction

Precip Depth

Timestamp (Date + Time)

Predict electricity consumption (meter_reading) using a trained ensemble model:

ExtraTreesRegressor

GradientBoostingRegressor

SplineTransformer + Ridge Regression

Weighted Voting Regressor

Fully automated:

Missing data handling

Encoding

Feature scaling

Log transformations

Non-linear feature expansion (splines)

🔬 Machine Learning Models Explained
1. ExtraTrees Regressor

Tree-based ensemble learning.

Pros: Handles non-linearity and large datasets, robust to noise.

2. Gradient Boosting Regressor

Builds trees sequentially to reduce error.

Pros: High accuracy, good for tabular data.

Tuned with RandomizedSearchCV.

3. Spline Regression (SplineTransformer + Ridge)

Uses B-splines to model smooth non-linear relationships.

Ridge regularization controls overfitting.

Excellent for modeling continuous weather/building features.

4. Weighted Voting Regressor

Final model that combines all base learners:

weight = 1 / RMSE_model


Models with better accuracy influence more the final prediction.

Provides strong, stable performance.

📊 Preprocessing Pipeline

Applied automatically before training and prediction:

Custom Transformers
✔ HourExtractor

Extracts the hour from timestamp (captures daily energy cycles).

✔ LogTransformer

Stabilizes skewed features like square_feet.

Other preprocessing

OneHotEncoder for categorical features

StandardScaler for numerical features

SplineTransformer for smooth non-linearity

Imputation of missing values

ColumnTransformer for pipeline assembly

🗂 Folder Structure
electricity_prediction_ml/
│
├── data/
│   ├── train.csv
│   ├── building_metadata.csv
│   └── weather_train.csv
│
├── models/
│   └── model.pkl              # Final trained pipeline
│
├── notebooks/
│   ├── AED.ipynb              # Exploratory Data Analysis
│   └── Model.ipynb            # Prototype and experiments
│
├── electricity_prediction_ml/
│   ├── custom_transformers.py # LogTransformer, HourExtractor
│   ├── train_model.py         # Final training script
│   ├── evaluation.py          # Independent test evaluation
│
├── app/
│   └── streamlit_app.py       # Streamlit prediction app
│
└── README.md

🧪 Exploratory Data Analysis (EDA)

Performed in AED.ipynb:

Missing values analysis

Duplicate analysis

Distributions & histograms

Correlation heatmap

Energy consumption by primary_use

Weather variable analysis

Log-transformations

Detection of influential features

EDA ensures high-quality training data.

🧰 Training Pipeline

train_model.py performs:

Loading & merging the ASHRAE dataset

Preprocessing (ColumnTransformer)

Training 3 optimized models

Combining them with a weighted VotingRegressor

Saving the complete pipeline:

model.pkl


This file contains:

✔ Preprocessing
✔ Custom transformers
✔ Ensemble model

→ Fully ready for inference.

📉 Model Evaluation

evaluation.py tests the final model on previously unseen data (20% test split).

Metrics calculated:

R² Score
RMSE


Ensures consistent performance in real-world usage.

🌐 Streamlit Web Application

Run the prediction UI:

streamlit run app/streamlit_app.py


Features:

Enter all building characteristics

Enter daily weather conditions

Select a date & hour

Click “Predict”

Output:

Predicted consumption: XX.XX kWh


Ideal for energy engineers and facility managers.

⚙ Installation
1. Clone repository
git clone <YOUR_REPO_HERE>
cd electricity_prediction_ml

2. Create virtual environment
python -m venv .venv
.venv/Scripts/activate   # Windows
source .venv/bin/activate  # Linux/macOS

3. Install dependencies
pip install -r requirements.txt

▶ Usage
Train the model
python electricity_prediction_ml/train_model.py

Evaluate performance
python electricity_prediction_ml/evaluation.py

Launch the Streamlit web app
streamlit run app/streamlit_app.py

📝 Notes

.gitignore excludes:

datasets

saved models

virtual environment

The model can be retrained anytime using train_model.py

