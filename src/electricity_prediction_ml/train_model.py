import pandas as pd
import numpy as np
import joblib

\
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor



#  ========= Custom transformers=====


class HourExtractor(BaseEstimator, TransformerMixin):
    # simple transformer: prend timestamp → extrait l'heure
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.to_datetime(X['timestamp']).dt.hour.to_frame()

    def get_feature_names_out(self, input_features=None):
        return ['hour']


class LogTransformer(BaseEstimator, TransformerMixin):
    #  log1p
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_log" for c in input_features])


# ---------------------------
#   Load data
# ---------------------------

def load_and_merge():
    # chargement des fichiers
    data = pd.read_csv("data/train.csv")
    building = pd.read_csv("data/building_metadata.csv")
    weather = pd.read_csv("data/weather_train.csv")

    # merges en chaîne
    df = data.merge(building, on="building_id", how="left")
    df = df.merge(weather, on=["site_id", "timestamp"], how="left")

    # on veut seulement meter = 0 (électricité)
    df = df[df["meter"] == 0]

    return df


# ---------------------------
#   Preprocessing
# ---------------------------

def build_preprocessing():

    # pipeline square_feet → log → scale
    log_area = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("log", LogTransformer()),
        ("sc", StandardScaler())
    ])

    # pipeline categorical → impute + onehot
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # pipeline num simple
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])

    # assemblage
    pre = ColumnTransformer([
        ("log_sqft", log_area, ["square_feet"]),
        ("hour", HourExtractor(), ["timestamp"]),
        ("cat", cat_pipe, ["primary_use", "site_id"]),
        ("num", num_pipe, [
            "air_temperature", "cloud_coverage", "dew_temperature",
            "precip_depth_1_hr", "sea_level_pressure",
            "wind_direction", "wind_speed"
        ]),
        ("floor", SimpleImputer(strategy="median"), ["floor_count"])
    ])

    return pre


# ---------------------------
#   Training models
# ---------------------------

def train_models(X, y):
    idx = np.random.choice(X.shape[0], 40000, replace=False)
    idx2 = np.random.choice(X.shape[0], 1000000, replace=False)
    X_all_data = X[idx2]
    y_all_data = y.iloc[idx2]
    X = X[idx]
    y = y.iloc[idx]

    # ExtraTrees d'abord
    
    ext = ExtraTreesRegressor(n_jobs=15, random_state=47)
    params_ext = {"max_features": np.arange(5, X.shape[1], 5)}

    search_ext = GridSearchCV(ext, params_ext, cv=3, scoring="neg_mean_squared_error")
    search_ext.fit(X, y)
    best_ext = search_ext.best_estimator_
    print(search_ext.best_params_)
    print('fin extrTree ')
    # GradientBoosting
    GB = GradientBoostingRegressor(random_state=47)
    params_GB = {
        "n_estimators": [200, 300, 400],
        "max_depth": range(3, 5),
        "learning_rate": [ 0.1, 0.2]
    }
    
    search_GB = RandomizedSearchCV(
        GB, params_GB, n_iter=10, cv=3, scoring="neg_mean_squared_error"
    )
    search_GB.fit(X, y)
    best_GB = search_GB.best_estimator_
    print(search_GB.best_params_)
    print('fin gradient boosting')
    # spline + ridge
    spline_model = Pipeline([
        ("spline", SplineTransformer(degree=4, n_knots=8)),
        ("ridge", Ridge(alpha=0.1))
    ])
    
    params_ridge = {
        "spline__degree": [3, 4, 5],
        "spline__n_knots": range(4, 12),
        "ridge__alpha": [0.01,0.2, 0.1, 1.0]
    }

    search_ridge = RandomizedSearchCV(
        spline_model, params_ridge,
        n_iter=10, cv=3,
        scoring="neg_mean_squared_error"
    )
    search_ridge.fit(X, y)

    best_ridge = search_ridge.best_estimator_
    print('best params for ridge : ')
    print(search_ridge.best_params_)
    print('fin  ridge' )

    # calcul poids → basé sur RMSE
    rmse_ext = np.sqrt(mean_squared_error(y, best_ext.predict(X)))
    rmse_gb = np.sqrt(mean_squared_error(y, best_GB.predict(X)))
    rmse_ridge = np.sqrt(mean_squared_error(y, best_ridge.predict(X)))

    weights = [1/rmse_ext, 1/rmse_gb, 1/rmse_ridge]

    # voting
    voting = VotingRegressor(
        estimators=[
            ("ext", best_ext),
            ("gb", best_GB),
            ("ridge", best_ridge)
        ],
        weights=weights
    )
    print('fitting the entire data')
    voting.fit(X_all_data, y_all_data)
    return voting


# ---------------------------
#   Main pipeline
# ---------------------------

def main():
    print("Loading data")
    data = load_and_merge()

    X = data.drop("meter_reading", axis=1)
    y = data["meter_reading"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("preprocessing")
    preprocessing = build_preprocessing()

    print("Fitting preprocessing")
    preprocessing.fit(X_train)

    X_train_tr = preprocessing.transform(X_train)

    print("Training models")
    final_model = train_models(X_train_tr, y_train)

   
    final_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", final_model)
    ])
    y_pred = final_model.predict(X_test)
    score = r2_score(y_test,y_pred)
    print('r2 : ')
    print(str(score))
    print("aving model.pkl")
    joblib.dump(final_pipeline, "model.pkl")

    print("Done !")


if __name__ == "__main__":
    main()
