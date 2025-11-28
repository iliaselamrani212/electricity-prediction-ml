from costum_transformers import LogTransformer ,HourExtractor
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
def load_and_merge():
    data = pd.read_csv("data/train.csv")
    building = pd.read_csv("data/building_metadata.csv")
    weather = pd.read_csv("data/weather_train.csv")

    df = data.merge(building, on="building_id", how="left")
    df = df.merge(weather, on=["site_id", "timestamp"], how="left")

    df = df[df["meter"] == 0]
    return df


# ========= MAIN evaluation ==========

def main():
    print("Loading saved model…")
    model = joblib.load("models/model.pkl")

    print("Loading dataset…")
    df = load_and_merge()

    X = df.drop("meter_reading", axis=1)
    y = df["meter_reading"]

    print("Splitting test set…")
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Predicting on test set…")
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n===== MODEL EVALUATION =====")
    print(f"R² score : {r2:.4f}")
    print(f"RMSE     : {rmse:.4f}")


if __name__ == "__main__":
    main()