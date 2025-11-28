import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
from electricity_prediction_ml.costum_transformers import LogTransformer, HourExtractor


MODEL_PATH = "models/model.pkl"  


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_input_df(
    building_id: str,
    site_id: str,
    primary_use: str,
    square_feet: float,
    year_built: int,
    floor_count: float,
    timestamp: datetime,
    air_temperature: float,
    cloud_coverage: float,
    dew_temperature: float,
    precip_depth_1_hr: float,
    sea_level_pressure: float,
    wind_direction: float,
    wind_speed: float,
) -> pd.DataFrame:

    ts = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "building_id": [int(building_id)],
        "meter": [0],
        "timestamp": [ts],
        "site_id": [int(site_id)],
        "primary_use": [primary_use],
        "square_feet": [square_feet],
        "year_built": [year_built],
        "floor_count": [floor_count],
        "air_temperature": [air_temperature],
        "cloud_coverage": [cloud_coverage],
        "dew_temperature": [dew_temperature],
        "precip_depth_1_hr": [precip_depth_1_hr],
        "sea_level_pressure": [sea_level_pressure],
        "wind_direction": [wind_direction],
        "wind_speed": [wind_speed],
    }

    return pd.DataFrame(data)


def main():

    st.set_page_config(
        page_title="Energy Consumption Predictor",
        
        layout="centered",
    )

    st.title("Prédiction de la consommation électrique d'un bâtiment")

  

    model = load_model()

    st.sidebar.header("Informations bâtiment")

    # --- TEXT FIELD building_id ---
    building_id = st.sidebar.text_input("Building ID", value="0")

    # --- TEXT FIELD site_id ---
    site_id = st.sidebar.text_input("Site ID", value="0")

    # --- Primary Use dropdown ---
    PRIMARY_USE_OPTIONS = [
        "Education",
        "Office",
        "Entertainment/Public Assembly",
        "Lodging/Residential",
        "Public Safety",
        "Healthcare",
        "Warehouse/Storage",
        "Food Sales and Service",
        "Service",
        "Technology/Science",
        "Parking",
        "Manufacturing/Industrial",
        "Religious Worship",
        "Utility",
        "Retail",
        "Other",
    ]

    primary_use = st.sidebar.selectbox("Primary Use", PRIMARY_USE_OPTIONS, index=0)

    square_feet = st.sidebar.number_input(
        "Surface (square_feet)", min_value=10.0, value=50000.0, step=100.0
    )

    year_built = st.sidebar.number_input(
        "Année de construction", min_value=1900, max_value=2025, value=2000, step=1
    )

    floor_count = st.sidebar.number_input(
        "Nombre d'étages", min_value=1.0, value=4.0, step=1.0
    )

    st.sidebar.header("🌡️ Conditions météo & temps")

    date_input = st.sidebar.date_input("Date", value=datetime.now().date())
    time_input = st.sidebar.time_input("Heure", value=datetime.now().time())
    timestamp = datetime.combine(date_input, time_input)

    air_temperature = st.sidebar.slider(
        "Air Temperature (°C)", min_value=-30.0, max_value=50.0, value=20.0, step=0.5
    )

    dew_temperature = st.sidebar.slider(
        "Dew Temperature (°C)", min_value=-30.0, max_value=30.0, value=10.0, step=0.5
    )

    cloud_coverage = st.sidebar.slider(
        "Cloud Coverage", min_value=0.0, max_value=10.0, value=5.0, step=1.0
    )

    precip_depth_1_hr = st.sidebar.slider(
        "Precip Depth 1h (mm)", min_value=0.0, max_value=50.0, value=0.0, step=1.0
    )

    sea_level_pressure = st.sidebar.slider(
        "Sea Level Pressure (mbar)", min_value=900.0, max_value=1050.0, value=1013.0, step=1.0
    )

    wind_direction = st.sidebar.slider(
        "Wind Direction (°)", min_value=0.0, max_value=360.0, value=180.0, step=10.0
    )

    wind_speed = st.sidebar.slider(
        "Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=3.0, step=0.5
    )

   

    if st.button("Prédire la consommation"):

        X_input = build_input_df(
            building_id=building_id,
            site_id=site_id,
            primary_use=primary_use,
            square_feet=square_feet,
            year_built=year_built,
            floor_count=floor_count,
            timestamp=timestamp,
            air_temperature=air_temperature,
            cloud_coverage=cloud_coverage,
            dew_temperature=dew_temperature,
            precip_depth_1_hr=precip_depth_1_hr,
            sea_level_pressure=sea_level_pressure,
            wind_direction=wind_direction,
            wind_speed=wind_speed,
        )

        y_pred = model.predict(X_input)[0]

        st.success(f"Consommation prédite : **{y_pred:.2f} kWh** (meter_reading)")
        st.caption("Note : estimation basée sur les données ASHRAE.")


if __name__ == "__main__":
    main()
