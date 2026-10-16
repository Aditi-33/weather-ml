
import pandas as pd
import joblib

# Load the trained model
saved = joblib.load("models/weather_model.joblib")
model = saved["model"]
features = saved["features"]

# Load Indore's weather data
data = pd.read_csv("archive/daily_data.csv")
data = data[data["city_name"].str.strip().str.lower() == "indore"].copy()
data["datetime"] = pd.to_datetime(data["datetime"])
data = data.sort_values("datetime").drop_duplicates("datetime")

# Get the latest day's weather
latest = data.iloc[[-1]]
today = latest["datetime"].iloc[0]

# Predict the next day's maximum temperature
prediction = model.predict(latest[features])[0]

print("\n--- Indore Weather Prediction ---")
print("Latest data date:", today.strftime("%d-%m-%Y"))
print("Prediction date:", (today + pd.Timedelta(days=1)).strftime("%d-%m-%Y"))
print("Predicted maximum temperature:", round(prediction, 1), "°C")
print("\nNote: This is an ML estimate, not a live weather forecast.")
