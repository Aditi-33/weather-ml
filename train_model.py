import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("archive/daily_data.csv")

# Select Indore and arrange records by date
data = data[data["city_name"].str.strip().str.lower() == "indore"].copy()
data["datetime"] = pd.to_datetime(data["datetime"])
data = data.sort_values("datetime").drop_duplicates("datetime")

# Today's weather measurements
features = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
]

# Tomorrow's maximum temperature is our target
data["next_day_temp"] = data["temperature_2m_max"].shift(-1)
data["next_date"] = data["datetime"].shift(-1)

# Only use pairs of consecutive days
data = data[
    data["next_date"] - data["datetime"] == pd.Timedelta(days=1)
].dropna(subset=["next_day_temp"])

X = data[features]
y = data["next_day_temp"]

# Keep the latest 20% for testing
split = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train the model
model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestRegressor(n_estimators=100, random_state=42)
)
model.fit(X_train, y_train)

# Check prediction error
predictions = model.predict(X_test)
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2), "°C")

# Save model and feature names
os.makedirs("models", exist_ok=True)
joblib.dump({"model": model, "features": features}, "models/weather_model.joblib")

print("Model trained and saved successfully!")
