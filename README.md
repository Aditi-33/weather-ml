# Weather Prediction ML

A simple Machine Learning project that predicts the next day's maximum temperature in Indore, using historical weather data.

## About the Project

This project uses historical daily weather records to train a machine learning model. It predicts the next day's maximum temperature based on the previous day's weather measurements.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Random Forest Regression

## Dataset

The project uses a historical weather dataset containing records for multiple Indian cities. The data is filtered to use only records for Indore.

The dataset includes temperature, precipitation, and wind measurements.

## Machine Learning Model

**Algorithm:** Random Forest Regressor

**Input features:**
- Maximum temperature
- Minimum temperature
- Mean temperature
- Precipitation
- Maximum wind speed

**Target:** Next day's maximum temperature.

## Model Evaluation

The model achieved a Mean Absolute Error (MAE) of **1.17°C** on the historical test set.

MAE measures the average absolute difference between the predicted and actual temperatures. This result applies to the test data and does not guarantee future forecast accuracy.

## Project Structure

```text
weather-ml/
├── archive/
│   └── daily_data.csv
├── models/
│   └── weather_model.joblib
├── train_model.py
├── weather_predictor.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

1. Install the required libraries:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Train the model:

   ```bash
   python train_model.py
   ```

3. Generate a prediction:

   ```bash
   python weather_predictor.py
   ```

## Limitations

This model learns from historical weather patterns. It does not use live weather observations or atmospheric forecasts, so its predictions may differ from actual future weather.

## Future Improvements

- Add a simple web interface.
- Include more weather features.
- Compare different regression algorithms.
- Integrate live weather data for improved forecasting.


