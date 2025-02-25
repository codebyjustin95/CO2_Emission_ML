from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Automatically find the latest saved model
model_dir = r"d:\CO2_Emission_Prediction"
model_files = [f for f in os.listdir(model_dir) if f.startswith("co2_emission_prediction_") and f.endswith(".pkl")]

if not model_files:
    raise FileNotFoundError("No saved model found in the directory.")

model_filename = model_files[0]  # Use the first found model
print(f"Loading model: {model_filename}")

# Load the trained model, scaler, and feature names
model = joblib.load(os.path.join(model_dir, model_filename))
scaler = joblib.load(os.path.join(model_dir, "co2_emission_scaler.pkl"))
feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))  # Ensure this file exists!

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        year = int(request.form['year'])
        country = request.form['country']

        # Create an empty DataFrame with all feature names (set to 0)
        input_data = pd.DataFrame(0, index=[0], columns=feature_names)

        # Assign the entered year
        if "year" in input_data.columns:
            input_data["year"] = year  

        # One-hot encode the country (if it was part of training)
        country_column = f"CountryName_{country}"
        if country_column in input_data.columns:
            input_data[country_column] = 1  # Set the selected country to 1
        else:
            return render_template('result.html', prediction=f"Error: Country '{country}' not found in training data", year=year, country=country)

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)

        return render_template('result.html', prediction=prediction[0], year=year, country=country)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}", year=year, country=country)

if __name__ == '__main__':
    app.run(debug=True)
