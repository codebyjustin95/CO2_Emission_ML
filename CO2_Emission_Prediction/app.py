'''
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
'''

import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
import os

app = Flask(__name__)

# Load models and scaler
model_path = 'co2_emission_prediction_random_forest.pkl'
scaler_path = 'co2_emission_scaler.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load country list and feature names from saved files or define them here
# Sample countries
countries = ["China", "United States", "India", "Russia", "Japan", "Germany", 
             "Iran", "South Korea", "Saudi Arabia", "Indonesia", "Canada", 
             "Mexico", "Brazil", "South Africa", "Australia", "United Kingdom", 
             "Turkey", "Italy", "France", "Poland"]

# Sample indicator names (adjust based on your trained model)
indicators = ["GDP (current US$)", "Population, total", 
              "Energy use (kg of oil equivalent per capita)",
              "Electric power consumption (kWh per capita)", 
              "Urban population (% of total)",
              "Industry, value added (% of GDP)"]

# Default values for each feature (you could calculate these from your training data)
default_values = {
    "GDP (current US$)": 1000000000000,
    "Population, total": 100000000,
    "Energy use (kg of oil equivalent per capita)": 2000,
    "Electric power consumption (kWh per capita)": 5000,
    "Urban population (% of total)": 60,
    "Industry, value added (% of GDP)": 25
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from form
            feature_dict = {}
            
            # Get country and create dummy variables if required
            country = request.form.get('country')
            
            # Get values for each indicator
            for indicator in indicators:
                value = request.form.get(indicator.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('%', 'pct'))
                if value:
                    feature_dict[indicator] = float(value)
                else:
                    feature_dict[indicator] = default_values[indicator]
            
            # Create DataFrame from form data
            input_df = pd.DataFrame([feature_dict])
            
            # Add year feature if used in model
            input_df['Year'] = 2025  # Future prediction year
            
            # Prepare country one-hot encoding if used in model
            # This part depends on how your model was trained
            # For simplicity, we're assuming the model expects these columns
            
            # Scale features
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Format prediction
            prediction_formatted = f"{prediction:,.2f}"
            
            return render_template('predict.html', 
                                  prediction=prediction_formatted,
                                  country=country,
                                  features=feature_dict,
                                  countries=countries,
                                  indicators=indicators)
        
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    # GET request - show the prediction form
    return render_template('predict.html', 
                          countries=countries,
                          indicators=indicators,
                          default_values=default_values)

@app.route('/explore')
def explore():
    # Load sample visualization data
    # In a real app, you'd load actual data here
    
    # Sample data for demonstration
    years = list(range(2000, 2020))
    emissions = {
        "China": [3038, 3366, 3694, 4525, 5288, 5790, 6414, 6791, 7035, 7699, 8257, 9019, 9533, 9572, 9223, 9228, 9123, 9839, 10064, 10175],
        "United States": [5860, 5884, 5879, 5951, 6049, 6082, 5967, 6020, 5833, 5424, 5610, 5444, 5225, 5371, 5412, 5251, 5147, 5073, 5244, 5107],
        "India": [1013, 1041, 1053, 1103, 1153, 1210, 1293, 1410, 1568, 1738, 1751, 1841, 1954, 2019, 2161, 2271, 2309, 2412, 2546, 2616]
    }
    
    # Create time series chart for top emitters
    df = pd.DataFrame({
        'Year': years * 3,
        'Country': ['China'] * 20 + ['United States'] * 20 + ['India'] * 20,
        'CO2 Emissions (Mt)': emissions['China'] + emissions['United States'] + emissions['India']
    })
    
    fig1 = px.line(df, x='Year', y='CO2 Emissions (Mt)', color='Country',
                 title='CO2 Emissions by Top Countries (2000-2019)')
    
    # Create comparison chart for latest year
    latest_data = pd.DataFrame({
        'Country': ['China', 'United States', 'India', 'Russia', 'Japan'],
        'CO2 Emissions (Mt)': [10175, 5107, 2616, 1678, 1106]
    })
    
    fig2 = px.bar(latest_data, x='Country', y='CO2 Emissions (Mt)',
                 title='CO2 Emissions by Top 5 Countries (2019)')
    
    # Convert to JSON for the template
    chart1_json = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    chart2_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('explore.html', chart1JSON=chart1_json, chart2JSON=chart2_json)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
    
