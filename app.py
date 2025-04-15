import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
import os

app = Flask(__name__)

# Load models and scaler
model_path = 'co2_emission_prediction_random_forest.pkl'
scaler_path = 'co2_emission_scaler.pkl'

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Model type: {type(model)}")
    print(f"Scaler type: {type(scaler)}")
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")

# Load country list and feature names from saved files or define them here
countries = ["China", "United States", "India", "Russia", "Japan", "Germany", 
             "Iran", "South Korea", "Saudi Arabia", "Indonesia", "Canada", 
             "Mexico", "Brazil", "South Africa", "Australia", "United Kingdom", 
             "Turkey", "Italy", "France", "Poland"]

indicators = ["GDP (current US$)", "Population, total", 
              "Energy use (kg of oil equivalent per capita)",
              "Electric power consumption (kWh per capita)", 
              "Urban population (% of total)",
              "Industry, value added (% of GDP)"]

default_values = {
    "GDP (current US$)": 1000000000000,
    "Population, total": 100000000,
    "Energy use (kg of oil equivalent per capita)": 2000,
    "Electric power consumption (kWh per capita)": 5000,
    "Urban population (% of total)": 60,
    "Industry, value added (% of GDP)": 25
}

@app.route('/')
def splash_screen():
    return render_template('particles.html')

@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            print("POST request received")
            feature_dict = {}
            country = request.form.get('country')
            print(f"Country selected: {country}")
            
            for indicator in indicators:
                field_name = indicator.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('%', 'pct')
                value = request.form.get(field_name)
                print(f"Indicator {indicator}: {value}")
                feature_dict[indicator] = float(value) if value else default_values[indicator]
            
            country_features = {f"CountryName_{c}": 0 for c in countries}
            if country in country_features:
                country_features[f"CountryName_{country}"] = 1
            
            feature_dict.update(country_features)
            input_df = pd.DataFrame([feature_dict])
            input_df['Year'] = 2025
            
            if hasattr(scaler, 'feature_names_in_'):
                expected_features = scaler.feature_names_in_
                for feature in expected_features:
                    if feature not in input_df.columns:
                        input_df[feature] = np.zeros(len(input_df))
                input_df = input_df[expected_features]
            
            print(f"Input DataFrame after reordering: {input_df.columns}")
            input_scaled = scaler.transform(input_df)
            print(f"Scaled input shape: {input_scaled.shape}")
            prediction = model.predict(input_scaled)[0]
            prediction_formatted = f"{prediction:,.2f}"
            
            return render_template('predict.html', 
                                  prediction=prediction_formatted,
                                  country=country,
                                  features=feature_dict,
                                  countries=countries,
                                  indicators=indicators,
                                  year=2025)
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template('error.html', error=str(e))
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