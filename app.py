from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

<<<<<<< Updated upstream
=======
dummy_data = pd.read_csv("Synthetic_Forest_fire_dataset.csv")
>>>>>>> Stashed changes

@app.route('/')
def hello_world():
    return render_template("forest_fire.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        area = request.form['area']
        oxygen = float(request.form['oxygen'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        vegetation_density = float(request.form['vegetation_density'])
        proximity_to_water = request.form['proximity_to_water']
        soil_type = request.form['soil_type']

        # Create a new DataFrame with the user input
        user_data = pd.DataFrame([[area, oxygen, temperature, humidity, wind_speed, vegetation_density, proximity_to_water, soil_type]],
                                columns=["Area", "Oxygen", "Temperature", "Humidity", "Wind Speed", "Vegetation Density", "Proximity to Water", "Soil Type"])

        # Make a prediction
        prediction = model.predict(user_data)
        probability = model.predict_proba(user_data)[:, 1]

        prediction_text = 'Your Forest is in Danger.' if prediction[0] == 1 else 'Your Forest is Safe.'
        prediction_class = 'danger' if prediction[0] == 1 else 'safe'

        return render_template('forest_fire.html', prediction=prediction_text, prediction_class=prediction_class, probability=f'{probability[0]:.2f}')
    except Exception as e:
        # Handle errors and display an error message
        error_message = f"An error occurred: {str(e)}"
        return render_template('forest_fire.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
