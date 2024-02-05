from flask import Flask, request, jsonify, render_template
from model_training import PredictPipeline
import pandas as pd


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the HTML form
        gender = request.form.get('gender')
        age = float(request.form.get('age'))
        height = float(request.form.get('height'))
        weight = float(request.form.get('weight'))
        duration = float(request.form.get('duration'))
        heart_rate = float(request.form.get('heart_rate'))
        body_temp = float(request.form.get('body_temp'))

        # Create a DataFrame with the input data
        input_data = {
     'Gender': [gender],
    'Age': [age],
    'Height': [height],
    'Weight': [weight],
    'Duration': [duration],
    'Heart_Rate': [heart_rate],
    'Body_Temp': [body_temp]
        }
        #Preprocess input_data as needed (e.g., convert to the right format)

        pipeline =  PredictPipeline()
        prediction = pipeline.predict(input_data)

        # For demonstration purposes, returning a dummy prediction
        

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
