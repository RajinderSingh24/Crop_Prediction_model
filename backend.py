from flask import Flask, request, render_template
import numpy as np
import pickle

# Load models
dtr = pickle.load(open('models/dtr.pkl', 'rb'))
preprocessor = pickle.load(open('models/preprocessor.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Year = request.form['Year']
    rainfall = request.form['average_rain_fall_mm_per_year']
    pesticides = request.form['pesticides_tonnes']
    temp = request.form['avg_temp']
    area = request.form['Area']
    item = request.form['Item']

    features = np.array([[Year, rainfall, pesticides, temp, area, item]], dtype=object)
    transformed = preprocessor.transform(features)
    prediction = dtr.predict(transformed)[0]

    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
