from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Initialize Flask application
app = Flask(__name__)

# Load the model and encoders
model = joblib.load('models/crop_recommendation_model.pkl')
label_encoder_season = joblib.load('models/season_encoder.pkl')
label_encoder_state = joblib.load('models/state_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input data from the form
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    rainfall = float(request.form['rainfall'])
    season = request.form['season']
    state = request.form['state']

    # Check if 'season' value is unseen and fit the encoder if necessary
    if season not in label_encoder_season.classes_:
        label_encoder_season.classes_ = np.append(label_encoder_season.classes_, [season])

    # Check if 'state' value is unseen and fit the encoder if necessary
    if state not in label_encoder_state.classes_:
        label_encoder_state.classes_ = np.append(label_encoder_state.classes_, [state])

    # Encode categorical features
    season_encoded = label_encoder_season.transform([season])[0]
    state_encoded = label_encoder_state.transform([state])[0]

    # Create a feature vector for prediction
    features = np.array([[temperature, humidity, rainfall, season_encoded, state_encoded]])

    # Predict the crop
    predicted_crop = model.predict(features)[0]

    # Render the result page with the input values and predicted crop
    return render_template('results.html', temperature=temperature, humidity=humidity, rainfall=rainfall,
                           season=season, state=state, recommended_crop=predicted_crop)

if __name__ == '__main__':
    app.run(debug=True)
