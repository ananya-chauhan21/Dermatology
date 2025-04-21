from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and label decoder
model = joblib.load('dermatology_rf_model.pkl')
label_decoder = joblib.load('label_decoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['age']),
            int(request.form['erythema']),
            int(request.form['scaling']),
            int(request.form['definite_borders']),
            int(request.form['itching']),
            int(request.form['koebner']),
            int(request.form['family_history']),
            int(request.form['scalp_involvement']),
            int(request.form['follicular_papules']),
            int(request.form['acanthosis'])
        ]
        prediction = model.predict([features])[0]
        disease = label_decoder[prediction]

        return render_template('index.html', prediction_text=f'Predicted Condition: {disease}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
