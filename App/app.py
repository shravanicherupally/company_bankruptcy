from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained model
model_path = os.path.join("model", "model.pkl")
model = joblib.load(model_path)

# Load fitted scaler (used during preprocessing)
scaler_path = os.path.join("model", "scaler.pkl")
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get input values from form
        company = request.form['company']
        x1 = float(request.form['x1'])     # Current Ratio
        x4 = float(request.form['x4'])     # Total Asset Turnover
        x6 = float(request.form['x6'])     # Retained Earnings to Total Assets
        x10 = float(request.form['x10'])   # Net Income to Total Assets

        # Combine inputs into array and scale
        input_features = np.array([[x1, x4, x6, x10]])
        input_scaled = scaler.transform(input_features)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Convert numeric result to label
        prediction_label = "Failed" if prediction == 1 else "Alive"

        return render_template('index.html', prediction=prediction_label, company=company)

    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
