from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained model and scaler
model_path = os.path.join("model", "model.pkl")
model = joblib.load(model_path)

scaler_path = os.path.join("model", "scaler.pkl")
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Retrieve form inputs
        company = request.form['company']
        x1 = float(request.form['x1'])     # Current Ratio
        x4 = float(request.form['x4'])     # Total Asset Turnover
        x6 = float(request.form['x6'])     # Retained Earnings to Total Assets
        x10 = float(request.form['x10'])   # Net Income to Total Assets

        # Prepare feature array and scale it
        input_features = np.array([[x1, x4, x6, x10]])
        input_scaled = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_label = "Failed" if prediction == 1 else "Alive"

        # Print details for debugging
        print("Prediction Details:")
        print(f"Company: {company}")
        print(f"X1: {x1}, X4: {x4}, X6: {x6}, X10: {x10}")
        print(f"Prediction: {prediction_label}")

        # Pass all values back to template
        return render_template('index.html',
                               prediction=prediction_label,
                               company=company,
                               x1=x1, x4=x4, x6=x6, x10=x10)

    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
