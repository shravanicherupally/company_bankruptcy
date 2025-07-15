from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load(os.path.join("model", "model.pkl"))
scaler = joblib.load(os.path.join("model", "scaler.pkl"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    inputs = {}

    if request.method == "POST":
        try:
            # Get inputs
            company_name = request.form.get("company_name")
            x1 = float(request.form.get("x1"))
            x4 = float(request.form.get("x4"))
            x6 = float(request.form.get("x6"))
            x10 = float(request.form.get("x10"))

            # Store inputs for UI rendering
            inputs = {
                "company_name": company_name,
                "x1": x1,
                "x4": x4,
                "x6": x6,
                "x10": x10
            }

            # Format and scale input
            features = np.array([[x1, x4, x6, x10]])
            features_scaled = scaler.transform(features)

            # Predict
            result = model.predict(features_scaled)[0]
            prediction = "Failed" if result == 1 else "Alive"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, inputs=inputs)

if __name__ == "__main__":
    app.run(debug=True)
