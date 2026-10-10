
import pickle
from flask import Flask, request, render_template
import numpy as np

application = Flask(__name__)
app = application

# Load Ridge model and StandardScaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Get input values from the HTML form
            Temperature = float(request.form.get("Temperature"))
            RH = float(request.form.get("RH"))
            Ws = float(request.form.get("Ws"))
            Rain = float(request.form.get("Rain"))
            FFMC = float(request.form.get("FFMC"))
            DMC = float(request.form.get("DMC"))
            ISI = float(request.form.get("ISI"))
            Classes = float(request.form.get("Classes"))
            Region = float(request.form.get("Region"))

            # Arrange inputs in the training feature order
            new_data = np.array([[
                Temperature, RH, Ws, Rain, FFMC,
                DMC, ISI, Classes, Region
            ]])

            # Scale the input and predict
            new_data_scaled = scaler.transform(new_data)
            prediction = ridge_model.predict(new_data_scaled)

            result = round(float(prediction[0]), 2)

            print("FWI Prediction:", result)

            return render_template("home.html", result=result)

        except Exception as e:
            return f"Prediction error: {e}", 500

    return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
