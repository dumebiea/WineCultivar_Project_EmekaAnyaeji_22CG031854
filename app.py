from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'wine_cultivar_model.pkl')
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    
    if request.method == 'POST':
        try:
            # Extract features from form
            features = [
                float(request.form['alcohol']),
                float(request.form['flavanoids']),
                float(request.form['color_intensity']),
                float(request.form['hue']),
                float(request.form['proline']),
                float(request.form['magnesium'])
            ]
            
            # Predict
            final_features = np.array([features])
            prediction = model.predict(final_features)[0]
            
            # Map to name
            cultivar_map = {0: "Cultivar 1", 1: "Cultivar 2", 2: "Cultivar 3"}
            result = cultivar_map.get(prediction, "Unknown")
            
            prediction_text = f"Predicted Origin: {result}"
            
        except ValueError:
            prediction_text = "Error: Please enter valid numbers."

    return render_template('index.html', prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)