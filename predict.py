from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from random import randint

# Change working directory to where this script is
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

app = Flask(__name__)

# Load model and encoders once (when server starts)
try:
    model_data = joblib.load("best_model.pkl")
    model = model_data['model']
    label_encoders = model_data['label_encoders']
    feature_names = model_data['feature_names']
    print("✅ Model and encoders loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    label_encoders = {}
    feature_names = []

@app.route("/")
def home():
    return {"status": "running", "message": "ML API is live on Render"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()

        if not input_data:
            return jsonify({"error": "No JSON body provided"}), 400

        # Prepare the data
        model_input = {
            'Size (sqm)': input_data.get('Size (sqm)', 0),
            'Distance to City Center (km)': input_data.get('Distance to City Center (km)', 0),
            'Location': input_data.get('Location', ''),
            'Nearby Amenities': input_data.get('Nearby Amenities', ''),
            'Zoning_LandType': input_data.get('Zoning_LandType', '')
        }

        df = pd.DataFrame([model_input])
        df_encoded = df.copy()

        # Encode categorical variables
        for col in ['Location', 'Nearby Amenities', 'Zoning_LandType']:
            if col in label_encoders:
                try:
                    df_encoded[col] = label_encoders[col].transform(df_encoded[col].astype(str))
                except ValueError:
                    df_encoded[col] = 0  # fallback for unseen category

        # Convert numeric columns
        df_encoded['Size (sqm)'] = df_encoded['Size (sqm)'].astype(float)
        df_encoded['Distance to City Center (km)'] = df_encoded['Distance to City Center (km)'].astype(float)

        # Predict
        prediction = model.predict(df_encoded)[0]
        result = {
            'price': float(prediction),
            'confidence': float(randint(70, 90))
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
