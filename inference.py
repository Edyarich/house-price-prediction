from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd
import sys

sys.path.append('./code/')

from data_processing import preprocess_test_data

app = Flask(__name__)

with open('model.pkl', 'rb') as fd:
    model, metadata = pickle.load(fd)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        input_df = pd.DataFrame(input_data['features'], index=[0])  # Single-row DataFrame
        X_processed, _ = preprocess_test_data(input_df, metadata)

        prediction = model.predict(X_processed)
        prediction_exp = np.exp(prediction) - 1  # Reverse log transformation

        return jsonify({
            'prediction': float(prediction_exp[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
