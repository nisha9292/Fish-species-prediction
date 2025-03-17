from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        length = float(request.form.get('length', 0))
        height = float(request.form.get('height', 0))
        width = float(request.form.get('width', 0))

        features = np.array([[length, height, width]])
        prediction = model.predict(features)[0]
        
        return jsonify({'species': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

    