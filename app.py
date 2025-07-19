from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Load your model
model = load_model('model.h5')  # Change name if your file differs

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data = data['image']
        image_data = image_data.split(",")[1]  # Remove base64 header
        image_bytes = base64.b64decode(image_data)

        # Convert to PIL image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))  # match model input shape
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]

        return jsonify({
            'age': str(int(prediction[0])),
            'gender': 'Male' if prediction[1] > 0.5 else 'Female',
            'bmi': round(float(prediction[2]), 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
