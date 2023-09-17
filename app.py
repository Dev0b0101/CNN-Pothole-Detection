from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import model_from_json
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Load the CNN model
model = None

with open('model/pothole_cnn.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('model/model_weights.h5')

# Define a function to preprocess the image before prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (300, 300))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            image = preprocess_image(image_path)
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)
            class_labels = ['normal', 'pothole']
            predicted_label = class_labels[predicted_class]
            # You can customize this part to display the prediction results as needed.
            return render_template('result.html', prediction=predicted_label)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)