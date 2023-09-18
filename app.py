from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import model_from_json
import cv2
from PIL import Image
from PIL.ExifTags import TAGS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
model = None

with open('model/pothole_cnn.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('model/model_weights.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (300, 300))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_exif_data(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        return exif_data
    except (AttributeError, KeyError, IndexError):
        return None
def get_gps_info(exif_data):
    if not exif_data:
        return None

    for tag, value in exif_data.items():
        if TAGS.get(tag) == 'GPSInfo':
            return value
def extract_latitude_longitude(gps_info):
    if not gps_info:
        return None

    latitude = gps_info[2][0] + gps_info[2][1] / 60 + gps_info[2][2] / 3600
    longitude = gps_info[4][0] + gps_info[4][1] / 60 + gps_info[4][2] / 3600
    return latitude, longitude

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
            if True:
                exif_data = get_exif_data(image_path)
                if exif_data:
                    gps_info = get_gps_info(exif_data)
                    if gps_info:
                        latitude, longitude = extract_latitude_longitude(gps_info)
                        print(f"Latitude: {latitude}, Longitude: {longitude}")
                    else:
                        print("GPS information not found in the photo's EXIF data.")
                else:
                    print("EXIF data not found in the photo.")

            class_labels = ['Normal', 'Pothole']
            predicted_label = class_labels[predicted_class]
            return render_template('result.html', prediction=predicted_label)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)