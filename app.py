from flask import Flask, render_template, request, redirect, url_for
import keras.models
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

app = Flask(__name__)

# Load your TensorFlow CNN model
model = tf.keras.models.load_model('/Users/huixian/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/Hackathon/mindspark-hackathon-Huixian-Gong/new_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle image upload
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename != '':
                # Ensure the file is an image (you can add more validation)
                if image_file.filename.endswith(('.jpg', '.jpeg', '.png')):

                    print(type(image_file))
                    print(request.path)
                    # Load and preprocess the image using TensorFlow
                    img = image.load_img(image_file.filename, target_size=(300, 300))  # Adjust target size as needed
                    img = img[:, :, 0] / 255.0  # Normalize the image
                    img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = tf.keras.applications.vgg16.preprocess_input(img)  # Adjust preprocessing for your model

                    # Make predictions using your TensorFlow CNN model
                    predictions = model.predict(img)
                    # Process the predictions as needed

                    return render_template('result.html', predictions=predictions)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
