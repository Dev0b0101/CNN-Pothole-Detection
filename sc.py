import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np

# Load your H5 model
model_path = '/Users/huixian/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/Hackathon/mindspark-hackathon-Huixian-Gong/new_model.h5'
model = tf.keras.models.load_model(model_path)

# Define a function to preprocess the input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (300, 300))  # Adjust the size as needed
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Specify the path to the image you want to classify
image_path = '/Users/huixian/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/Hackathon/mindspark-hackathon-Huixian-Gong/Dataset-Pics/Pothole/107.jpg'

# Preprocess the input image
input_image = preprocess_image(image_path)

# Make predictions using the model
predictions = model.predict(input_image)
print(predictions)

# You can now work with the 'predictions' variable, which contains the model's output

# For example, if it's a classification model, you can print the predicted class
predicted_class = np.argmax(predictions)
print(f'Predicted Class: {predicted_class}')

# If you have class labels, you can map the class index to a label
class_labels = ['normal', 'pothole']  # Replace with your actual class labels
predicted_label = class_labels[predicted_class]
print(f'Predicted Label: {predicted_label}')
