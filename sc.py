import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np

model_path = '/Users/huixian/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/Hackathon/mindspark-hackathon-Huixian-Gong/new_model.h5'
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (300, 300))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

image_path = '/Users/huixian/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/Hackathon/mindspark-hackathon-Huixian-Gong/Dataset-Pics/Pothole/107.jpg'

input_image = preprocess_image(image_path)

predictions = model.predict(input_image)
print(predictions)

predicted_class = np.argmax(predictions)
print(f'Predicted Class: {predicted_class}')

class_labels = ['normal', 'pothole']
predicted_label = class_labels[predicted_class]
print(f'Predicted Label: {predicted_label}')
