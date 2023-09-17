import tensorflow as tf

# Define a function to load your model
def load_cnn_model(model_path):
    try:

        model = tf.keras.models.model_from_json(open(model_path).read())
        return model
    except Exception as e:
        raise Exception(f"Error loading the model: {str(e)}")

# You can also define a function for making predictions
def predict(model, data):
    # Perform any necessary data preprocessing here
    # For example, resize and preprocess images
    # Assuming 'data' is in a format suitable for your model

    # Make predictions using the loaded model
    predictions = model.predict(data)

    # Process the predictions as needed
    # For example, convert to class labels or return probabilities

    return predictions
