import tensorflow as tf

def load_cnn_model(model_path):
    try:
        model = tf.keras.models.model_from_json(open(model_path).read())
        return model
    except Exception as e:
        raise Exception(f"Error loading the model: {str(e)}")
    
def predict(model, data):
    predictions = model.predict(data)
    return predictions
