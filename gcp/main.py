from flask import Flask, request
import os
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Global variables
model = None
class_names = ["Early Blight", "Late Blight", "Healthy"]
BUCKET_NAME = "pratham-tf-models-1"

# Load model during initialization
def load_model():
    global model
    download_blob(BUCKET_NAME, "models/potatoes.h5", "/tmp/potatoes.h5")
    model = tf.keras.models.load_model("/tmp/potatoes.h5")

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

# Load model at startup
load_model()

@app.route("/", methods=["POST"])
def predict():
    image = request.files["file"]
    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256)) # image resizing
    )

    image = image/255 # normalize the image in 0 to 1 range

    img_array = tf.expand_dims(image, 0) # Corrected variable name here
    predictions = model.predict(img_array)

    print("Predictions:",predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {"class": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
