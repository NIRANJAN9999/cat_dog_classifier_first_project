from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "cat_dog_classifer.h5"
model = load_model(MODEL_PATH)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper function to check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Match your model's input size
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file part in the request", 400
        
        file = request.files['file']
        if file.filename == '':
            return "No file selected for upload", 400

        if file and allowed_file(file.filename):
            # Save the file temporarily
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            # Preprocess the image and make a prediction
            img_array = preprocess_image(filepath)
            prediction = model.predict(img_array)
            result = "Dog" if prediction[0] > 0.5 else "Cat"

            # Pass the result to the webpage
            return render_template('index.html', result=result, img_path=filepath)

    return render_template('index.html', result=None, img_path=None)

if __name__ == '__main__':
    app.run(debug=True)
