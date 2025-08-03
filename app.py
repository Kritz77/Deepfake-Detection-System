from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os

# Load trained model
model = load_model('best_model.h5')

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_filename = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            img = Image.open(file).convert("RGB")
            img = img.resize((224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            result = model.predict(img_array)[0][0]

            if result > 0.5:
                label = "Real"
                confidence = result * 100
            else:
                label = "Fake"
                confidence = (1 - result) * 100

            prediction = f"Prediction: {label} (Confidence: {confidence:.2f}%)"

            # Save image to static/uploads folder
            image_filename = file.filename
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            file.save(save_path)

    return render_template("index.html", prediction=prediction, image_path=image_filename)

if __name__ == "__main__":
    app.run(debug=True)
