from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
import cv2


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

export_path = "C:/Users/Vignesh/test/templates/saved_models/1685898848"
model = tf.keras.models.load_model(export_path)
with open(r"C:\Users\Vignesh\Desktop\dipproject\categories.json", 'r') as f:
    cat_to_name = json.load(f)
    classes = list(cat_to_name.values())


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded.')

    file = request.files['file']

    if not allowed_file(file.filename):
        return render_template('index.html', error='Invalid file type.')

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    if not os.path.exists(file_path):
        return render_template('index.html', error='Failed to save the file.')

    img = load_img(file_path, target_size=(299, 299))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)

    pred = model.predict(img_array)
    predicted_class_index = np.argmax(pred)
    predicted_class = classes[predicted_class_index]

    output_filename = generate_output_image(filename)

    return render_template('index.html', result=predicted_class, filename=filename, output_filename=output_filename)


def generate_output_image(filename):
    input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_filename = f'output_{filename}'
    output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    img = Image.open(input_filepath)
    if img is None:
        return None

    output_img = process_image(img)
    output_img.save(output_filepath, 'JPEG')

    return output_filename


def process_image(img):
    image = np.array(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    output_img = Image.fromarray(image)

    return output_img


if __name__ == '__main__':
    app.run(debug=True)

