from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image, ImageDraw, ImageFont
import subprocess
import os
from datetime import datetime
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def run_detector(detector, path):
    img = load_img(path)
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)
    result = {key: value.numpy() for key, value in result.items()}
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        # Ensure the 'uploads' directory exists
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        # Generate a unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_filename = f"uploaded_image_{timestamp}.jpg"
        image_path = os.path.join("uploads", image_filename)

        # Save the uploaded image
        file.save(image_path)

        # Load your object detection model using TensorFlow Hub
        detector = hub.Module("https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow1/variations/openimages-v4-ssd-mobilenet-v2/versions/1")
        detector_output = detector(tf.convert_to_tensor([image_path]), as_dict=True)
        class_names = detector_output["detection_class_names"]

        # Perform object detection
        detection_result = run_detector(detector, image_path)

        # Draw bounding boxes on the image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        boxes = detection_result["detection_boxes"]
        scores = detection_result["detection_scores"]

        for i in range(min(boxes.shape[0], 10)):
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = f"{class_names[i].decode('ascii')}: {int(100 * scores[i])}%"
            draw.rectangle([xmin * image.width, ymin * image.height, xmax * image.width, ymax * image.height],
                           outline="red", width=2)
            draw.text((xmin * image.width, ymin * image.height), display_str, font=ImageFont.load_default(), fill="red")

        # Save the result image
        result_image_path = "uploads/result.jpg"
        image.save(result_image_path)

        # Return the result image directly instead of the path
        return send_file(result_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
