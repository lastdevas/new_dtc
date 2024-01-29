import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import requests
from io import BytesIO
import tarfile
import os

# Google Drive file ID for the model
file_id = '1WMUbo1u8a5lwfuBSA2NOPCh58xQkaImh'
output_file = 'saved_model.pb'
url = f'https://drive.google.com/uc?id={file_id}'

# Download the model file from Google Drive using requests
response = requests.get(url)
if response.status_code == 200:
    with open(output_file, 'wb') as f:
        f.write(response.content)
else:
    st.error(f"Failed to download the model. Status code: {response.status_code}")
    st.stop()

# Load the saved model
model = tf.saved_model.load(output_file)

# Load label map (replace 'label_map.pbtxt' with your label map file)
label_map_path = 'path/to/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

def detect_objects(image):
    # Perform object detection
    img_array = np.array(image)
    input_tensor = tf.convert_to_tensor(np.expand_dims(img_array, 0), dtype=tf.float32)
    detections = model(input_tensor)

    # Visualization of the results of a detection
    vis_util.visualize_boxes_and_labels_on_image_array(
        img_array[0],
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.30,
        agnostic_mode=False
    )

    return Image.fromarray(img_array)

def main():
    st.title("Object Detection with Streamlit")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Objects"):
            image = Image.open(uploaded_file)
            detected_image = detect_objects(image)
            st.image(detected_image, caption="Detected Objects", use_column_width=True)

if __name__ == "__main__":
    main()
