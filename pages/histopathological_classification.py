import os
import base64

import numpy as np
import streamlit as st
import tensorflow as tf

from io import BytesIO
from PIL import Image as PILImage

from utils_streamlit import reset_st_state

MODEL = tf.saved_model.load("local_models/fine_tuned_path_foundation_tf")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set page configuration
st.set_page_config(
    page_title="Histopathological images Classification")

if reset := st.button("Reset"):
    reset_st_state()

st.title("Histopathological images Classification")

st.write(
    """
    Using Health AI Developer Foundations to develop a histopathological image classification model.
    """
)

st.header("Select or upload an image")
image_option = st.radio("Select or upload an image:", ('Upload', 'histopathology.png'))

uploaded_image = None
if image_option == 'Upload':
    uploaded_image = st.file_uploader("Image file", type=['JPEG', 'PNG'])

# Determining the image source
if uploaded_image is not None:
    image_data = uploaded_image.getvalue()
    st.image(uploaded_image)
elif image_option != 'Upload':
    image_path = os.path.join(BASE_DIR, "sample_data", image_option)
    print(image_path)
    with open(image_path, "rb") as file:
        image_data = file.read()
    st.image(image_data)
else:
    image_data = None

# Button to call API
if st.button("Submit") and image_data:
    with st.spinner("Analyzing the image, which can take up to 90 seconds..."):
        # Open the image, crop it, convert it to RGB format, and display it.
        img = PILImage.open(BytesIO(image_data)).crop((0, 0, 224, 224)).convert('RGB')

        # Convert the image to a Tensor and scale to [0, 1]
        tensor = tf.cast(tf.expand_dims(np.array(img), axis=0), tf.float32) / 255.0

        prediction_savedmodel = MODEL.serve(tensor)

        # Format friendly prediction response
        prediction_percentage = "{:.0%}".format(max(prediction_savedmodel.numpy()[0]))
        if np.argmax(prediction_savedmodel.numpy()[0]) == 0:
            prediction_label = "Benign"
        else:
            prediction_label = "Cancer"
        model_response = f"The image was classified as {prediction_label} with {prediction_percentage} certainty."

        st.markdown(model_response)
    pass
else:
    if image_data is None:
        st.warning("Please, upload an image file before submitting.")