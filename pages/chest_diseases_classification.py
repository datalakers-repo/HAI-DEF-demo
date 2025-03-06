import os
import base64

import numpy as np
import streamlit as st
import tensorflow as tf

from io import BytesIO
from PIL import Image as PILImage
from utils_streamlit import reset_st_state

import utils_cxr_foundation_model as cxr_utils

EMBEDDING_MODEL_PATH = "./local_models/cxr_foundation_model"
CLASSIFIER_MODEL_PATH = "./local_models/fine_tuned_cxr_foundation_tf"

# Carregar modelo classificador apenas uma vez
CLASSIFIER_MODEL = tf.saved_model.load("local_models/fine_tuned_cxr_foundation_tf")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set page configuration
st.set_page_config(
    page_title="Chest Fracture Classification")

if reset := st.button("Reset"):
    reset_st_state()

st.title("Chest Fracture Classification")

st.write(
    """
    Using Health AI Developer Foundations to develop a Chest Fracture classification model.
    """
)

st.header("Select or upload an image")
image_option = st.radio("Select or upload an image:", ('Upload', 'rib_fracture.jpeg'))

uploaded_image = None
if image_option == 'Upload':
    uploaded_image = st.file_uploader("Image file", type=['JPEG', 'PNG'])

# Determining the image source
if uploaded_image is not None:
    image_data = cxr_utils.load_process_image(image_bytes=uploaded_image.getvalue())
    st.image(uploaded_image)
elif image_option != 'Upload':
    image_path = os.path.join(BASE_DIR, "sample_data", image_option)
    print(image_path)
    with open(image_path, "rb") as file:
        image_data = cxr_utils.load_process_image(image_path=image_path)
    st.image(image_data)
else:
    image_data = None

# Button to call API
if st.button("Submit") and image_data:
    with st.spinner("Analyzing the image, which can take up to 90 seconds..."):
        
        # Gera o vetor de embeddings a partir da imagem processada
        embedding_vector = cxr_utils.generate_embeddings_from_image(image_data, EMBEDDING_MODEL_PATH)
        
        # Realiza a predição usando o modelo classificador fine-tuned
        pred = cxr_utils.predict_using_classifier_model(embedding_vector, CLASSIFIER_MODEL_PATH)
        
        
        # Extração do valor de predição
        probability = pred.numpy()[0][0]
        fracture_label = "Fracture Detected" if probability > 0.5 else "No Fracture"
        model_response = f"The image was classified as **{fracture_label}** with **{probability:.0%}** certainty."
        
    st.success("Analysis complete!")
    st.subheader("Prediction Output:")
    st.write(model_response)
else:
    if image_data is None:
        st.warning("Please, upload an image file before submitting.")