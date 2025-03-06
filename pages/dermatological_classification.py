import os
import base64

import numpy as np
import streamlit as st
import tensorflow as tf


from utils_streamlit import reset_st_state
import utils_derm_foundation_model as derm_utils

MODEL = tf.saved_model.load("local_models/fine_tuned_derm_foundation_tf")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set page configuration
st.set_page_config(
    page_title="Dermatological images Classification")

if reset := st.button("Reset"):
    reset_st_state()

st.title("Dermatological images Classification")

st.write(
    """
    Using Health AI Developer Foundations to develop a Dermatological image classification model.
    """
)


st.header("Select or upload an image")
image_option = st.radio("Select or upload an image:", ('Upload', 'eczema.jpg'))

uploaded_image = None
if image_option == 'Upload':
    uploaded_image = st.file_uploader("Image file", type=['JPEG', 'PNG'])
    
# Determining the image source
if uploaded_image is not None:
    image_data = derm_utils.load_process_image(image_bytes=uploaded_image.getvalue())
    st.image(uploaded_image)
elif image_option != 'Upload':
    image_path = os.path.join(BASE_DIR, "sample_data", image_option)
    print(image_path)
    image_data = derm_utils.load_process_image(image_path=image_path)
    st.image(image_data)
else:
    image_data = None
    
# Button to call API
if st.button("Submit") and image_data:
    with st.spinner("Analyzing the image, which can take up to 90 seconds..."):
        # set the model to generate the embeddings
        model = os.path.join(BASE_DIR, "./local_models/derm_foundation_model")
        # Generate the embedding vectors
        embedding_vector = derm_utils.generate_embeddings_from_image(image_data,model)
        
        # Set the classifier model
        classifier_model_path = "./local_models/fine_tuned_derm_foundation_tf"
        pred = derm_utils.predict_using_classifier_model(embedding_vector,classifier_model_path)

        # Format friendly prediction response
        labels = ['Eczema', 'Allergic Contact Dermatitis', 'Insect Bite', 
                          'Urticaria', 'Psoriasis', 'Folliculitis', 
                          'Irritant Contact Dermatitis', 'Tinea', 
                          'Herpes Zoster', 'Drug Rash']
        
        model_response, model_prediction = derm_utils.formating_prediction_response(pred,labels)

        st.markdown(model_response)
        st.markdown(model_prediction, unsafe_allow_html=True)
    pass
else:
    if image_data is None:
        st.warning("Please, upload an image file before submitting.")
