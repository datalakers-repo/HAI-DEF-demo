import tensorflow as tf
from io import BytesIO
from PIL import Image
import numpy as np

def load_process_image(image_path=None, image_bytes=None):
    if image_bytes:
        img = Image.open(BytesIO(image_bytes))
    elif image_path:
        img = Image.open(image_path)
    else:
        raise ValueError("É necessário fornecer 'image_path' ou 'image_bytes'")
    
    buf = BytesIO()
    img.convert('RGB').save(buf, 'PNG')
    return buf.getvalue()

def generate_embeddings_from_image(image_bytes,model):    
    # Load the local model
    loaded_model = tf.saved_model.load(model)

    # Format input
    input_tensor= tf.train.Example(features=tf.train.Features(
        feature={'image/encoded': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_bytes]))
        })).SerializeToString()
    
    # Call inference
    infer = loaded_model.signatures["serving_default"]
    output = infer(inputs=tf.constant([input_tensor]))

    # Extract the embedding vector
    embedding_vector = output['embedding'].numpy().flatten()
    print("Size of embedding vector:", len(embedding_vector))
    return embedding_vector

def predict_using_classifier_model(embedding_vector,classifier_model_path):
    emb = embedding_vector
    
    # The model expects a batch dimension so we convert to (1,6144)
    emb = np.expand_dims(emb, axis=0)
    
    # Load the model
    MODEL = tf.saved_model.load(classifier_model_path)
    
    # Predict
    pred = MODEL.serve(emb)
    return pred