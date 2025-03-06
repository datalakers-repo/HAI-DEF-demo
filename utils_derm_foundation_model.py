import os
from PIL import Image
from io import BytesIO
import tensorflow as tf
import matplotlib.pyplot as plt
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
    
def plot_embedding_vector(embedding_vector):
    plt.figure(figsize=(12, 4))
    plt.plot(embedding_vector)
    plt.title('Embedding Vector')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    
def predict_using_classifier_model(embedding_vector,classifier_model_path):
    emb = embedding_vector
    
    # The model expects a batch dimension so we convert to (1,6144)
    emb = np.expand_dims(emb, axis=0)
    
    # Load the model
    MODEL = tf.saved_model.load(classifier_model_path)
    
    # Predict
    pred = MODEL.serve(emb)
    return pred

def prediction_dict(pred, labels):
    pred_dict = {}
    for cond, conf in zip(labels, pred.numpy()[0]):
        pred_dict[cond] = round(float(conf), 3)  # Convertendo explicitamente para float

    # Ordenando o dicionário pelos valores de confiança, do maior para o menor
    sorted_dict = dict(sorted(pred_dict.items(), key=lambda item: item[1], reverse=True))

    print(f"Model's Predictions: {sorted_dict}")
    return sorted_dict

def formating_prediction_response(pred,labels):
    ########## Precisa verificar se é correto pegar a condição de maior probabilidade
    
    # Obtém os valores das previsões como numpy array
    pred_values = pred.numpy()[0]
    
    #cria um dicionario mapeando cada condicao a sua probabilidade
    pred_dict = {label: float(conf) for label, conf in zip(labels, pred_values)}
    
    #ordena as condicoes descrescentemente pela probabilidade
    sorted_preds = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
    

    # Índice da condição com maior probabilidade
    max_index = np.argmax(pred_values)

    # Nome da condição prevista com maior probabilidade
    prediction_label = labels[max_index]

    # Probabilidade formatada como porcentagem
    prediction_percentage = "{:.0%}".format(pred_values[max_index])

    # Resposta formatada
    model_response = f"The image was classified as {prediction_label} with {prediction_percentage} certainty."
    print(model_response)

    # Cria a mensagem formatada com todos os resultados
    response_lines = ["Classification Results:"]
    for condition, score in sorted_preds:
        response_lines.append(f" - **{condition}**: {score:.0%}")
        
    model_prediction = "<br>".join(response_lines)
    print(model_prediction)
    return model_response, model_prediction

# ############### Try out the embedding model ################
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# image_path = os.path.join(BASE_DIR, "sample_data", "eczema.jpg")
# image_bytes = load_process_image(image_path)
# model = os.path.join(BASE_DIR, "derm_foundation_model")
# embedding_vector = generate_embeddings_from_image(image_bytes,model)
# plot_embedding_vector(embedding_vector)

# ############### Try out the classifier model ################
# classifier_model_path = "./local_models/fine_tuned_derm_foundation_tf"
# pred = predict_using_classifier_model(embedding_vector,classifier_model_path)


# ############### Formatting the response ################
# # Precisa verificar se a saída da predição é nessa ordem mesmo
# labels = ['Eczema', 'Allergic Contact Dermatitis', 'Insect Bite', 
#                           'Urticaria', 'Psoriasis', 'Folliculitis', 
#                           'Irritant Contact Dermatitis', 'Tinea', 
#                           'Herpes Zoster', 'Drug Rash']

# # Prediction dict
# prediction_dict(pred,labels)

# # Predicting the condition with greater probability
# formating_prediction_response(pred,labels)