import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
modelo = tf.keras.models.load_model('saved_model/meu_modelo_gestos.keras')

# Add a title to the Streamlit application
st.title("Classificador de Gestos")

# Add an explanatory header
st.header("Faça upload de uma imagem de um gesto para classificação")

# Create a file uploader widget
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Resize the image to 50x50 pixels
    image = image.resize((50, 50))

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Expand the dimensions of the image array to match the model's input shape
    image_array = np.expand_dims(image_array, axis=0)

    # Normalize the image data
    image_array = image_array / 255.0

    st.write("Arquivo carregado com sucesso e pré-processado.")
    # The prediction logic will be added here later

# Use the model to make a prediction
predictions = modelo.predict(image_array)

# Get the index of the class with the highest probability
predicted_class_index = np.argmax(predictions)

# You can add code here later to get the actual class name using the index
st.write(f"Predicted class index: {predicted_class_index}")

# Use the model to make a prediction
predictions = modelo.predict(image_array)

# Get the index of the class with the highest probability
predicted_class_index = np.argmax(predictions)

# Get the list of class names from the training dataset
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_'] # Replace with your actual class names

# Get the predicted class name
predicted_class_name = class_names[predicted_class_index]

# Display the predicted class name
st.write(f"A imagem é classificada como: **{predicted_class_name}**")
