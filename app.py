import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("my_model.keras")

# Function to preprocess image for prediction
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to the input shape of your model
    image = np.array(image, dtype=np.float32)  # Convert to NumPy array

    # Check if the image has 3 channels (RGB)
    if image.shape[-1] == 3:
        # Normalize pixel values to [0, 1]
        image = np.clip(image, 1, 256)  

        # Convert to TensorFlow tensor and add batch dimension
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)  # Add batch dimension
        return image
    else:
        raise ValueError("Uploaded image is not in RGB format.")

# Function to make a prediction
def predict(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    return predictions

# Streamlit app structure
st.title("Potato Disease Classification")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Classifying...")
    predictions = predict(image)
    
    # Assuming you have 3 classes: Healthy, Disease A, Disease B
    class_names = ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy']
    predicted_class = class_names[np.argmax(predictions)]
    
    st.write(f"Prediction: {predicted_class}")
