import streamlit as st
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import torch

# Load the model and processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# Streamlit app
st.title('Vision Transformer Feature Extraction')

st.write('Upload an image to extract features using the Vision Transformer model.')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess and get features
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the features
    last_hidden_states = outputs.last_hidden_state

    # Display the shape of the output
    st.write(f"Shape of last hidden states: {last_hidden_states.shape}")

    # Optional: Display the feature tensor as a list (be cautious with large tensors)
    st.write("Feature tensor:", last_hidden_states.squeeze().tolist())
