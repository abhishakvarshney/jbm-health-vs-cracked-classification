import os
import numpy as np
import streamlit as st
import requests
from io import BytesIO
import uuid
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import time

start = time.time()

# Define Path
model_path = "./models/model.h5"
model_weights_path = "./models/weights.h5"
test_path = "data/test"

# Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)

# Define image parameters
img_width, img_height = 150, 150

# Prediction Function
def predict(file, display_img):

    x = load_img(file, target_size=(img_width, img_height, 2))
    # Display the test image
    st.image(display_img, use_column_width=True)
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    if answer == 1:
        st.success("Predicted: Healthy")
    elif answer == 0:
        st.success("Predicted: Cracked")

    return answer

st.title("Healthy and Cracked Detection")
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
      image = Image.open(img_file_buffer)
      filename = 'data/test/' + str(uuid.uuid4()) + '.jpg'
      image.save(filename)
else:
      filename = 'data/test/IMG20180905145547.jpg'
      image = Image.open(filename)
result = predict(filename, image)
if filename not in ['data/test/IMG20180905145547.jpg']:
      os.remove(filename)