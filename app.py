import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import pandas as pd
import keras

st.set_page_config(page_title="LeukemiaVision", layout="wide")
st.title("🧬 LeukemiaVision")

keras.config.enable_unsafe_deserialization()

try:
    model = tf.keras.models.load_model(
        "leukemia_model_fixed.h5",
        safe_mode=False
    )
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

class_names = ['Benign', 'Early', 'Pre', 'Pro']

def make_gradcam_heatmap(img_array, model):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap

def nucleus_analysis(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    area = np.sum(thresh == 255)
    ratio = area / (224*224)

    if ratio < 0.15:
        stage = "Benign"
    elif ratio < 0.30:
        stage = "Early"
    elif ratio < 0.50:
        stage = "Pre"
    else:
        stage = "Pro"

    return stage, thresh

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.resize(img, (224,224))

    st.image(img)

    img_array = img / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]
    confidence = float(np.max(pred))
    pred_class = class_names[np.argmax(pred)]

    heatmap = make_gradcam_heatmap(img_array, model)
    heatmap = cv2.resize(heatmap, (224,224))

    overlay = np.uint8(255 * heatmap)
    overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
    gradcam = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    nucleus_stage, thresh = nucleus_analysis(img)

    st.write("Prediction:", pred_class)
    st.write("Confidence:", round(confidence,3))
    st.write("Nucleus:", nucleus_stage)

    st.image(gradcam)
    st.image(thresh)
