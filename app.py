import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import PIL
import streamlit as st

# Load the model from TensorFlow Hub
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(uploaded_file):
    max_dim = 512
    img = uploaded_file.read()
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

def stylize_image(content_image, style_image):
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    stylized_image = tf.clip_by_value(stylized_image, 0.0, 1.0)
    return stylized_image.numpy()

def main():
    st.title("Styler")

    st.subheader("Upload Content Image")
    content_image = st.file_uploader("Choose a content image...", type=["jpg", "jpeg", "png"])
    st.subheader("Upload Style Image")
    style_image = st.file_uploader("Choose a style image...", type=["jpg", "jpeg", "png"])

    if content_image is not None and style_image is not None:
        content_img = load_img(content_image)
        style_img = load_img(style_image)

        st.subheader("Original Content Image")
        st.image(tensor_to_image(content_img), use_column_width=True)

        st.subheader("Original Style Image")
        st.image(tensor_to_image(style_img), use_column_width=True)

        if st.button("Stylize"):
            stylized_image = stylize_image(content_img, style_img)
            st.subheader("Stylized Image")
            st.image(tensor_to_image(stylized_image), use_column_width=True)

if __name__ == "__main__":
    main()