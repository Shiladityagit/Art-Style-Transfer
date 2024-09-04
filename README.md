
You said:
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
    main()              write a readme file so that when i paste there it will look like readme
ChatGPT said:
ChatGPT
Here's a sample README file for your project:

Image Stylization App
This project is a web application that allows users to apply artistic styles to their images using a deep learning model. The application leverages TensorFlow Hub's "Arbitrary Image Stylization" model, which transfers the style of one image onto another, creating a stylized version of the content image.

Features
Content Image Upload: Upload an image that you want to stylize.
Style Image Upload: Upload an image that represents the style you want to apply.
Image Stylization: Apply the style from the style image to the content image.
Display Results: View the original and stylized images within the app.
Technology Stack
Python
TensorFlow
TensorFlow Hub
Streamlit: For creating the web app interface.
Setup Instructions
Clone the Repository:

bash
Copy code
git clone https://github.com/your-repository.git
cd your-repository
Install the required packages: You can install the necessary packages by running:

# Image Stylization App

This project is a web application that allows users to apply artistic styles to their images using a deep learning model. The application leverages TensorFlow Hub's "Arbitrary Image Stylization" model, which transfers the style of one image onto another, creating a stylized version of the content image.

## Features

- **Content Image Upload**: Upload an image that you want to stylize.
- **Style Image Upload**: Upload an image that represents the style you want to apply.
- **Image Stylization**: Apply the style from the style image to the content image.
- **Display Results**: View the original and stylized images within the app.

## Technology Stack

- **Python**
- **TensorFlow**
- **TensorFlow Hub**
- **Streamlit**: For creating the web app interface.

## Model Information

The app uses the `arbitrary-image-stylization-v1-256` model from TensorFlow Hub. This model allows for the transfer of artistic styles to content images with remarkable accuracy and speed.

- **Model Link**: [Arbitrary Image Stylization v1](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

Check this application running here : https://huggingface.co/spaces/Shiladitya123Mondal/Art_style_transfer_app
