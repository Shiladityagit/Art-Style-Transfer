
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
