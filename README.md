
# EMNIST Character Recognition

This project implements a simple model for character recognition using the EMNIST (Extended Modified National Institute of Standards and Technology) dataset. We use a Sequential model from Keras with Dense layers, trained to classify input images into one of the 47 possible classes (0-9, A-Z, a, b, d, e, f, g, h, n, q, r, t).

## Setup

To run this project, you will need to install the required libraries. Run the following command in your terminal:
Then, clone this repository:

       pip install -r requirements.txt

## Usage

To use the program, simply run the app.py file for the gradio version:

        python app.py
This will launch a Gradio interface in your default web browser, allowing you to draw characters in a sketchpad. Once you draw a character, click 'Submit' to have the model predict the character.

## Live Demo

You can try a live demo of this project on Hugging Face Spaces [here](https://huggingface.co/spaces/Ayushnangia/Emnist-demo).