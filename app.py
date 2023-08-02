import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model and weights
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(47, activation='softmax')
])
model.load_weights('emnist_weights.h5')

# EMNIST labels for display
emnist_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

def predict(image):
    # Preprocess the image
    image = image.reshape(1, 28, 28) / 255.0

    # Make a prediction
    prediction = model.predict(image)

    # Get the index of the highest probability
    predicted_index = np.argmax(prediction[0])

    # Return the label
    return emnist_labels[predicted_index]

# correct this code
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model and weights
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(47, activation='softmax')
])
model.load_weights('emnist_weights.h5')

# EMNIST labels for display
emnist_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

def predict(image):
    # Preprocess the image
    image = image.reshape(1, 28, 28) / 255.0

    # Make a prediction
    prediction = model.predict(image)

    # Get the index of the highest probability
    predicted_index = np.argmax(prediction[0])

    # Return the label
    return emnist_labels[predicted_index]

# correct this code

iface = gr.Interface(
    fn=predict,
    inputs="sketchpad",
    outputs="label"
)

# Run the interface
iface.launch()

# Run the interface
iface.launch()
