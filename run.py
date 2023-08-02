import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from emnist import extract_training_samples, extract_test_samples
from matplotlib.widgets import Button

# Load the EMNIST Balanced dataset
x_train, y_train = extract_training_samples('balanced')
x_test, y_test = extract_test_samples('balanced')

# Normalize the input data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(47, activation='softmax')
])

emnist_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

# Load the saved model weights
model.load_weights('emnist_weights.h5')

# Make predictions
predictions = model.predict(x_test)

# Initialize the sample index
sample_index = 0

# Set up the plot
fig, ax = plt.subplots()
im_display = ax.imshow(x_test[sample_index], cmap='gray')
ax.axis('off')
title_text = ax.set_title('')
# Initialize the sample index and reveal state
sample_index = 0
reveal_state = False

# Function to update the plot with a new image and its predicted label
def update_plot(index, reveal=False):
    global reveal_state
    reveal_state = reveal

    im_display.set_data(x_test[index])

    if reveal_state:
        predicted_label = np.argmax(predictions[index])
        title_text.set_text(f'Predicted: {emnist_labels[predicted_label]}, True: {emnist_labels[y_test[index]]}')
    else:
        title_text.set_text('')

    fig.canvas.draw_idle()

# Function to handle the Next button click event
def next_button_onclick(event):
    global sample_index
    sample_index += 1
    if sample_index >= len(x_test):
        sample_index = 0
    update_plot(sample_index, reveal=False)

# Function to handle the Previous button click event
def prev_button_onclick(event):
    global sample_index
    sample_index -= 1
    if sample_index < 0:
        sample_index = len(x_test) - 1
    update_plot(sample_index, reveal=False)

# Function to handle the Reveal button click event
def reveal_button_on_clicked(event):
    global reveal_state
    reveal_state = not reveal_state
    update_plot(sample_index, reveal=reveal_state)

# Display the initial image and its prediction
update_plot(sample_index)

# Create and position the Next, Previous, and Reveal buttons
ax_next = plt.axes([0.8, 0.025, 0.1, 0.04])
ax_prev = plt.axes([0.7, 0.025, 0.1, 0.04])
ax_reveal = plt.axes([0.6, 0.025, 0.1, 0.04])

button_next = Button(ax_next, 'Next')
button_prev = Button(ax_prev, 'Previous')
button_reveal = Button(ax_reveal, 'Reveal')

# Attach the click event handlers to the buttons
button_next.on_clicked(next_button_onclick)
button_prev.on_clicked(prev_button_onclick)
button_reveal.on_clicked(reveal_button_on_clicked)

# Display the plot with buttons
plt.show()
