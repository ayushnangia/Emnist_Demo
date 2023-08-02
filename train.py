import tensorflow as tf
import numpy as np
from emnist import extract_training_samples, extract_test_samples

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

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Save the model weights
model.save_weights('emnist_weights.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
