import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Load and preprocess MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Define and compile the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, epochs=3)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy}")
print(f"Loss: {loss}")

# Save the model
model.save("digits.keras")

# Load and predict custom images
for x in range(1, 6):
    img = cv.imread(f"./{x}.png", 0)
    img = np.invert(np.array([cv.resize(img, (28, 28))]))
    img = tf.keras.utils.normalize(img, axis=1)
    prediction = model.predict(img)
    print(f"Prediction for image {x}: {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
