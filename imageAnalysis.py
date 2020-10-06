import tensorflow as tf
from tensorflow.keras import models, layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('trainset.csv')
dataset_size = len(dataset)
image_side = 28
number_channels = 1

dataset_labels = np.array(dataset['label'])
dataset_images = np.array(dataset.iloc[:,1:(image_side * image_side) + 1])
dataset_images = dataset_images / 255

dataset_images= np.array([img.reshape(image_side, image_side, number_channels) for img in dataset_images])

learn_labels = dataset_labels[0:int(0.8*dataset_size)]
validation_labels = dataset_labels[int(0.8*dataset_size):dataset_size]

learn_images = dataset_images[0:int(0.8*dataset_size)]
validation_images = dataset_images[int(0.8*dataset_size):dataset_size]

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_side, image_side, number_channels)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
history = model.fit(learn_images, learn_labels, epochs=8, validation_data=(validation_images, validation_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()