import tensorflow as tf
from tensorflow.keras import models, layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('trainset.csv')
test_dataset = pd.read_csv('testset.csv')
image_side = 28
number_channels = 1

dataset_labels = np.array(dataset['label'])
dataset_images = np.array(dataset.iloc[:,1:(image_side * image_side) + 1])
dataset_images = dataset_images / 255

dataset_images= np.array([img.reshape(image_side, image_side, number_channels) for img in dataset_images])

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
history = model.fit(dataset_images, dataset_labels, epochs=6, validation_split=0.2)

test_dataset_images = np.array(test_dataset.iloc[:,0:(image_side * image_side)])
test_dataset_images = test_dataset_images / 255

test_dataset_images= np.array([img.reshape(image_side, image_side, number_channels) for img in test_dataset_images])

predictions = model(test_dataset_images)
predictions = np.array(predictions)

output_data = {'ImageId': np.arange(1, len(predictions) + 1, 1), 'Label': [probabilities.argmax() for probabilities in predictions]}
output_data = pd.DataFrame(data = output_data)
output_data.to_csv('predictions.csv', index = False)

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='learn')
plt.plot(history.history['val_accuracy'], label = 'validation')
plt.legend()
plt.show()