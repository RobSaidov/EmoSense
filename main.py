#Importing modules

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator



#Setting the directory for the data

train_dir = "C:\\Users\\user\\PycharmProjects\\EmoSense\\fer2013\\train"
test_dir = "C:\\Users\\user\\PycharmProjects\\EmoSense\\fer2013\\test"

# Setting the image and batch sizes
IMG_SIZE = (48, 48)
BATCH_SIZE  = 64

# Creating ImageDataGenerator, basically for normalization
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True, shear_range=0.2)

# Same for test data (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Loading the training data from directories

train_generator = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="grayscale", class_mode="categorical")

# Loading the testing data from dir
test_generator = test_datagen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="grayscale", class_mode="categorical")


# #Building the model - CNN
#
# model = models.Sequential()
#
# #first layer
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
#
# # Second layer
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
#
# # third layer
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
#
# # flatten the output
# model.add(layers.Flatten())
#
# #connected layer
# model.add(layers.Dense(128, activation='relu'))
#
# # output layer, 7 means 7 emotions
# model.add(layers.Dense(7, activation='softmax'))
#
# #Ccompile the model
# model.compile(optimizer ="adam", loss = "categorical_crossentropy", metrics = ['accuracy'])
#
# #print the model summary
# model.summary()
#
# #training the model and validate it:
#
# history = model.fit(train_generator, epochs = 100, validation_data = test_generator)
#
# # Savign the model\
# model.save('EmoSense.keras')

#loading the model
model = keras.models.load_model('EmoSense.keras')

# evaluating the model on test data
loss, accuracy = model.evaluate(test_generator)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

# Loading an image for testing
img_path = 'C:\\Users\\user\\PycharmProjects\\EmoSense\\testPhotosFromInternet\\sad me.jpg'
img = image.load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the model input shape

# normalizing the pixel value
img_array /= 255.0

# predicting the emotion
predictions = model.predict(img_array)
predicted_emotion = np.argmax(predictions)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print(f'The emotion is: {emotion_labels[predicted_emotion]}')