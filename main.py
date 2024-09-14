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


#Building the model - CNN

model = models.Sequential()

#first layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Second layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# third layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# flatten the output
model.add(layers.Flatten())

#connected layer
model.add(layers.Dense(128, activation='relu'))

# output layer, 7 means 7 emotions
model.add(layers.Dense(7, activation='softmax'))

#Ccompile the model
model.compile(optimizer ="adam", loss = "categorical_crossentropy", metrics = ['accuracy'])

#print the model summary
model.summary()

#training the model and validate it:

history = model.fit(train_generator, epochs = 100, validation_data = test_generator)

# Savign the model\
model.save('EmoSense.keras')