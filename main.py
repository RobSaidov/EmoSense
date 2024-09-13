#Importing modules

import tesorflow as tf
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