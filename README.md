EmoSense: Real-Time Emotion Recognition using CNN

EmoSense is a cutting-edge emotion recognition system powered by a convolutional neural network (CNN). By leveraging deep learning techniques and real-time image processing, the system can detect and classify seven human emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. This project showcases an implementation of facial emotion recognition using the FER-2013 dataset.
Key Features

    Real-Time Emotion Detection: EmoSense processes images in real-time to identify the emotion expressed on a human face.
    CNN-Based Architecture: Built using TensorFlow and Keras, the model is optimized for accuracy and performance.
    Data Augmentation: Training images are dynamically augmented (rotation, zoom, flip, shear) to prevent overfitting and enhance generalization.
    Batch Normalization & Dropout: These techniques ensure the model remains robust and effective, accelerating training while minimizing overfitting.
    Support for 7 Emotions: Detects a comprehensive range of emotions, allowing for nuanced recognition.

Project Structure

    Data: We utilize the FER-2013 dataset for training and testing, which consists of grayscale images of facial expressions. The dataset is split into a training set and a testing set for model evaluation.
    Model: The CNN is built with three convolutional layers followed by batch normalization, max pooling, and dropout layers to ensure robust training. The final layer is a dense softmax classifier that outputs probabilities for the seven emotion categories.
    Image Processing: Real-time image preprocessing is performed with OpenCV to handle facial detection and alignment, preparing the data for emotion prediction.

Installation

To install the required libraries, run the following command:

bash

pip install -r requirements.txt

Dataset

We use the FER-2013 dataset. If not included, you can download it from Kaggle FER-2013. Make sure to place the dataset in the appropriate train and test directories.

bash

train_dir = "path_to_train_directory"
test_dir = "path_to_test_directory"

Model Architecture

python

# CNN Architecture
model = models.Sequential()

# Layer 1
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

# Layer 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

# Layer 3
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

# Flatten and Dense Layer
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

The model is trained using Adam optimizer and categorical cross-entropy loss, with accuracy metrics.
Usage
Training the Model

To train the model, simply run:

python

history = model.fit(train_generator, epochs=25, validation_data=test_generator)

Saving the Model

The trained model is saved using:

python

model.save('EmoSense2_0.keras')

Real-Time Emotion Detection

To perform real-time emotion detection, load an image and predict the emotion:

python

img_path = 'path_to_image'
img = image.load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
predictions = model.predict(img_array)

Results

After training, the model achieves X% validation accuracy with a loss of Y. The model performs particularly well on emotions such as Happy and Neutral, though subtle expressions like Disgust may require further refinement.
Future Work

    Implementing live webcam integration for real-time emotion detection.
    Improving accuracy on less frequent emotions like Disgust and Surprise.
    Expanding the dataset with more diverse expressions and lighting conditions.
    Exploring transfer learning for higher accuracy.

License

This project is licensed under the MIT License - see the LICENSE file for details.
