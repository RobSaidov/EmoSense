# EmoSense: AI-Powered Emotion Recognition System

EmoSense is an advanced AI system designed to classify human emotions from facial expressions in real-time using deep learning. Built with Convolutional Neural Networks (CNNs) and trained on the FER-2013 dataset, EmoSense is capable of identifying **seven distinct emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral, making it perfect for applications in human-computer interaction, mental health assessments, and user experience optimization.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Model Architecture](#model-architecture)
4. [Dataset](#dataset)
5. [Getting Started](#getting-started)
6. [Usage](#usage)
7. [Results & Performance](#results-performance)
8. [Contributing](#contributing)

## Project Overview
The goal of EmoSense is to bridge the gap between artificial intelligence and human emotional intelligence. By leveraging the power of deep learning, we enable machines to read and understand facial expressions with remarkable accuracy. Whether you're building emotionally aware AI or need emotion tracking in real-time, EmoSense has you covered. 

This project employs **TensorFlow/Keras** and **OpenCV**, using a CNN that has been fine-tuned with advanced techniques like **Batch Normalization**, **Dropout**, and **Data Augmentation** to prevent overfitting and increase the generalizability of the model.

## Features
- **Real-Time Emotion Detection**: Analyze video frames or images in real-time to classify emotions.
- **Seven Emotion Categories**: Detects emotions such as **Angry, Disgust, Fear, Happy, Sad, Surprise,** and **Neutral**.
- **Optimized CNN Model**: Efficient architecture with dropout layers and batch normalization for improved performance.
- **Data Augmentation**: Uses rotation, zoom, and flipping for training to enhance model robustness.
- **Pre-Trained Model**: Load an already trained model and start predicting emotions immediately.

## Model Architecture
The EmoSense model is a deep CNN that processes 48x48 pixel grayscale images. The architecture has been optimized for emotion detection:

- **Input Layer**: 48x48 grayscale facial image.
- **Convolutional Layers**: Three convolutional layers with ReLU activations for feature extraction.
  - Layer 1: 32 filters, Batch Normalization, MaxPooling, Dropout
  - Layer 2: 64 filters, Batch Normalization, MaxPooling, Dropout
  - Layer 3: 128 filters, Batch Normalization, MaxPooling, Dropout
- **Fully Connected Layer**: A dense layer with 128 units followed by dropout.
- **Output Layer**: Softmax classifier for 7 emotion categories.

This architecture balances complexity and computational efficiency, ensuring fast real-time predictions while maintaining high accuracy.

## Dataset
We use the **FER-2013 dataset**, which contains over 35,000 labeled grayscale images of faces, each categorized into one of seven emotional expressions. The images are 48x48 pixels in size, making them ideal for fast processing through the CNN.

- **Training Data**: ~28,000 images
- **Testing Data**: ~7,000 images

The dataset is preprocessed with normalization (rescaling pixel values to [0, 1]) and augmented to avoid overfitting by applying transformations such as rotations, zoom, and flips.

## Getting Started
Follow these instructions to get a copy of EmoSense running on your local machine:

### Prerequisites
- Python 3.x
- TensorFlow/Keras
- OpenCV
- Pandas, NumPy, Matplotlib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/robsaidov/EmoSense.git
   cd EmoSense
   
2. Set up a Virtual Environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

3. Install Dependencies: Create a requirements.txt file in the project root with the following content:
   -tensorflow
   -pandas
   -numpy
   -matplotlib
   -opencv-python

   Then run:
   ```bash
   pip install -r requirements.txt

5. Download the FER-2013 Dataset:

   -Download the Dataset: You can find the FER-2013 dataset on Kaggle.
   -Place the Dataset: Extract the dataset and place the training and testing folders in the fer2013 directory within the project.

## Usage

  Training the Model

    Prepare your Data: Ensure your dataset is organized into train and test directories as described.
    Run the Training Script:
        Modify the paths in main.py to point to your dataset directories.
        Execute the script:

        ```bash

        python main.py

Using the Model for Prediction

    Load the Model: Use the trained model to make predictions.
    Run the Prediction Script:
        Modify the img_path variable in main.py to point to your test image.
        Execute the script to see the emotion prediction.

Results & Performance

The model's performance is evaluated based on its accuracy and loss metrics on the test dataset. The script will print the validation loss and accuracy after evaluation.
Future Improvements

    Expand Emotion Categories: Include more emotion categories to enhance the model’s robustness.
    Real-Time Video Processing: Integrate with real-time video streams for live emotion detection.
    Fine-Tuning: Explore more advanced architectures and hyperparameter tuning for improved accuracy.
    Deployment: Develop a web or mobile application for easier accessibility of the emotion recognition system.

Contributing

We welcome contributions to EmoSense! If you’d like to contribute, please follow these guidelines:

    Fork the Repository: Create a personal copy of the repository.
    Create a Branch: Work on a feature or fix in a separate branch.
    Submit a Pull Request: Propose your changes and describe them in detail.

Please ensure that your code adheres to our coding standards and includes relevant tests.
License

This project is licensed under the MIT License - see the LICENSE file for details.

  

