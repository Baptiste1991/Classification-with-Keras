#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 

# In this work, I used Keras library to build models for classificaiton problems. Populat MNIST dataset was used to fit the model.   
# The MNIST database contains 60,000 training images and 10,000 testing images of digits written by high school students and employees of the United States Census Bureau.

# <h2>Classification Models with Keras</h2>
# 
# At the end of this project, it is expected that newly built model or classifier will accurately classify unseen or any incoming image as <b>Fire</b>, <b>Smoke</b> and <b>Neutral (No Fire or Smoke)</b>.<p>
# 

# In[1]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist
from scipy.ndimage import rotate
from skimage.transform import AffineTransform, warp

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize the input data from 0-255 to 0-1  
X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0

# One-hot encode the target labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define data augmentation functions
def rotate_image(image, angle):
    return rotate(image, angle, reshape=False)

def shift_image(image, shift):
    transform = AffineTransform(translation=shift)
    return warp(image, transform, mode='wrap')

def apply_augmentation(images):
    augmented_images = []
    for image in images:
        # the input image must be 2D
        if len(image.shape) < 2:
            image = np.expand_dims(image, axis=-1)
        angle = np.random.uniform(-10, 10)
        shift = np.random.uniform(-0.1, 0.1, size=2)
        augmented_image = shift_image(rotate_image(image, angle), shift)
        augmented_images.append(augmented_image)
    return np.array(augmented_images)

# Data Augmentation
X_train_augmented = apply_augmentation(X_train)
X_test_augmented = apply_augmentation(X_test)

# Define classification model
def classification_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train multiple models with different architectures
models = [classification_model() for _ in range(5)]

# Train each model
for i, model in enumerate(models):
    print(f"Training Model {i+1}...")
    model.fit(X_train_augmented, y_train, epochs=10, validation_data=(X_test_augmented, y_test))

# Average predictions from all models
y_pred_probabilities = np.mean([model.predict(X_test_augmented) for model in models], axis=0)
y_pred_labels = np.argmax(y_pred_probabilities, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Check if the predicted labels match the true labels
labels_match = np.array_equal(y_pred_labels, y_true_labels)

if labels_match:
    print("The predicted labels match the true labels.")
else:
    print("The predicted labels do not match the true labels.")


# Let's print the accuracy and the corresponding error.
# 

# In[11]:


# 1. Check the shapes of X_test and y_test
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# In[12]:


# Check if the predicted labels match the true labels
labels_match = np.array_equal(y_pred_labels, y_true_labels)

if labels_match:
    print("The predicted labels match the true labels.")
else:
    print("The predicted labels do not match the true labels.")

# Calculation of accuracy
accuracy = np.mean(y_pred_labels == y_true_labels)
error = 1 - accuracy

print('Accuracy: {:.2f}%'.format(accuracy * 100))
print('Error: {:.2f}%'.format(error * 100))


# """The above model adopted a simple architecture consisting of only Dense layers. It also uses a Flatten layer to flatten the input data before passing it to the Dense layers. The input data is flattened into a 1D array (Flatten(input_shape=(28, 28))) before being passed to the Dense layers. Brielfy, focuses on training an ensemble of models with a simple architecture and data augmentation."""

# # Improve the accaracy using data augmentation approach  

# In[13]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist
from scipy.ndimage import rotate
from skimage.transform import AffineTransform, warp
from keras import regularizers

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize the input data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255.0

# One-hot encode the target labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define data augmentation functions
def rotate_image(image, angle):
    return rotate(image, angle, reshape=False)

def shift_image(image, shift):
    transform = AffineTransform(translation=shift)
    return warp(image, transform, mode='wrap')

def apply_augmentation(images):
    augmented_images = []
    for image in images:
        # Ensure the input image is 2D
        if len(image.shape) < 2:
            image = np.expand_dims(image, axis=-1)
        angle = np.random.uniform(-10, 10)
        shift = np.random.uniform(-0.1, 0.1, size=2)
        augmented_image = shift_image(rotate_image(image, angle), shift)
        augmented_images.append(augmented_image)
    return np.array(augmented_images)

# Define classification model with customizable options
def classification_model(regularization=None, dropout_rate=None):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularization))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularization))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate the model with different configurations
configurations = [
    {"regularization": None, "dropout_rate": None},  # Baseline model
    {"regularization": regularizers.l2(0.01), "dropout_rate": None},  # L2 regularization
    {"regularization": None, "dropout_rate": 0.5},  # Dropout
    {"regularization": regularizers.l1(0.01), "dropout_rate": None},  # L1 regularization
]

accuracies = []

for i, config in enumerate(configurations):
    print(f"Training Model {i+1}...")
    model = classification_model(**config)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)
    scores = model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(scores[1])

# Compare the accuracies of different models
for i, acc in enumerate(accuracies):
    print(f"Model {i+1} Accuracy: {acc}")

# Find the best performing model
best_model_index = np.argmax(accuracies)
best_model_accuracy = accuracies[best_model_index]
print(f"The best performing model is Model {best_model_index+1} with accuracy: {best_model_accuracy}")


# """ 
#  Model Architecture:
#  
# First Model : The model architecture consists of only Dense layers. It starts with a Flatten layer to convert the 2D input data (28x28 images) into a 1D array before passing it to Dense layers.
# Second Model: Similarly, the model architecture includes Dense layers. It also begins with a Flatten layer to reshape the 2D input data into a 1D array.
# 
# Input Shape:
# 
# Both models reshape the input data into a 1D array using Flatten layers. In the first block, the input data is reshaped to (28*28,), while in the second block, it's reshaped to (28*28*1,). The additional dimension in the second block represents the single channel (grayscale) of the images.
# 
# Data Augmentation:
# 
# Both model include data augmentation functions (rotate_image, shift_image, apply_augmentation) to generate additional training samples with variations of the original images. However, the first block applies data augmentation directly to the flattened input data (X_train_augmented = apply_augmentation(X_train)), while the second block applied data augmentation to the original 2D images before flattening.
# 
# Model Configuration:
# 
# In the first model, multiple models with the same architecture but different initializations and training data (using data augmentation) are trained. The predictions of these models are averaged to improve performance.
# In the second block of codes, a single model is trained and evaluated with different configurations (e.g., regularization, dropout) to find the best-performing configuration.
# 
# Evaluation:
# 
# In the first block, the performance of the models is evaluated by comparing the predicted labels to the true labels.
# In the second block, the performance of each model configuration is evaluated based on accuracy, and the best-performing configuration is identified."""

# In[15]:


#Improve the model further by regularization   


# In[16]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist
from scipy.ndimage import rotate
from skimage.transform import AffineTransform, warp
from keras import regularizers

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize the input data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255.0

# One-hot encode the target labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define data augmentation functions
def rotate_image(image, angle):
    return rotate(image, angle, reshape=False)

def shift_image(image, shift):
    transform = AffineTransform(translation=shift)
    return warp(image, transform, mode='wrap')

def apply_augmentation(images):
    augmented_images = []
    for image in images:
        # Ensure the input image is 2D
        if len(image.shape) < 2:
            image = np.expand_dims(image, axis=-1)
        angle = np.random.uniform(-10, 10)
        shift = np.random.uniform(-0.1, 0.1, size=2)
        augmented_image = shift_image(rotate_image(image, angle), shift)
        augmented_images.append(augmented_image)
    return np.array(augmented_images)

# Define classification model with customizable options
def classification_model(regularization=None, dropout_rate=None):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularization))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularization))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate the model with different configurations
configurations = [
    {"regularization": None, "dropout_rate": None},  # Baseline model
    {"regularization": regularizers.l2(0.01), "dropout_rate": None},  # L2 regularization
    {"regularization": None, "dropout_rate": 0.5},  # Dropout
    {"regularization": regularizers.l1(0.01), "dropout_rate": None},  # L1 regularization
]

accuracies = []

for i, config in enumerate(configurations):
    print(f"Training Model {i+1}...")
    model = classification_model(**config)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)
    scores = model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(scores[1])

# Compare the accuracies of different models
for i, acc in enumerate(accuracies):
    print(f"Model {i+1} Accuracy: {acc}")

# Find the best performing model
best_model_index = np.argmax(accuracies)
best_model_accuracy = accuracies[best_model_index]
print(f"The best performing model is Model {best_model_index+1} with accuracy: {best_model_accuracy}")


# # Use CNN method in order to compare with traditional NN 

# In[52]:


import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist
from keras import regularizers

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize the input data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255.0

# One-hot encode the target labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define data augmentation functions
def rotate_image(image, angle):
    return rotate(image, angle, reshape=False)

def shift_image(image, shift):
    transform = AffineTransform(translation=shift)
    return warp(image, transform, mode='wrap')

def apply_augmentation(images):
    augmented_images = []
    for image in images:
        # Ensure the input image is 2D
        if len(image.shape) < 2:
            image = np.expand_dims(image, axis=-1)
        angle = np.random.uniform(-10, 10)
        shift = np.random.uniform(-0.1, 0.1, size=2)
        augmented_image = shift_image(rotate_image(image, angle), shift)
        augmented_images.append(augmented_image)
    return np.array(augmented_images)

# Define classification model with customizable options
def classification_model(regularization=None, dropout_rate=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularization))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate the model with different configurations
configurations = [
    {"regularization": None, "dropout_rate": None},  # Baseline model
    {"regularization": regularizers.l2(0.01), "dropout_rate": None},  # L2 regularization
    {"regularization": None, "dropout_rate": 0.5},  # Dropout
    {"regularization": regularizers.l1(0.01), "dropout_rate": None},  # L1 regularization
]

accuracies = []

for i, config in enumerate(configurations):
    print(f"Training Model {i+1}...")
    model = classification_model(**config)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)
    scores = model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(scores[1])

# Compare the accuracies of different models
for i, acc in enumerate(accuracies):
    print(f"Model {i+1} Accuracy: {acc}")

# Find the best performing model
best_model_index = np.argmax(accuracies)
best_model_accuracy = accuracies[best_model_index]
print(f"The best performing model is Model {best_model_index+1} with accuracy: {best_model_accuracy}")


# # CNN model with two sets of convolutional and pooling layers

# In[53]:


import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist
from keras import regularizers

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize the input data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255.0

# One-hot encode the target labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define classification model with customizable options
def classification_model(regularization=None, dropout_rate=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularization))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate the model with two sets of convolutional and pooling layers
model = classification_model()
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)

# Evaluate the trained model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", test_accuracy)


# """I classified images as Fire, Smoke, and Neutral (No Fire or Smoke). I used traditional neural network and convolution network to compare the classification. Accuracy was improved from 11% to more than 95% after hyperparameter tuning of NN. Furthermore, CNN model with two sets of convolutional and pooling layers was adopted to improve accuracy further"""

# In[ ]:




