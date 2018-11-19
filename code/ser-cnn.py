#!/usr/bin/python3

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model, to_categorical
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import accuracy_score, f1_score

import glob
import imageio
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
from datetime import datetime

# Image processing
IMAGE_PATH='../test-images'

# 5 emotional categories
NUM_EMOTION=5
SER_H5="ser-2.h5"

image_input_files = sorted(glob.glob(path.join(IMAGE_PATH, '*.jpg')))
images = [imageio.imread(im) for im in image_input_files]

images = np.asarray(images)
n_images = len(image_input_files)

labels = np.zeros(n_images)
image_size = np.asarray([images.shape[0], images.shape[1], images.shape[2], images.shape[3]])
print(image_size)
images = images/256

for i in range(1, n_images):
     labels[i] = int(i / 3)

labels_encoded=to_categorical(labels)
print(labels_encoded)

train_indices=n_images-1

x_train = images[:]
y_train = labels_encoded[:]

#---------------------------------------
# CNN network processing
model = Sequential()

# 1st layer
model.add(Conv2D(filters = 64, kernel_size = (5,5), strides = (3,3), padding = 'same', input_shape = (180,1500,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.2))

#2nd layer
model.add(Conv2D(128, kernel_size = (3,3), strides = (2,2), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.2))

#3rd layer
model.add(Conv2D(256, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(NUM_EMOTION, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

BATCH_SIZE=100
EPOCHS= 600

# Early stopping callback
PATIENCE = 6
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')

# Place the callbacks in a list
#callbacks = [early_stopping, tensorboard]

# Train the model
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

print("saving model")

model.save(SER_H5)

