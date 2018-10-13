#!/usr/bin/python3

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import accuracy_score, f1_score

import glob
import numpy as np
import os.path as path
import imageio
import matplotlib.pyplot as plt
from datetime import datetime

# Image processing
GOOD_IMAGE_PATH='/home/chainwu/AI/spectrum/good/scaled'
BAD_IMAGE_PATH='/home/chainwu/AI/spectrum/bad/scaled'
# 5 emotional categories
NUM_EMOTION=5

good_path = glob.glob(path.join(GOOD_IMAGE_PATH, '*.jpg'))
bad_path = glob.glob(path.join(BAD_IMAGE_PATH, '*.jpg'))
file_paths = good_path + bad_path
images = [imageio.imread(path) for path in file_paths]
images = np.asarray(images)

image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
images = images/256

n_images = images.shape[0]
labels = np.zeros(n_images)
for i in range(len(good_path)):
    labels[i] = 1

#print(labels)

# Training set processing
# Split into test and training sets
TRAIN_TEST_SPLIT = 0.3

# Split at the given index
split_index = int(TRAIN_TEST_SPLIT * n_images)
shuffled_indices = np.random.permutation(n_images)
train_indices = shuffled_indices[0:split_index]
test_indices = shuffled_indices[split_index:]

# Split the images and the labels
x_train = images[train_indices, :, :]
y_train = labels[train_indices]
x_test = images[test_indices, :, :]
y_test = labels[test_indices]

# Visualize images

def visualize_data(positive_images, negative_images):
    # INPUTS
    # positive_images - Images where the label = 1 (True)
    # negative_images - Images where the label = 0 (False)

    figure = plt.figure()
    count = 0
    for i in range(positive_images.shape[0]):
        count += 1
        figure.add_subplot(2, positive_images.shape[0], count)
        plt.imshow(positive_images[i, :, :])
        plt.axis('off')
        plt.title("1")

        figure.add_subplot(1, negative_images.shape[0], count)
        plt.imshow(negative_images[i, :, :])
        plt.axis('off')
        plt.title("0")
    plt.show()

# Number of positive and negative examples to show
N_TO_VISUALIZE = 10

# Select the first N positive examples
positive_example_indices = (y_train == 1)
positive_examples = x_train[positive_example_indices, :, :]
positive_examples = positive_examples[0:N_TO_VISUALIZE, :, :]

# Select the first N negative examples
negative_example_indices = (y_train == 0)
negative_examples = x_train[negative_example_indices, :, :]
negative_examples = negative_examples[0:N_TO_VISUALIZE, :, :]

# Call the visualization function
#visualize_data(positive_examples, negative_examples)

#---------------------------------------
# CNN network processing
model = Sequential()

# 1st layer
model.add(Conv2D(filters = 64, kernel_size = (5,5), strides = (3,3), padding = 'same', input_shape = (256,256,3), activation = 'relu'))
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
model.add(Dense(NUM_CLASS, activation='softmax'))

model.summary()
#plot_model(model, to_file='model.png', show_shapes=True)
#optim = SGD(lr=10e-3,momentum=0.0, decay=0.0, nesterov=False)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

BATCH_SIZE=100
EPOCHS= 100

# Early stopping callback
PATIENCE = 6
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')

# TensorBoard callback
LOG_DIRECTORY_ROOT = ''
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "/tmp/".format(LOG_DIRECTORY_ROOT, now)
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

# Place the callbacks in a list
callbacks = [early_stopping, tensorboard]

# Train the model
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)

#-------- Testing on data ---
# Make a prediction on the test set
test_predictions = model.predict(x_test)
test_predictions = np.round(test_predictions)

# Report the accuracy
accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy: " + str(accuracy))

import matplotlib.pyplot as plt
def visualize_incorrect_labels(x_data, y_real, y_predicted):
    # INPUTS
    # x_data      - images
    # y_data      - ground truth labels
    # y_predicted - predicted label
    count = 0
    figure = plt.figure()
    incorrect_label_indices = (y_real != y_predicted)
    y_real = y_real[incorrect_label_indices]
    y_predicted = y_predicted[incorrect_label_indices]
    x_data = x_data[incorrect_label_indices, :, :, :]

    maximum_square = np.ceil(np.sqrt(x_data.shape[0]))

    for i in range(x_data.shape[0]):
        count += 1
        figure.add_subplot(maximum_square, maximum_square, count)
        plt.imshow(x_data[i, :, :, :])
        plt.axis('off')
        plt.title("Predicted: " + str(int(y_predicted[i])) + ", Real: " + str(int(y_real[i])), fontsize=10)

    plt.show()

visualize_incorrect_labels(x_test, y_test, np.asarray(test_predictions).ravel())
