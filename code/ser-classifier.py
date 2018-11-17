#!/usr/bin/env python3

import keras
import matplotlib
import imageio
import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model

SER_MODEL = 'ser-2.h5'
images=[]
test_predictions = None

def cnn_classify():
    print("----")
    file = input("Please enter the filename: ")
    images = [[imageio.imread(file)]]

    model = load_model(SER_MODEL)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    test_predictions = model.predict(images)
    #print("classify:",test_predictions)
    
    plt.text(10, -3, 'Anger', size=10, ha="center", va="center",
    	     bbox=dict(boxstyle="round",
        	       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),))
    plt.text(0, 7, 'Happiness', size=10, ha="center", va="center",
    	     bbox=dict(boxstyle="round",
        	       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),))
    plt.text(0, -3, 'Neutral', size=10, ha="center", va="center",
    	     bbox=dict(boxstyle="round",
        	       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),))
    plt.text(-10, -3, 'Sarcasm', size=10, ha="center", va="center",
    	     bbox=dict(boxstyle="round",
        	       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),))
    plt.text(0, -13, 'Surprise', size=10, ha="center", va="center",
    	     bbox=dict(boxstyle="round",
        	       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),))

    plt.axis([-15, 15, -15, 15], 'off')

    x = [10, 0,  0, -10, 0]
    y = [ 0, 10, 0,   0, -10]
    c = test_predictions[0]
    
    for i in range(0, 5):
        col = abs(1-c[i])
        #print(x[i], y[i], col)
        plt.plot(x[i], y[i], '.', color=(col,col,col), markeredgecolor='r', markeredgewidth=1, markersize=20)
        
    plt.show()
    return

def main():
    try:
        cnn_classify()
        
    except Exception as e:
        print("Exception! ", e)

if __name__ == '__main__':
    main()
