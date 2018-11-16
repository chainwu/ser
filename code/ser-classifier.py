#!/usr/bin/env python3

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model

SER_MODEL = 'ser-1.h5'

def plot_label():
    plt.text(0, -3, 'Neutral', size=10, ha="center", va="center",
    	     bbox=dict(boxstyle="round",
        	       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),))
    plt.text(0, 7, 'Anger', size=10, ha="center", va="center",
    	     bbox=dict(boxstyle="round",
        	       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),))
    plt.text(10, -3, 'Happiness', size=10, ha="center", va="center",
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


def plot_baseplane():
    plt.axis([-15, 15, -15, 15], 'off')

    x = [10, -10, 0, 0, 0]
    y = [ 0,   0, 0, 10, -10]

    plt.plot(x, y, 'o', color='1', markeredgecolor='r', markersize=20)
    plot_label()

    plt.show()

def load_cnn():
    model = load_model(SER_MODEL)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def read_input():

def cnn_classify():
    model.predict()
    
def main():
    try:
        plot_baseplane()
        plot_label()
        load_cnn()
        read_input()
        cnn_classify()
        
    except:
        print("Exception!")

if __name__ == '__main__':
    main()
