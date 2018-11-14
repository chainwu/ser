#!/usr/bin/env python3

#import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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


plt.axis([-15, 15, -15, 15], 'off')

x = [10, -10, 0, 0, 0]
y = [ 0,   0, 0, 10, -10]

plt.plot(x, y, 'o', color='1', markeredgecolor='r', markersize=20)
plot_label()

#circle1 = plt.Circle((0, 0), 1, color='r')
#circle2 = plt.Circle((0, 10), 1, color='blue')
#circle3 = plt.Circle((10, 0), 1, color='g', clip_on=False)
#circle4 = plt.Circle((-10, 0), 1, color='g', clip_on=False)
#circle5 = plt.Circle((0, -10), 1, color='g', clip_on=False)
plt.show()
plt.pause(5)
