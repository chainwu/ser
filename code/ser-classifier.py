#!/usr/bin/env python3

#import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.axis([0, 10, 0, 1], 'off')

circle1 = plt.Circle((0, 0), 1, color='r')
circle2 = plt.Circle((0, 10), 1, color='blue')
circle3 = plt.Circle((10, 0), 1, color='g', clip_on=False)
circle4 = plt.Circle((-10, 0), 1, color='g', clip_on=False)
circle5 = plt.Circle((0, -10), 1, color='g', clip_on=False)
plt.show()
plt.pause(5)
