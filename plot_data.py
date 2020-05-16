# plot the data and the line of regression
# for the special case of one-dimensional input

import matplotlib.pyplot as plt
import numpy as np
import random as rnd

from hypfunc import hypothesis

xt, yt = [], [] # t for training data

filename = "training_set.data"
with open(filename) as f:
    lines = f.readlines()

for line in lines:
    line = line.rstrip()
    x, y = line.split(" ")
    xt.append(float(x))
    yt.append(float(y))

xl, yl = [], [] # l for regression line
for i in range(100):
    xin = rnd.random() * (max(xt) - min(xt)) + min(xt)
    xl.append(xin)
    xin = [xin]
    yl.append(hypothesis(xin))

plt.style.use("seaborn")
plt.figure()
plt.plot(xt, yt, "o", xl, yl, "r")
plt.show()
