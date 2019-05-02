'''
'''

import tensorflow as tf
import numpy as np
from sklearn.datasets import make_blobs
from sklearn import preprocessing
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dagmm import DAGMM

# Build toy dataset
data, _ = make_blobs(n_samples=2000, n_features=2, centers=5, random_state=123)
data = preprocessing.scale(data)
data[300] = [0, 2]
data[500] = [-2, -2]

plt.scatter(data[:,0], data[:,1], marker='.')
plt.savefig('images/data.png')
plt.close()

# Build model
model = DAGMM(2)
model.fit(data, 128, 1000, 100)

pred = model.predict(data)
plt.plot(pred, 'o-')
plt.plot([300,500], [pred[300], pred[500]], 'ro-')
plt.savefig('images/pred.png')
plt.close()

# Draw the decision boundary
xx,yy = np.meshgrid(np.arange(-2,2,0.01), np.arange(-2,2,0.01))
data = np.c_[xx.ravel(), yy.ravel()]
pred = model.predict(data).reshape(xx.shape)

mean = np.mean(pred)
pred[pred>mean*3] = mean*3

plt.contourf(xx, yy, pred, 10, alpha = 0.6, cmap = plt.cm.hot)
C = plt.contour(xx, yy, pred, 10, colors = 'black')
plt.clabel(C, inline = True, fontsize = 10)
plt.savefig('images/boundary.png')

