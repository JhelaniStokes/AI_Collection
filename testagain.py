import numpy as np
import tkinter as tk
from PIL import Image, ImageGrab, ImageOps
import scipy.io as sio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = np.load('pca_vectors.npz')
vectors = data['vectors']
data= np.load('weights.npz')
weights = data['weights']
data = np.load('data_demeaned.npz')
X = data['X']

imageVector = X[:, 8437]
img_array = np.reshape(imageVector, (28,28))
# plt.imshow(img_array, cmap="gray")
# plt.show()
imageProjected = np.dot(vectors, imageVector)
#distances = np.linalg.norm(weights-imageProjected[:, None], axis = 0)
cov_matrix = np.cov(weights)
cov_inv = np.linalg.pinv(cov_matrix)
distances = np.array([
    np.dot((imageProjected - weights[:, i]).T, cov_inv @ (imageProjected - weights[:, i]))
    for i in range(10)
])
print(np.argmin(distances))
