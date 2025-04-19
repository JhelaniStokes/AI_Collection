import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

mnist = sio.loadmat(r'C:\Data\mnist-original.mat')





data = mnist['data']
labels = mnist['label']



indices_0 = np.where(labels == 0)[1][:1000]
indices_1 = np.where(labels == 1)[1][:1000]

data0 = data[:, indices_0]
data1 = data[:, indices_1]

xPrime = np.concatenate((data0, data1), axis=1)



mean = xPrime.mean(axis=0)

X = xPrime - mean

Xcov = np.dot(X, X.T)

pcaX = PCA()

pcaX.fit(Xcov)
values = pcaX.explained_variance_
vectors = pcaX.components_



fig, axes = plt.subplots(2, 5, figsize=(12, 10))

for i, ax in enumerate(axes.flatten()):
    image = vectors0[i].reshape((28, 28))
    ax.imshow(image, cmap='gray')
    ax.set_title(f"{i+1}")
    ax.axis('off')
plt.tight_layout()
plt.show()
