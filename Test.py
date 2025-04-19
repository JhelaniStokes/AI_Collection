import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

mnist = sio.loadmat(r'C:\Data\mnist-original.mat')




# 2. Extract the image data and labels.
# Adjust the keys 'data' and 'label' to match those in your .mat file.
data = mnist['data']
labels = mnist['label']



indices_0 = np.where(labels == 0)[1][:1000]
indices_1 = np.where(labels == 1)[1][:1000]

data0 = data[:, indices_0]
data1 = data[:, indices_1]

xPrime = np.concatenate((data0, data1), axis=1)



mean = xPrime.mean(axis=0)

X = xPrime - xPrime.mean(axis=0)

data0cov = np.dot(data0, data0.T)

pca0 = PCA()

pca0.fit(data0cov)
values0 = pca0.explained_variance_
vectors0 = pca0.components_

sample_indices = [449, 257, 1828, 1149, 629, 1691, 1476, 1715, 1439, 1993]
sample = X[:, sample_indices]

fig, axes = plt.subplots(2, 5, figsize=(12, 10))

for i, ax in enumerate(axes.flatten()):
    image = sample[:,i].reshape((28, 28))
    ax.imshow(image, cmap='gray')
    ax.set_title(f"{i+1}")
    ax.axis('off')
plt.tight_layout()
plt.show()