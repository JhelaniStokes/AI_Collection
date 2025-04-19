import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

mnist = sio.loadmat(r'C:\Data\mnist-original.mat')





data = mnist['data']
labels = mnist['label']


'''
indices_0 = np.where(labels == 0)[1][:1000]
indices_1 = np.where(labels == 1)[1][:1000]

data0 = data[:, indices_0]
data1 = data[:, indices_1]

xPrime = np.concatenate((data0, data1), axis=1)
'''


'''
mean = xPrime.mean(axis=0)

X = xPrime - mean

Xcov = np.dot(X, X.T)

pcaX = PCA()

pcaX.fit(Xcov)
values = pcaX.explained_variance_
vectors = pcaX.components_




'''
'''
fig, axes = plt.subplots(5, 6, figsize=(12, 10))


for i, ax in enumerate(axes.flatten()):
    image = vectors[i].reshape((28, 28))
    ax.imshow(image, cmap='gray')
    ax.set_title(f"{i+1}")
    ax.axis('off')
plt.tight_layout()
plt.show()
'''
'''
k= [1, 3, 10, 30, 100, 500, 784]
x0 = xPrime[:, 500]
x1 = xPrime[:, 1500]

fig, axes = plt.subplots(2, 7, figsize=(12, 10))
col = 0
for x in k:

    w0 = np.dot(vectors[:x, :], x0)
    re0 = np.dot(w0, vectors[:x, :])
    re0I = re0.reshape((28, 28))
    axes[0, col].imshow(re0I, cmap = "gray")
    axes[0, col].set_title(f"{x}")
    axes[0, col].text(0.5, 40, f"error: {round(np.linalg.norm(x0-re0), 5)}", fontsize=10)
    col += 1

col = 0
for x in k:

    w1 = np.dot(vectors[:x, :], x1)
    re1 = np.dot(w1, vectors[:x, :])
    re1I = re1.reshape((28, 28))
    axes[1, col].imshow(re1I, cmap="gray")
    axes[1, col].set_title(f"{x}")
    axes[1, col].text(0.5, 40, f"error: {round(np.linalg.norm(x1-re1), 5)}", fontsize=10)
    col += 1
plt.show()
'''

sample_indices = [449, 257, 1828, 1149, 629, 1691, 1476, 1715, 1439, 1993]
a = [700, 825, 261, 634, 879, 757, 3, 307, 623, 797, 34, 682, 55, 113, 657, 603, 91, 654,
 114, 773, 545, 403, 633, 194, 529, 714, 474, 701, 441, 177, 314, 999, 76, 395, 910, 61,
 840, 543, 676, 531, 13, 41, 512, 476, 420, 627, 574, 86, 906, 860, 120, 738, 38, 130,
 308, 940, 721, 844, 358, 671, 455, 866, 921, 926, 598, 202, 581, 453, 24, 755, 904, 673,
 272, 245, 284, 900, 857, 792, 315, 991, 65, 927, 17, 77, 650, 357, 482, 388, 609, 193,
 326, 518, 605, 961, 303, 817, 376, 632, 365, 928]
b = [1700, 1825, 1261, 1634, 1879, 1757, 1003, 1307, 1623, 1797, 1034, 1682, 1055, 1113,
     1657, 1603, 1091, 1654, 1114, 1773, 1545, 1403, 1633, 1194, 1529, 1714, 1474, 1701,
     1441, 1177, 1314, 1999, 1076, 1395, 1910, 1061, 1840, 1543, 1676, 1531, 1013, 1041,
     1512, 1476, 1420, 1627, 1574, 1086, 1906, 1860, 1120, 1738, 1038, 1130, 1308, 1940,
     1721, 1844, 1358, 1671, 1455, 1866, 1921, 1926, 1598, 1202, 1581, 1453, 1024, 1755,
     1904, 1673, 1272, 1245, 1284, 1900, 1857, 1792, 1315, 1991, 1065, 1927, 1017, 1077,
     1650, 1357, 1482, 1388, 1609, 1193, 1326, 1518, 1605, 1961, 1303, 1817, 1376, 1632,
     1365, 1928]

wA = np.dot(vectors, X[:, a].mean(axis = 1))
wB = np.dot(vectors, X[:, b].mean(axis = 1))

cfied = []

sample = np.zeros((784, 10))
sampleP = X[:, sample_indices]
for i in range(10):
    sample[:, i] = np.dot(vectors, sampleP[:, i])


for i in range(10):
    if np.linalg.norm(sample[:, i]-wA)<np.linalg.norm(sample[:, i]-wB):
        cfied.append(0)
    else:
        cfied.append(1)

print(cfied)


#plt.show()






