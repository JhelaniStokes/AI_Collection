import numpy as np
import tkinter as tk
from PIL import Image, ImageGrab, ImageOps
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import center_of_mass
import scipy.io as sio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = np.load('pca_vectors.npz')
vectors = data['vectors']
data= np.load('weights.npz')
weights = data['weights']
def center_digit(imgArray):
    """Centers the digit in the image by shifting its center of mass to the middle."""
    cy, cx = center_of_mass(imgArray)
    shift_x = imgArray.shape[1] // 2 - int(cx)
    shift_y = imgArray.shape[0] // 2 - int(cy)
    return np.roll(imgArray, shift=(shift_y, shift_x), axis=(0, 1))

def clear():
    canvas.delete("all")


def submit(weights=weights, vectors = vectors):
    """Converts the canvas drawing to a 28x28 NumPy array."""
    # Get the canvas position within the window
    x = window.winfo_rootx() + canvas.winfo_x()
    y = window.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # Capture the canvas content as an image
    img = ImageGrab.grab(bbox=(x, y, x1, y1))
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))

    # Convert to NumPy array
    img_array = np.array(img)
    img_array = center_digit(img_array)
    # plt.imshow(img_array, cmap="gray")
    # plt.show()

    print("Max pixel value:", np.max(img_array))
    print("Min pixel value:", np.min(img_array))
    print("Mean pixel value:", np.mean(img_array))

    imageVector = img_array.flatten()
    imageVector = (imageVector - np.mean(imageVector)) / np.std(imageVector)
    imageProjected = np.dot(vectors, imageVector)

    imageProjected = (imageProjected - np.mean(imageProjected)) / np.std(imageProjected)
    weights = (weights - np.mean(weights, axis=0)) / np.std(weights, axis=0)

    #print(imageProjected[:10])
    # distances = np.linalg.norm(weights - imageProjected[:, None], axis=0)
    # cov_matrix = np.cov(weights)
    # cov_inv = np.linalg.pinv(cov_matrix)
    # distances = np.array([
    #     np.dot((imageProjected - weights[:, i]).T, cov_inv @ (imageProjected - weights[:, i]))
    #     for i in range(10)
    # ])
    cos_sim = cosine_similarity(imageProjected.reshape(1, -1), weights.T)
    predicted_class = np.argmax(cos_sim)

    print(predicted_class)



def start_draw(event):
    """Records the initial point of drawing."""
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw(event):
    """Draws a line from the last point to the new point as the mouse moves."""
    global last_x, last_y
    canvas.create_line(last_x, last_y, event.x, event.y, fill="black", width=20, capstyle=tk.ROUND, smooth=True)
    last_x, last_y = event.x, event.y  # Update last position







window = tk.Tk()
window.geometry('600x450')
window.title('canvas')
window.configure(bg="lightgray")

window.grid_rowconfigure(0, minsize=360)
window.grid_rowconfigure(1, weight=1)

window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)

canvas = tk.Canvas(window, width = 550, height = 350, bg = 'white')
canvas.grid(row=0, column=0, columnspan=2, sticky='nsew')
canvas.bind("<ButtonPress-1>", start_draw)  # Mouse click
canvas.bind("<B1-Motion>", draw)

clearButton = tk.Button(window, text='clear', command = clear).grid(row=1, column=0, sticky="ew", padx=20, pady=10)

submitButton = tk.Button(window, text='submit', command = submit).grid(row=1, column=1, sticky="ew", padx=20, pady=10)




window.mainloop()