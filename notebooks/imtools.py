import matplotlib.pyplot as plt
import numpy as np
import cv2


def load_image(path, size=(256, 256)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def plot_images(images, scale=1.0):
    """
    Plots a list of images in a single row with axes turned off.

    Args:
        images (list): List of images to plot.
        scale (float): Scale factor for the image size.
    """
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(scale * num_images, scale))
    if num_images == 1:
        axes = [axes]  # Ensure axes is iterable for a single image
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_grid(images, scale=1):
    """
    Plot a list of images in a square grid.
    
    Args:
        images: List/array of images
        scale: Scale factor for figure size (default=1)
    """
    N = len(images)
    rows = int(np.sqrt(N))
    cols = int(np.ceil(N / rows))
    
    fig = plt.figure(figsize=(cols * scale, rows * scale))
    
    for i, img in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()