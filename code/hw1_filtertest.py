# CSCI 1430
# Homework 1 Image Filtering
#
import os
import numpy as np
import matplotlib.pyplot as plt
from helpers import load_image, save_image
from student import my_imfilter

"""
This function loads an image, and then attempts to filter that image
using different kernels as a testing routine.
"""
def filter_test(img_path):
    results_dir = '..' + os.sep + 'results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # =========================================================================
    # LOAD IMAGE
    # =========================================================================
    test_image = load_image(img_path)

    # =========================================================================
    # APPLY FILTERS
    # =========================================================================
    # Identity filter - does nothing regardless of padding method
    identity_filter = np.asarray(
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    identity_image = my_imfilter(test_image, identity_filter)

    # Small box blur filter (5x5) - removes some detail
    blur_filter = np.ones((5, 5), dtype=np.float32)
    blur_filter /= np.sum(blur_filter, dtype=np.float32)
    blur_image = my_imfilter(test_image, blur_filter)

    # Large Gaussian blur (separable) - 1D kernel applied as column then row
    # Removes lots of detail
    s, k = 4, 12
    # Column vector of Gaussian PDF values
    gauss_1d = np.array([[np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s)] for z in range(-k, k+1)], dtype=np.float32)
    large_blur_image = my_imfilter(test_image, gauss_1d)    # vertical
    large_blur_image = my_imfilter(large_blur_image, gauss_1d.T)  # horizontal

    # Sobel filter - responds to horizontal changes
    sobel_filter = np.asarray(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_image = my_imfilter(test_image, sobel_filter)
    sobel_image = np.clip(sobel_image + 0.5, 0.0, 1.0)  # center around 0.5

    # Laplacian filter - responds to blobs
    laplacian_filter = np.asarray(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    laplacian_image = my_imfilter(test_image, laplacian_filter)
    laplacian_image = np.clip(laplacian_image + 0.5, 0.0, 1.0)  # center around 0.5

    # Original minus blur - lets through detail or 'high frequencies' -> high pass
    high_pass_image = test_image - blur_image
    high_pass_image = np.clip(high_pass_image + 0.5, 0.0, 1.0)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    save_image(results_dir + os.sep + 'identity_image.png', identity_image)
    save_image(results_dir + os.sep + 'blur_image.png', blur_image)
    save_image(results_dir + os.sep + 'large_blur_image.png', large_blur_image)
    save_image(results_dir + os.sep + 'sobel_image.png', sobel_image)
    save_image(results_dir + os.sep + 'laplacian_image.png', laplacian_image)
    save_image(results_dir + os.sep + 'high_pass_image.png', high_pass_image)

    # =========================================================================
    # DISPLAY ALL RESULTS
    # =========================================================================
    results = [
        ('Input', test_image),
        ('Identity', identity_image),
        ('Box Blur (5x5)', blur_image),
        ('Gaussian Blur (25x25)', large_blur_image),
        ('Sobel (edges)', sobel_image),
        ('Laplacian', laplacian_image),
        ('Original - Box Blur', high_pass_image),
    ]

    _, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for ax, (title, img) in zip(axes, results):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    # Hide the unused subplot (8th cell)
    axes[-1].axis('off')

    plt.tight_layout()
    plt.show()
