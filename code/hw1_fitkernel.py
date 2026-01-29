# CSCI 1430
# Homework 1 Image Filtering
#
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.signal import convolve2d
from student import fit_kernel_gd, fit_kernel_ls

"""
Part 2: Fitting a kernel to a target image given an input image.

Possible 'mode' values:
- 'gd' -> gradient descent
- 'ls' -> least squares
"""
def recover_kernel(mode='gd'):
    
    original_image = (io.imread('../data/bicycle.bmp', as_gray=True)).astype(np.float32)
    # Load the target image as a numpy file, 
    # as kernel output is signed int16 and doesn't fit in a standard image file.
    target_image = np.load('../data/mysterykerneltarget_1x.npy')
    
    ksize = 3
    if mode == 'gd':
        K = fit_kernel_gd(original_image, target_image, ksize)
    elif mode == 'ls':
        K = fit_kernel_ls(original_image, target_image, ksize)
    else:
        K = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]  # Default identity matrix

    optimizedK_image = convolve2d(original_image, K, mode='same')
    error_image = np.abs(optimizedK_image - target_image)

    # Pretty print out the recovered kernel
    print(np.array2string(K, precision=4, suppress_small=True))

    # Visualize outputs
    results = [
        ('Original', original_image),
        ('Target', target_image),
        ('Filtered with K', optimizedK_image),
        ('Error', error_image),
    ]

    _, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, (title, img) in zip(axes, results):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    return K