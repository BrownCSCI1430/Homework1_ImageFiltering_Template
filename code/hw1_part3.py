"""
Part 3: Kernel inverse problem (edge filter recovery).
These utilities generate a target edge image from a known kernel and recover the
kernel with least squares or hand-rolled gradient descent.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import io
from student import fit_kernel_ls

SOBEL_X = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    dtype=np.float32,
)


def create_target_image(image_path, kernel=SOBEL_X):
    """
    Build a target edge image by filtering with a known kernel.
    This mirrors the padding/boundary behavior used in kernel recovery so that
    the recovered filter matches the generator.
    """
    image = (io.imread(image_path, as_gray=True)).astype(np.float32)
    edge_image = convolve2d(image, kernel, mode="same", boundary="symm")
    return edge_image

def recover_kernel(image_path, k=3, ridge=0.0, display=True):
    """
    Convenience wrapper used by the CLI to generate a target edge map, recover
    the kernel with least squares, and visualize the reconstruction error.
    """
    target = create_target_image(image_path)
    K = fit_kernel_ls(image_path, target, k)

    image = (io.imread(image_path, as_gray=True)).astype(np.float32)
    filtered = convolve2d(image, K, mode="same", boundary="symm")
    error = np.abs(target - filtered)

    # Print out the recovered kernel 
    print(K)

    if display:
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(target, cmap="gray")
        plt.title("target")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(filtered, cmap="gray")
        plt.title("filtered")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(error, cmap="gray")
        plt.title("error")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return K, filtered, target, error