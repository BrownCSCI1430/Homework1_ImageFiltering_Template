# CSCI 1430
# Homework 1 Image Filtering
#
import numpy as np
from scipy.signal import convolve2d

def my_imfilter(image, filter, pad_mode='constant'):
    """
    Apply a filter to an image. Return the filtered image.

    Inputs:
        image: numpy nd-array of dim (m, n) or (m, n, c)
        filter: numpy nd-array of dim (k, k)
        pad_mode: padding mode for np.pad. Options include:
            'constant' - pad with zeros
            'edge'     - pad with edge values
            'reflect'  - reflect values at boundary (default)
            'symmetric'- reflect including edge values
            'wrap'     - wrap around to opposite edge

    Returns:
        filtered_image: numpy nd-array of same size as input

    Raises:
        Exception if filter has any even dimension.
    """
    filtered_image = np.zeros_like(image)

    return filtered_image


"""
EXTRA CREDIT: FFT-based filtering
"""
def my_imfilter_fft(image, filter, pad_mode='constant'):
    """
    Apply a filter to an image using FFT. Return the filtered image.

    Inputs:
        image: numpy nd-array of dim (m, n) or (m, n, c)
        filter: numpy nd-array of dim (k, k)
        pad_mode: padding mode for np.pad. Options include:
            'constant' - pad with zeros (default)
            'edge'     - pad with edge values
            'reflect'  - reflect values at boundary
            'symmetric'- reflect including edge values
            'wrap'     - wrap around to opposite edge

    Returns:
        filtered_image: numpy nd-array of same size as input

    Raises:
        Exception if filter has any even dimension.
    """
    filtered_image = np.zeros_like(image)

    return filtered_image


def fit_kernel_gd(I, T, k, lr=0.01, num_iters=5000):
    """
    Gradient descent kernel fit: find K (k x k) s.t. my_imfilter(I, K) ≈ T.
    Uses MSE loss with your hand-derived gradients from the written questions.

    TIP 0: The target image was generated with zero padding and a 3 x 3 kernel.

    TIP 1: This is going to perform _a lot_ of convolution.
           Unless your my_imfilter function is very fast,
           use scipy.signal.convolve2d instead with mode='same'.

    TIP 2: A good initialization always helps gradient descent.
           Our target kernel has values between [-200,200]

    I: input image (H, W), grayscale float
    T: target output image (H, W)
    k: odd kernel size
    lr: learning rate
    num_iters: number of gradient descent iterations

    Returns:
      K: (k, k) kernel
    """
    K = np.zeros((k, k))

    return K


def fit_kernel_ls(I, T, k):
    """
    Least-squares kernel fit: find K (k x k) s.t. my_imfilter(I, K) ≈ T.

    TIP 0: The target image was generated with zero padding and a 3 x 3 kernel.

    I: input image (H, W), grayscale float
    T: target output image (H, W)
    k: odd kernel size

    Returns:
      K: (k, k) kernel
    """
    K = np.zeros((k, k))

    return K


def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
    Inputs:
    - image1 -> The image from which to take the low frequencies.
    - image2 -> The image from which to take the high frequencies.
    - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                          blur that will remove high frequencies.

    Task:
    - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
    - Combine them to create 'hybrid_image'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it.
    #     The amount of blur that works best will vary per image pair

    # Generate a 1x(2k+1) gaussian kernel with mean=0 and sigma=cutoff_frequency


    # (2) Remove the low frequencies from image2.
    #     Subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.

    # (3) Combine the high frequencies and low frequencies

    # (4) Clip the image for display purpose
    #     (not doing this could cause value errors)

    low_frequencies = np.zeros_like(image1)
    high_frequencies = np.zeros_like(image1)
    hybrid_image = np.zeros_like(image1)

    return low_frequencies, high_frequencies, hybrid_image
