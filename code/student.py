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
    if filter.shape[0] % 2 == 0 or filter.shape[1] % 2 == 0:
        raise Exception('Filter dimensions must be odd.')
    if len(filter.shape) > 2:
        print('Filter must be 2D not 3D; removing other dimensions from filter.')
        filter = np.squeeze(filter[:, :, 0])

    num_rows = image.shape[0]
    num_cols = image.shape[1]
    num_channels = 1
    if len(image.shape) > 2:
        num_channels = image.shape[2]
    else:
        # For convenience, turn image into a 3D array with a singleton last dimension
        image = np.expand_dims(image, axis=2)

    # Check if kernel is an array, and turn it into a 2D matrix
    if len(filter.shape) == 1:
        print('Turning array kernel into 2D matrix; assuming column vector')
        filter = np.expand_dims(filter, axis=1)

    # Implementing convolution, so rotate the filter 180deg
    filter = np.rot90(filter, 2)

    pad_rows = filter.shape[0] // 2
    pad_cols = filter.shape[1] // 2

    # Pad input image
    padded_image = np.pad(image, ((pad_rows, pad_rows), (pad_cols, pad_cols), (0, 0)), mode=pad_mode)

    filtered_image = np.zeros(padded_image.shape, dtype=padded_image.dtype)

    for c in range(num_channels):
        for i in range(pad_rows, pad_rows + num_rows):
            for j in range(pad_cols, pad_cols + num_cols):
                patch = padded_image[i - pad_rows:i + pad_rows + 1, j - pad_cols:j + pad_cols + 1, c]
                filtered_image[i, j, c] = np.sum(patch * filter)

    # Crop filtered image back to original size
    filtered_image = filtered_image[pad_rows:pad_rows + num_rows, pad_cols:pad_cols + num_cols]
    if num_channels == 1:
        filtered_image = np.squeeze(filtered_image)

    ### END OF STUDENT CODE ####
    ############################

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
    filter_rows, filter_cols = filter.shape
    num_channels = image.shape[2]
    if filter_rows % 2 == 0 or filter_cols % 2 == 0:
        raise Exception('Filter dimensions must be odd.')

    pad_rows = (filter_rows - 1) // 2
    pad_cols = (filter_cols - 1) // 2

    # Pad input image
    padded_image = np.pad(image, ((pad_rows, pad_rows), (pad_cols, pad_cols), (0, 0)), mode=pad_mode)

    # Pad filter to match padded image size
    padded_filter = np.zeros((padded_image.shape[0], padded_image.shape[1]))
    padded_filter[0:filter_rows, 0:filter_cols] = filter

    filtered_image = np.zeros(padded_image.shape)
    for c in range(num_channels):
        fft_result = np.fft.ifft2(np.fft.fft2(padded_image[:, :, c]) * np.fft.fft2(padded_filter))
        filtered_image[:, :, c] = fft_result.real

    # Crop back to original size
    filtered_image = filtered_image[2 * pad_rows:, 2 * pad_cols:, :]

    ### END OF STUDENT CODE ####
    ############################

    return filtered_image


def fit_kernel_gd(I, T, k, lr=0.01, num_iters=10000):
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
    H, W = I.shape
    p = k // 2
    N = H * W

    # Pad input image (same padding as my_imfilter uses internally)
    I_padded = np.pad(I, ((p, p), (p, p)), mode='constant')

    # Initialize kernel randomly
    rng = np.random.default_rng()
    K = rng.standard_normal((k, k)).astype(np.float32) * 10

    for iteration in range(num_iters):
        # Forward pass: T_pred = conv(I, K)
        T_pred = convolve2d(I, K, mode='same', boundary='fill')

        # Compute error and MSE loss
        error = T_pred - T
        loss = np.mean(error ** 2)

        # Compute gradient dL/dK
        # Since my_imfilter rotates K by 180° internally (convolution),
        # we first compute gradient w.r.t. the rotated kernel, then rotate back.
        #
        # dL/dK_rot[m,n] = (2/N) * sum_{i,j} error[i,j] * I_padded[i+m, j+n]
        grad_K_rot = np.zeros((k, k), dtype=np.float32)
        for m in range(k):
            for n in range(k):
                grad_K_rot[m, n] = np.sum(error * I_padded[m:m + H, n:n + W])
        grad_K_rot *= (2.0 / N)

        # Rotate gradient back to get dL/dK
        grad_K = np.rot90(grad_K_rot, 2)

        # Gradient descent update
        K = K - lr * grad_K

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss:.6f}")

    ### END OF STUDENT CODE ####
    ############################

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
    H, W = I.shape
    p = k // 2

    # Zero-pad input
    Ip = np.pad(I, ((p, p), (p, p)), mode='constant')

    # Build A and b: each row is a flattened patch, b is corresponding target pixel
    A = np.zeros((H * W, k * k), dtype=np.float32)
    b = np.zeros((H * W,), dtype=np.float32)

    r = 0
    for i in range(H):
        for j in range(W):
            patch = Ip[i:i + k, j:j + k]
            A[r] = patch.reshape(-1)
            b[r] = T[i, j]
            r += 1

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    K = x.reshape(k, k)

    # my_imfilter rotates kernel by 180° internally (convolution),
    # but our A uses the patch "as-is". So we return the rotated kernel.
    K = np.rot90(K, 2)

    ### END OF STUDENT CODE ####
    ############################

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

    # Generate a 1x(2k+1) gaussian kernel with mean=0 and sigma=s
    s, k = cutoff_frequency, cutoff_frequency * 2
    probs = np.asarray(
        [np.exp(-z * z / (2 * s * s)) / np.sqrt(2 * np.pi * s * s) for z in range(-k, k + 1)],
        dtype=np.float32
    )
    kernel = np.outer(probs, probs)

    low_frequencies = my_imfilter(image1, kernel)

    # (2) Remove the low frequencies from image2.
    #     Subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    low_frequencies_image2 = my_imfilter(image2, kernel)
    temp_high_freq = image2 - low_frequencies_image2
    high_frequencies = np.clip(temp_high_freq, -0.5, 0.5)

    # (3) Combine the high frequencies and low frequencies
    hybrid_image = low_frequencies + temp_high_freq

    # Clipping the image for display purpose (removing this line could cause value error)
    hybrid_image = np.clip(hybrid_image, 0.0, 1.0)

    return low_frequencies, high_frequencies, hybrid_image
