# Homework 1 Image Filtering Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io
from math import floor

def my_imfilter(image, filter):
  """
  Your function should meet the requirements laid out on the homework webpage.
  Apply a filter to an image. Return the filtered image.
  Inputs
  - image: numpy nd-array of dim (m,n) or (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
  Errors if:
  - filter has any even dimension -> raise an Exception with a suitable error message.
  """
  filtered_image = np.zeros(image.shape)

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1
  if len(filter.shape) > 2:
    print( 'Filter must be 2D; removing other dimensions from filter.')
    filter = np.squeeze(filter[:,:,0])

  numRow = image.shape[0]
  numCol = image.shape[1]
  numChannel = 1
  if len(image.shape) > 2:
    numChannel = image.shape[2]
  else:
    # For convenience, turn image into a 3D array with a singleton last dimension
    image = np.expand_dims(image,axis=2)

  ## Check if kernel is an array, and turn it into a 2D matrix
  if len(filter.shape) == 1:
    print('Turning array kernel into 2D matrix; assuming column vector')
    filter = np.expand_dims(filter,axis=1)

  # Implementing convolution, so rotate the filter 180deg
  filter = np.rot90( filter, 2 )

  #
  fRowHalf = floor(filter.shape[0] / 2.0)
  fColHalf = floor(filter.shape[1] / 2.0)

  # Pad input image
  # Change to mode='reflect' for reflection padding
  pimage = np.pad( image, ((fRowHalf,fRowHalf), (fColHalf,fColHalf), (0,0)), mode='reflect')

  filtered_image = np.zeros(pimage.shape, dtype=pimage.dtype)

  for k in range(numChannel):
    for i in range(fRowHalf,fRowHalf+numRow): #iterate through rows
      for j in range(fColHalf,fColHalf+numCol): #iterate through cols
        # Collect image part
        ip = pimage[i-fRowHalf:i+fRowHalf+1,j-fColHalf:j+fColHalf+1,k]
        # Write to filtered image
        filtered_image[i,j,k] = np.sum(ip * filter)

  # Crop filtered image back to original size
  filtered_image = filtered_image[fRowHalf:fRowHalf+numRow,fColHalf:fColHalf+numCol]
  if numChannel == 1:
    filtered_image = np.squeeze(filtered_image)

  ### END OF STUDENT CODE ####
  ############################

  return filtered_image


"""
EXTRA CREDIT placeholder function
"""
def my_imfilter_fft(image, filter):
  """
  Your function should meet the requirements laid out on the homework webpage.
  Apply a filter to an image. Return the filtered image.
  Inputs
  - image: numpy nd-array of dim (m,n) or (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
  Errors if:
  - filter has any even dimension -> raise an Exception with a suitable error message.
  """
  filtered_image = np.zeros(image.shape)

  (a,b) = filter.shape 
  (m,n,c) = image.shape 
  if (a%2 == 0 or b%2 == 0): 
    raise Exception("Filter dimensions must be odd.")
  row_edge = int((a-1)/2) #pad length on row and column
  col_edge = int((b-1)/2)
  padded_image = np.zeros((m+a-1,n+b-1,c))
  padded_image[row_edge:row_edge+m,col_edge:col_edge+n,:] = image
  padded_filter = np.zeros((m+a-1,n+b-1))
  padded_filter[0:a,0:b] = filter
  filtered_image = np.zeros(padded_image.shape) 
  for i in range(c): 
    fft_image = np.fft.ifft2(np.fft.fft2(padded_image[:,:,i])*np.fft.fft2(padded_filter))
    filtered_image[:,:,i] = fft_image
  filtered_image = filtered_image[2*row_edge:,2*col_edge:,:]

  ### END OF STUDENT CODE ####
  ############################

  return filtered_image


def fit_kernel_ls(I_path, T, k):
    """
    Least-squares kernel fit: find K (k x k) s.t. my_imfilter(I, K) ≈ T.

    I_path: path to input image
    T: target output image (H,W)
    k: odd kernel size

    Returns:
      K: (k,k) kernel
    """
    image = (io.imread(I_path, as_gray=True)).astype(np.float32)
    H, W = image.shape
    p = k // 2

    # reflect-pad input
    Ip = np.pad(image, ((p, p), (p, p)), mode="symmetric")

    # Build A and b: each row is a flattened patch, b is corresponding target pixel
    A = np.zeros((H * W, k * k), dtype=np.float32)
    b = np.zeros((H * W,), dtype=np.float32)

    r = 0
    for i in range(H):
        for j in range(W):
            patch = Ip[i : i + k, j : j + k]
            A[r] = patch.reshape(-1)
            b[r] = T[i, j]
            r += 1


    x, *_ = np.linalg.lstsq(A, b, rcond=None)

    # my_imfilter rotates kernel by 180° internally (convolution),
    # but our A uses the patch "as-is". So we return the rotated kernel.
    K = x.reshape(k, k)
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
  # (1) Remove the high frequencies from image1 by blurring it. The amount of
  #     blur that works best will vary with different image pairs
  # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
  s, k = cutoff_frequency, cutoff_frequency*2
  probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
  kernel = np.outer(probs, probs)
  
  # Your code here:
  low_frequencies = my_imfilter(image1, kernel)

  # (2) Remove the low frequencies from image2. The easiest way to do this is to
  #     subtract a blurred version of image2 from the original version of image2.
  #     This will give you an image centered at zero with negative values.
  low_frequencies_image2 = my_imfilter(image2, kernel)
  temp_high_freq = image2 - low_frequencies_image2
  high_frequencies = np.clip(temp_high_freq, -0.5, 0.5)

  # (3) Combine the high frequencies and low frequencies
  hybrid_image = low_frequencies + temp_high_freq
  
  # Clipping the image for display purpose (removing this line could cause value error)
  hybrid_image = np.clip(hybrid_image, 0.0, 1.0)

  return low_frequencies, high_frequencies, hybrid_image
