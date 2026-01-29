# CSCI 1430
# Homework 1 Image Filtering
#
import numpy as np
import skimage
import matplotlib.pyplot as plt
from helpers import load_image, save_image, equalize_image_sizes
from student import my_imfilter, gen_hybrid_image

"""
Part 3: Create a hybrid image!
"""
def vis_hybrid_image(hybrid_image):
    """
    Visualize a hybrid image by progressively downsampling the image and
    concatenating all of the images together.
    """
    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = 1 if hybrid_image.ndim == 2 else 3

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales + 1):
        # add padding
        output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                            dtype=np.float32)))
        # downsample image
        cur_image = skimage.transform.rescale(cur_image, scale_factor, mode='reflect', channel_axis=2)
        # pad the top to append to the output
        pad = np.ones((original_height - cur_image.shape[0], cur_image.shape[1],
                       num_colors), dtype=np.float32)
        tmp = np.vstack((pad, cur_image))
        output = np.hstack((output, tmp))
    return output

def hybrid_img_generation(img_one_path, img_two_path):
    # Setup
    # Read images and convert to floating point format
    image1 = load_image(img_one_path)
    image2 = load_image(img_two_path)

    image1, image2 = equalize_image_sizes(image1, image2)

    """
    For your write up, there are several additional test cases in 'data'.
    Feel free to make your own, too.
    For best results, you'll need to align the images in a photo editor.
    
    The hybrid images will differ depending on which image you
    assign as image1 (for the low frequencies) and which image
    you assign as image2 (for the high frequencies).
    """
    
    ## Hybrid Image Construction ##
    #
    # cutoff_frequency is the standard deviation, in pixels, of the Gaussian
    # blur that will remove high frequencies. You may tune this per image pair
    # to achieve better results.
    cutoff_frequency = 7
    low_frequencies, high_frequencies, hybrid_image = gen_hybrid_image(
        image1, image2, cutoff_frequency)

    ## Visualize outputs ##
    vis = vis_hybrid_image(hybrid_image)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])

    # First row: input images and frequency components
    row1 = [
        ('Image 1 (low freq source)', (image1 * 255).astype(np.uint8)),
        ('Image 2 (high freq source)', (image2 * 255).astype(np.uint8)),
        ('Low Frequencies', (low_frequencies * 255).astype(np.uint8)),
        ('High Frequencies', ((high_frequencies + 0.5) * 255).astype(np.uint8)),
    ]
    for i, (title, img) in enumerate(row1):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    # Second row: hybrid image at multiple scales (spans all columns)
    ax_vis = fig.add_subplot(gs[1, :])
    ax_vis.imshow(vis)
    ax_vis.set_title('Hybrid Image (multiple scales)')
    ax_vis.axis('off')

    plt.tight_layout()
    plt.show()

    ## Save outputs ##
    #
    save_image('../results/low_frequencies.png', low_frequencies)
    out_high = np.clip(high_frequencies + 0.5, 0.0, 1.0)
    save_image('../results/high_frequencies.png', out_high)
    save_image('../results/hybrid_image.png', hybrid_image)
    save_image('../results/hybrid_image_scales.png', vis)
