# CSCI 1430
# Homework 1 Image Filtering
#
import numpy as np
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import resize

def load_image(path):
    return img_as_float32(io.imread(path))

def save_image(path, im):
    return io.imsave(path, img_as_ubyte(im.copy()))

# given two differently sized images, resize them so they have the same shape
def equalize_image_sizes(im_one, im_two):
    assert im_one.shape[2] == im_two.shape[2], 'the third dimension of these images do not match'
    # resizes by adding/subtracting half of the difference between the image's width and height
    x_resize = (im_one.shape[0] - im_two.shape[0]) / 2
    y_resize = (im_one.shape[1] - im_two.shape[1]) / 2
    im_one = resize(im_one, (int(im_one.shape[0] - x_resize), int(im_one.shape[1] - y_resize), im_one.shape[2]))
    im_two = resize(im_two, (int(im_two.shape[0] + x_resize), int(im_two.shape[1] + y_resize), im_two.shape[2]))
    return im_one, im_two
