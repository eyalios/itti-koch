'''
this is the input\output file.
contains the following functions:
*read_image -> receives a path and open a given image file.
'''
import cv2
import numpy as np
from scipy.misc import imread as imread
import layer1
from scipy.ndimage.filters import convolve
###CONSTANTS#####
GRAYCLE = 1
RGB = 2
TWO_D = 2

RED = 0
GREEN = 1
BLUE = 2
YELLOW = 3

#####functions###


def read_image(filename):
    """
    reads an image from a given path.
    filename: the name of the image to read.
    """
    image = imread(filename)
    image = image.astype(np.float64)
    image /= 255
    return image

def resize_to_given_scale(scale,im):
    '''

    :param scale: the shape to change to
    :param im: the image to resize
    :return: the image at the same size as the scale 4 of the given pyramid.
    '''
    return cv2.resize(im, (scale[1] , scale[0]), interpolation = cv2.INTER_LINEAR)



def start_process(image):
    if(image.shape[0] < 640 or image.shape[1] < 480):

        image = resize_to_given_scale((640,480),image)
    print("hello")
    layer1.get_color_pyramids(image)
    layer1.get_angle_pyramid(image)
    return layer1.get_intesity_pyramid(image)
'''
check = read_image("test1.jpg")
#check = read_image("testim.jpg")
start_process(check)
'''