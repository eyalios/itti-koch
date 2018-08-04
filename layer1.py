'''
this layer of the model, is the independent feature based mapping.
'''
import numpy as np
from scipy.misc import imread as imread
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import layer2
import cv2
###CONSTANTS#####
GRAYCLE = 1
RGB = 2
TWO_D = 2
MIN_IM_SIZE = 2
RED = 0
GREEN = 1
BLUE = 2
YELLOW = 3
GAUSSIAN_SIZE = 3
PYRAMID_DEPTH = 8

def get_gaussain_filter(size):
    '''
    function gets a size, and returns the coresponding 1D gaussian filter:
    '''
    if size == 1:
        return [1]
    #creating the base to start convloving [1,1] to achive the gaussain filter.
    base = np.array([1,1])
    filter_g = np.array([1,1])
    #convolving amount of times to get right size filter.
    for i in range(size - 2):
        filter_g = np.convolve(base,filter_g)
    #return normailzed to sum = 1 filter_g, and reshaped to be used for 2d convolution
    return (filter_g/sum(filter_g)).reshape(1,size)



def bulid_gaussian_pyramid_helper(im, max_levels, gaussian_filter):
    """
    recursive function, creates gaussian pyramid of a given image up to max_levels
    @:param im - the input image
    @:param - max levels - how deep should the pyramid go.
    @:param - gaussian filter, the filter to use for the blur.
    return a pythin list of all the images in the pyramid and the vector used to covolve the image.
    """
    height, width = im.shape
    if(max_levels == 1 or height < MIN_IM_SIZE or width < MIN_IM_SIZE ):
        return[im]
    #creating the downscaled image of given input im, using the gaussian filter.
    blured_im = convolve(im,gaussian_filter,mode="reflect")
    blured_im = convolve(blured_im,gaussian_filter.T,mode="reflect")
    #slicing image taking only every other pixel
    scaled_down_im = blured_im[::2,::2]
    #calling the function recusivly to add next level of pyramid.
    pyramid = bulid_gaussian_pyramid_helper(scaled_down_im,max_levels - 1, gaussian_filter)
    #adding the image of this level to the pyramid.
    pyramid.insert(0,im)
    return pyramid



def build_gaussian_pyramid(im, max_levels, filter_size):
    '''
    function creates a gaussian image pyramid and returns it as a python array.
    @:param im- a graycle image.
    @:param max_levels- how deep should the pyramid go.
    @:param filter size- what size gaussian filter to use.
    '''
    gaussian_filter = get_gaussain_filter(filter_size)
    #creating pyramid with recursive helper function
    return bulid_gaussian_pyramid_helper(im,max_levels,gaussian_filter)

def get_color_pyramids(image):
    '''
    function creates a tuple of 4 pyramids one for each color channel- RED, GREEN, BLUE, YELLOW.
    :param image:
    :return: an array of the 4 color pyramids, red,green,blue,yellow.
    '''

    #isolating image colors
    pure_red = image[:,:,RED]
    pure_green = image[:,:,GREEN]
    pure_blue = image[:,:,BLUE]

    #creating the broadly tuned color channels.
    channel_red = pure_red - (pure_green + pure_blue) / 2
    channel_green  = pure_green-(pure_red + pure_blue) / 2
    channel_blue = pure_blue  - (pure_red + pure_green) / 2
    channel_yellow = (pure_red + pure_green) / 2 - abs(pure_red - pure_green)/2 - pure_blue

    #creating the color pyramids.
    red_pyr = build_gaussian_pyramid(channel_red,PYRAMID_DEPTH,GAUSSIAN_SIZE)
    green_pyr = build_gaussian_pyramid(channel_green,PYRAMID_DEPTH,GAUSSIAN_SIZE)
    blue_pyr = build_gaussian_pyramid(channel_blue,PYRAMID_DEPTH,GAUSSIAN_SIZE)
    yellow_pyr = build_gaussian_pyramid(channel_yellow,PYRAMID_DEPTH,GAUSSIAN_SIZE)

    layer2.get_color_feature_maps([red_pyr,green_pyr,blue_pyr,yellow_pyr])
    return

def get_intesity_pyramid(image):
    '''
    :param image: the input image.
    :return: the intesity gaussain pyramid.
    '''
    intesity_image = image[:,:,RED] + image[:,:,GREEN] + image[:,:,BLUE]
    intesity_image = intesity_image / 3
    return  layer2.get_intensity_feature_maps(build_gaussian_pyramid(intesity_image,PYRAMID_DEPTH,GAUSSIAN_SIZE))


def build_gabor_filters():
    '''

    :return: an array of the proper gabor filters for theta = 0,45,90,135
    '''
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi/4 ):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 7.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def get_angle_pyramid(image):
    '''
    :param image:
    :return: the_angle_pyramid for 0,45,90,135
    '''

    angled_pyramid = []
    #getting the intesity image.
    intesity_image = image[:, :, RED] + image[:, :, GREEN] + image[:, :, BLUE]
    intesity_image = intesity_image / 3
    #creating the reqired kernels
    gabor_filters = build_gabor_filters()
    #getting the angle pyramids.
    for theta_ker in gabor_filters:
        angled_filtered_image = cv2.filter2D(intesity_image, cv2.CV_32F, theta_ker)
        angled_pyramid.append(build_gaussian_pyramid(angled_filtered_image,PYRAMID_DEPTH,GAUSSIAN_SIZE))

    layer2.get_angle_feature_maps(angled_pyramid)
    return

