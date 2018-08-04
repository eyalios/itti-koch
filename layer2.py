import numpy as np
from scipy.misc import imread as imread
from scipy.ndimage.filters import convolve
import cv2
import layer3
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


def pad_im_with_zeros(im):
    """
    gets an image an pads it with a 0 every other pixel
    @:param im- the image to pad.
    """
    height, width = im.shape
    #setting a new zeros np array with double the size of the given im.
    padded_im = np.zeros((height*2,width*2))
    padded_im[::2,::2] = im
    return padded_im


def expand_im(im,dest_size):
    """
    function gets an image
    return the expanded image with a convolution of the gaussian filter filter.
    """
    return cv2.resize(im, (dest_size[1],dest_size[0]), interpolation = cv2.INTER_LINEAR)

def substract_images_of_different_size(big_im, small_im):
    '''
    :param big_im:
    :param small_im:
    :return: the difference between the two images pixel by pixel, after scaling up 'small im'
     using interpolation.
    '''
    enlarged_im = expand_im(small_im, big_im.shape)
    return big_im - enlarged_im

def get_color_feature_maps(color_pyr):
    """
    this feature map is calculated by substracting all combos between lvl(1,2,3) and (6,7), i.e (7th and 8th lvls).
    :return: the 12 colors feature maps as an array, the first array is 6 entries are the RG map and the 6 others are the
    BY maps.
    """
    RG_color_maps = []
    BY_color_maps = []

    red_pyr,green_pyr,blue_pyr,yellow_pyr = color_pyr
    for C in range (1,4):
        for S in range (6,8):
            #caluclating the current RG map using the colors pyrmamids.
            big_RG = red_pyr[C] - green_pyr[C]
            small_RG = green_pyr[S] - red_pyr[S]
            current_RG_map = abs(substract_images_of_different_size(big_RG,small_RG))

            # caluclating the current BY map using the colors pyrmamids.
            big_BY = blue_pyr[C] - yellow_pyr[C]
            small_BY = yellow_pyr[S] - blue_pyr[S]
            current_BY_map = abs(substract_images_of_different_size(big_BY,small_BY))

            RG_color_maps.append(current_RG_map)
            BY_color_maps.append(current_BY_map)

    layer3.create_color_conspicuty_map([RG_color_maps, BY_color_maps])


def get_intensity_feature_maps(intensity_pyr):
    '''
    :param im
    :return:  this feature map is calculated by substracting all combos between lvl(1,2,3) and (6,7),
     i.e (7th and 8th lvls).
    '''
    intesity_maps = []
    for C in range(1, 4):
        for S in range(6, 8):
            # caluclating the current intesity map using the intesity pyramid
            current_intesity_map = abs(substract_images_of_different_size(intensity_pyr[C] , intensity_pyr[S]))
            #adding the map to the array
            intesity_maps.append(current_intesity_map)
    return    layer3.create_intensity_conspicuty_map(intesity_maps)

def get_angle_feature_maps(angle_pyr):
    '''
     the angle feature maps is calculated by substracting all combos between lvl(1,2,3) and (6,7),
     i.e (7th and 8th lvls).
    :param im
    :return: 0,45,90,135 feature maps
    '''
    p0,p45,p90,p135 = angle_pyr
    map0  = []
    map45 = []
    map90 = []
    map135 = []
    for C in range(1, 4):
        for S in range(6, 8):
            # caluclating the current angle0 map.
            current_intesity_map = abs(substract_images_of_different_size(p0[C] , p0[S]))
            #adding the map to the array
            map0.append(current_intesity_map)

            # caluclating the current angle45 map.
            current_intesity_map = abs(substract_images_of_different_size(p45[C] , p45[S]))
            #adding the map to the array
            map45.append(current_intesity_map)

            # caluclating the current angle90 map.
            current_intesity_map = abs(substract_images_of_different_size(p90[C] , p90[S]))
            #adding the map to the array
            map90.append(current_intesity_map)

            # caluclating the current angle135 map.
            current_intesity_map = abs(substract_images_of_different_size(p135[C] , p135[S]))
            #adding the map to the array
            map135.append(current_intesity_map)
    layer3.create_angle_conspicuty_map([map0,map45,map90,map135])


