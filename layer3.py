import numpy as np
from skimage.feature import peak_local_max as local_maxima
from scipy.misc import imread as imread
from scipy.ndimage.filters import convolve
import layer4
import cv2

def strech_0_Max(im):
    """
    stretches given image values between 0 and max value of image
    """
    min_val = np.amin(im)
    max_val = np.amax(im)
    if(min_val!=max_val):
        im =  (im - min_val) * (max_val/(max_val-min_val))
    return im

def normalize_image(im):
    '''
    1. normalizing image values between 0 and the image max.
    2. finding global maxima and m local maximas.
    3. multiplying the image by (GM-avg(LM))^2.
    :param im:
    :return: the N.() normalized image.
    '''
    im = strech_0_Max(im)
    max_val = np.amax(im)
    #now we will get the average of all the local maximas.
    local_max = local_maxima(im,min_distance = 9 , num_peaks = 200)
    #switching the shape of the array to match im defenitions
    local_max = np.transpose(local_max)
    #getting the avg.
    local_max_vals = im[local_max[0],local_max[1]]
    max_avg  = np.average(local_max_vals)

    #this will be the image normalizing factor:
    factor = (max_val - max_avg) ** 2

    #using the factor to normailze image.
    return im * factor

def resize_to_given_scale(scale,im):
    '''

    :param scale: the shape to change to
    :param im: the image to resize
    :return: the image at the same size as the scale 4 of the given pyramid.
    '''
    return cv2.resize(im, (scale[1] , scale[0]), interpolation = cv2.INTER_LINEAR)


def create_color_conspicuty_map(color_feature_maps):
    '''

    :param color_feature_maps, (calucleted in layer2)
    :return:
    '''

    RG_feature_maps, BY_feature_maps = color_feature_maps
    #the calculation should be done on scale 4 of the pyramid.
    scale4 = RG_feature_maps[2].shape
    #setting empty map to stary adding value to.
    map = np.zeros(scale4)
    counter = 0

    #going through all feature maps created, changing size to scale 4, and adding them together.
    for C in range(1, 4):
       for S in range(6, 8):

         cur_RG = normalize_image(RG_feature_maps[counter])
         cur_RG = resize_to_given_scale(scale4, cur_RG)

         cur_BY = normalize_image(BY_feature_maps[counter])
         cur_BY = resize_to_given_scale(scale4, cur_BY)

         map = map + (cur_BY + cur_RG)

         counter+=1

    #passing up, the normalized map that was computed.
    layer4.pass_color_conspicuity_map(normalize_image(map))
    return


def create_angle_conspicuty_map(angle_feature_maps):
    '''

    :param angle_feature_maps:
    :return:
    '''
    feature_maps0, feature_maps45, feature_maps90, feature_maps135 = angle_feature_maps
    # the calculation should be done on scale 4 of the pyramid.
    scale4 = feature_maps0[2].shape
    # setting empty map to stary adding value to.
    map = np.zeros(scale4)
    counter = 0

    # going through all feature maps created, changing size to scale 4, and adding them together.
    for C in range(1, 4):
        for S in range(6, 8):
            cur_0 = normalize_image(feature_maps0[counter])
            cur_0 = resize_to_given_scale(scale4, cur_0)

            cur_45 = normalize_image(feature_maps45[counter])
            cur_45 = resize_to_given_scale(scale4, cur_45)

            cur_90 = normalize_image(feature_maps90[counter])
            cur_90 = resize_to_given_scale(scale4, cur_90)

            cur_135 = normalize_image(feature_maps135[counter])
            cur_135 = resize_to_given_scale(scale4, cur_135)

            map = map + (cur_0 + cur_45 + cur_90 + cur_135)

            counter += 1
    # passing up, the normalized map that was computed.
    layer4.pass_angle_conspicuty_map(map)
    return


def create_intensity_conspicuty_map(intensity_feature_maps):
    '''

    :param intensity_feature_maps (calucleted in layer2)
    :return:
    '''
    # the calculation should be done on scale 4 of the pyramid.
    scale4 = intensity_feature_maps[2].shape
    # setting empty map to stary adding value to.
    map = np.zeros(scale4)
    counter = 0

    # going through all feature maps created, changing size to scale 4, and adding them together.
    for C in range(1, 4):
        for S in range(6, 8):
            cur_intensity = normalize_image(intensity_feature_maps[counter])
            cur_intensity = resize_to_given_scale(scale4, cur_intensity)

            map = map + cur_intensity

            counter += 1

    # passing up, the normalized map that was computed.
    return layer4.pass_intensity_conspicuty_map(normalize_image(map))


