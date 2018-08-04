import numpy as np
from skimage.feature import peak_local_max as local_maxima
from scipy.misc import imread as imread
from scipy.ndimage.filters import convolve
import output
import cv2
color_map = 0
intensity_map = 0
angle_map = 0

def pass_color_conspicuity_map(color_conspicuity_map):
    '''
    saves the data in this layer.
    :param color_conspicuty_map:
    :return:
    '''
    global color_map
    color_map = color_conspicuity_map


def pass_angle_conspicuty_map(angle_conspicuity_map):
    '''
    saves the data in this layer.
    :param angle_conspicuty_map:
    :return:
    '''
    global angle_map
    angle_map = angle_conspicuity_map


def pass_intensity_conspicuty_map(intensity_conspicuity_map):
    '''
    saves the data in this layer. and calls the next stage of processing.
    :param angle_conspicuty_map:
    :return:
    '''
    global intensity_map
    intensity_map = intensity_conspicuity_map
    return create_saliency_map()


def create_saliency_map():
    '''
    computes the saliency map.
    :return:
    '''
    saliency_map = angle_map + intensity_map + color_map
    saliency_map = saliency_map / 3
    reset_layer()
    return  output.present_output(saliency_map)


def reset_layer():
    '''
    resets this layer maps.
    :return:
    '''
    color_map = 0
    intensity_map = 0
    angle_map = 0
