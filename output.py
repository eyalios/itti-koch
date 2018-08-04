import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max as local_maxima
from scipy.misc import imread as imread
import cv2

def read_image(filename):
    """
    reads an image from a given path.
    filename: the name of the image to read.
    """
    image = imread(filename)
    image = image.astype(np.float64)
    image /= 255
    return image



def winner_takes_all_loops(saliency_map, rounds, gaussian_size):
    '''

    :param saliency_map:
    :param rounds: how many rounds of winner takes all to run
    :param gaussian_size: what is the size of the gaussian to be used for the heat map.
    :return: a heat map with the most salient areas.
    '''
    #calculating the needed factor to create the gaussian and the final heat_map.
    half_gaussian_size = int(gaussian_size/2)

    #the size of the map is "gaussain_size" bigger then the original image, but it is later fixed.
    heat_map_x_size = saliency_map.shape[0] + gaussian_size
    heat_map_y_size = saliency_map.shape[1] + gaussian_size

    heat_map = np.zeros((heat_map_x_size,heat_map_y_size))

    #creating the gaussain to be the center of heat points.

    gaussian = cv2.getGaussianKernel(ksize=gaussian_size, sigma = 20)
    gaussian  = gaussian * np.transpose(gaussian)

    #finding the most salient location in the image.
    max_cordinates = local_maxima(saliency_map, min_distance = 15 , num_peaks = rounds)

    #creating the heat map by placing gaussain kernels at the salient points.
    for i in range(rounds):
        if(max_cordinates.shape[0] -1 < i):
            break
        cur_cordinate = max_cordinates[i]
        row_start  = cur_cordinate[0]
        row_end    = cur_cordinate[0] + gaussian_size

        col_start  = cur_cordinate[1]
        col_end    = cur_cordinate[1] + gaussian_size


        heat_map[int(row_start):int(row_end), int(col_start):int(col_end)] += gaussian
        heat_map[np.where(heat_map < 0)] = 0

    #resizing the heatmap to its original size.

    heat_map = heat_map[half_gaussian_size:saliency_map.shape[0] + half_gaussian_size,
                   half_gaussian_size:saliency_map.shape[1] + half_gaussian_size]
    return strech_0_1(heat_map)

def strech_0_Max(im):
    """
    stretches given image values between 0 and max value of image
    """
    min_val = np.amin(im)
    max_val = np.amax(im)
    if(min_val != max_val):
        im = (im - min_val) * (1 / (max_val - min_val))
    return im



def strech_0_1(im):
    """
    stretches given image values between 0 and max value of image
    """
    min_val = np.amin(im)
    max_val = np.amax(im)
    if(min_val != max_val):
        im =  (im - min_val) * (1/(max_val-min_val))
    return im

def present_output(im):
    im = strech_0_Max(im)
    im = winner_takes_all_loops(im, 25, 81)
    return im


def show_im(im):
    im = strech_0_Max(im)
    im = winner_takes_all_loops(im,25,81)
    im1 = read_image("testEnd1.jpg")
    #plt.imshow(im, cmap='gray')
    res = NSS(im,im1)
    print(res)
    plt.imshow(im,cmap='gray')
    return res





