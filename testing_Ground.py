from tkinter import filedialog
from tkinter import messagebox
from tkinter import *
from PIL import ImageTk,Image

import matplotlib.pyplot as plt
import layer1
import input
import layer2
import layer3
import math
import numpy as np
import  os
import cv2
from scipy.misc import imread as imread
from scipy.misc import imsave as imsave
from skimage.feature import peak_local_max as local_maxima
from skimage import img_as_float as img_as_float


def strech_0_Max(im):
    """
    stretches given image values between 0 and max value of image
    """
    min_val = np.amin(im)
    max_val = np.amax(im)
    if(min_val != max_val):
        im = (im - min_val) * (1 / (max_val - min_val))
    return im


def NSS(salincey_map, fixation_image):
    '''
    :param salincey_map:
    :param fixation_image:
    :return: the NSS score of the saliency map and the fixation image.
    '''

    # getting the fixation map to the same shape as the saliency map:
    fixation_image = cv2.resize(fixation_image, (salincey_map.shape[1], salincey_map.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
    # turning the fixation image into a binary image.
    fixation_image[fixation_image <= 0] = 0
    fixation_image[fixation_image > 0] = 1

    salincey_map = strech_0_Max(salincey_map)
    #creating the normalized sailency map as required to compute NSS
    sail_var = np.var(salincey_map)
    if(sail_var == 0):
        return 0
    sail_exp = np.mean(salincey_map)
    normalized_saliency_map = (salincey_map-sail_exp)/np.sqrt(sail_var)
    # getting the amount of pixels that had fixation.
    num_fixated_pixels = sum(sum(fixation_image))

    sum_of_matrices = sum(normalized_saliency_map[fixation_image != 0])
    if(num_fixated_pixels != 0):
         return sum_of_matrices / num_fixated_pixels
    return 0
def read_image(filename):
    """
    reads an image from a given path.
    filename: the name of the image to read.
    """
    image = imread(filename)
    image = image.astype(np.float64)
    image /= 255
    return image


def test_all_Samples():
    '''
    function tests the algorithm using
    :return:
    '''
    fixations = np.zeros((700,1000,1000))
    counter = 0
    NSS_AVG = 0
    NSS_SUM = 0
    #getting all fixation pics to an array.
    for root, dirs, files in os.walk('C:/Users/USER/Documents/seminar/FIXATIONMAPS'):
        for file in files:
            with open(os.path.join(root, file), "r") as auto:
                fixations[counter] = read_image(auto.name)
                counter += 1


    num_of_files = counter
    counter = 0
    divide = 0
    #in this loop sending each target photo to be modeled and then using NSS to
    #mesure how well it did againts the fixation map.
    for root, dirs, files in os.walk('C:/Users/USER/Documents/seminar/Targets'):
        for file in files:
            with open(os.path.join(root, file), "r") as auto:
                #if(counter < 434):
                   # counter +=1
                   # continue
                target = input.start_process(read_image(auto.name))
                #summing NSS results.
                cur_nss= NSS(target, fixations[counter])
                if(cur_nss == 0):
                    print(auto.name)
                    counter += 1
                else:
                    NSS_SUM += cur_nss
                    print(NSS_SUM)
                    counter += 1
                    divide += 1
                    print(["cur avg:", NSS_SUM / divide])
                    print(counter)
    return NSS_SUM / divide


def create_heat_map(original_im, saliency_map):
    alpha = 0.4
    original_im = cv2.resize(original_im, (saliency_map.shape[1], saliency_map.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
    max_salient = np.max(saliency_map)
    heat_step  = max_salient /7
    color_heat_map = np.zeros((original_im.shape[0],original_im.shape[1],4))
    #color_heat_map[:,:,3] = 0.5
    red_spots = np.argwhere(saliency_map > max_salient - heat_step)
    yellow_spots = np.argwhere(np.logical_and(saliency_map <= max_salient - heat_step , saliency_map > max_salient - 2*heat_step))
    green_spots  = np.argwhere(np.logical_and(saliency_map <= max_salient - 2* heat_step , saliency_map > max_salient - 3 * heat_step))
    blue_spots  = np.argwhere(saliency_map <= max_salient - 3* heat_step)
    color_heat_map[red_spots[:,0],red_spots[:,1]] = (1,0,0,0.5)
    color_heat_map[yellow_spots[:,0],yellow_spots[:,1]] = (1,1,0,0.5)
    color_heat_map[green_spots[:,0], green_spots[:,1]] = (0,1,0,0.5)
    color_heat_map[blue_spots[:,0],blue_spots[:,1]] = (0,0,0,0.5)

    if (original_im.shape[2] == 3):
        temp_im = np.ones((original_im.shape[0], original_im.shape[1], 4))
        temp_im[:, :, 0:3] = original_im
        original_im = temp_im

    cv2.addWeighted(color_heat_map, alpha, original_im, 1 - alpha,
                    0, color_heat_map)


    return color_heat_map


def test_heat(im):
    heat = input.start_process(im)
    create_heat_map(im,heat)



class MyFirstGUI:

    def __init__(self, master):
        self.master = master
        master.title("Itti & Koch visual Saliency algorithm")
        self.v = StringVar()

        self.label1 = Label(master, text="enter full file path or browse to it,then click run").pack()
        self.e1 = Entry(master,width = 120)
        self.e1.pack()
        self.button2 = Button(text="Browse for image", command=lambda: browse_button(self.e1))
        self.button2.pack()

        self.label2 = Label(master, text="if there is a fixation map,enter full file path or browse to it,then click run").pack()
        self.e2 = Entry(master, width=120)
        self.e2.pack()
        self.button3 = Button(text="Browse for fixation map", command=lambda: browse_button(self.e2))
        self.button3.pack()
        self.button4 = Button(text="create sailency map", command= self.create_heat_map_for_gui)
        self.button4.pack()

        self.panel = Label()
        self.im_text = Label(text = "<-Saliency map \n heat map").pack(side=RIGHT)
        self.panel.pack(fill=BOTH, expand=1,side=RIGHT)


        self.panel1 = Label()
        self.im_text = Label(text="input image->").pack(side=LEFT)
        self.panel1.pack(fill=BOTH, expand=1,side = LEFT)
        self.sail_text = Label().pack()

        self.panel2 = Label()
        self.panel2.pack(side = BOTTOM)


        def browse_button(e1):
            filename = filedialog.askopenfilename()
            e1.delete(0,'end')
            e1.insert(0, filename)
            return filename

    def create_heat_map_for_gui(self):
        '''
        getting the data from the entries, creating and displaying the salient heat map.
        if there was a fixation map given, will give the nss resault.
        :return:
        '''
        #getting the file paths from gui.
        original_im = (self.e1.get())
        fixation_map = self.e2.get()
        got_fixation = False
        #checking if a fixation map was given.
        if(fixation_map!=''):
            got_fixation = True
        #if the original image path was not found ask for it again
        if(not (os.path.isfile(original_im))):
            messagebox.showinfo("INPUT IMAGE ERROR", "the file path you inserted was not found")
            return
        if (not (os.path.isfile(fixation_map)) and got_fixation):
            messagebox.showinfo("FIXATION_MAP", "the file path you inserted was not found")
            return
        #now reading the files and creating the heat map.
        original_im = read_image(original_im)


        #running the algorithm and creating the sailency heat maps.
        salicney_map = input.start_process(original_im)
        heat_map = create_heat_map(original_im, salicney_map)

        #if a fixation map was given computing the NSS.
        if(got_fixation):
            fixation_map = read_image(fixation_map)
            res = NSS(salicney_map,fixation_map)
            self.panel2.config(text= "NSS score between fixation map and Saliency map is " + str(res) )
        else:
            self.panel2.config(text = "")

        ####producing the output to gui.
        #creating the heatmap image and original image in a format presentable in gui
        imsave('outfile.png', heat_map)
        imsave('original_im.png',original_im)
        heat_map_for_gui = Image.open('outfile.png').resize((350,350))
        im_for_gui = Image.open('original_im.png').resize((350,350))

        heat_map_for_gui = ImageTk.PhotoImage(image=heat_map_for_gui)
        im_for_gui = ImageTk.PhotoImage(image=im_for_gui)

        self.panel.config(image = heat_map_for_gui)
        self.panel.image = heat_map_for_gui


        self.panel1.config(image = im_for_gui)
        self.panel1.image = im_for_gui




root = Tk()



width, height = root.winfo_screenwidth(), root.winfo_screenheight()

root.geometry('%dx%d+0+0' % (width,height*0.8))

my_gui = MyFirstGUI(root)
root.mainloop()
#test_heat(read_image('testim8.jpg'))

#print(test_all_Samples())
