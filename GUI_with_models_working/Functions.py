import string
import os

import wx
import cv2
import numpy as np
import scipy.misc
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json
from PIL import Image
from wx.lib.pubsub import pub

#Focus the fundus image on the disk only for segmented image
def remove_segmented_background(segmented_img):
    all_coordinates = []
    disk_lst = []
    for x in range(224):
        for y in range (224):
            try:
                test_list = []
                all_coordinates.append([x,y])
                if segmented_img[x,y] !=0:
                    if (segmented_img[x+1,y] !=0 or segmented_img[x+1,y+1] !=0 or segmented_img[x,y+1] !=0 or segmented_img[x-1,y] !=0 or segmented_img[x-1,y-1] !=0 or segmented_img[x+1,y-1] !=0 or segmented_img[x,y-1] !=0 or segmented_img[x-1,y+1] !=0):
                        radius = 10
                        for x_square in range (radius):   #Test that the area surrounding the square has more than 30% purple in it
                            for y_square in range (radius):
                                if segmented_img[x-x_square,y-y_square]==0:
                                    test_list.append(segmented_img[x,y])
                                if segmented_img[x+x_square,y+y_square]==0:
                                    test_list.append(segmented_img[x,y])
                        if (len(test_list)/((2*radius)**2))>0.3: #if the purple part covers more than 30% of the test square
                            segmented_img[x,y]=0
                    else:
                        segmented_img[x,y]=0
            except: #if x+x_square is greater than 224 (dimensions of image)
                try:
                    test_list = []
                    all_coordinates.append([x,y])
                    if segmented_img[x,y] !=0:
                        if (segmented_img[x+1,y] !=0 or segmented_img[x+1,y+1] !=0 or segmented_img[x,y+1] !=0 or segmented_img[x-1,y] !=0 or segmented_img[x-1,y-1] !=0 or segmented_img[x+1,y-1] !=0 or segmented_img[x,y-1] !=0 or segmented_img[x-1,y+1] !=0):
                            radius = 5
                            for x_square in range (radius):   #Test that the area surrounding the square has more than 5% purple in it
                                for y_square in range (radius):
                                    if segmented_img[x-x_square,y-y_square]==0:
                                        test_list.append(segmented_img[x,y])
                                    if segmented_img[x+x_square,y+y_square]==0:
                                        test_list.append(segmented_img[x,y])
                            if (len(test_list)/((2*radius)**2))>0.05: #if the purple part covers more than 5% of the test square
                                segmented_img[x,y]=0
                        else:
                            segmented_img[x,y]=0
                except:
                    try:
                        test_list = []
                        all_coordinates.append([x,y])
                        if segmented_img[x,y] !=0:
                            if (segmented_img[x+1,y] !=0 or segmented_img[x+1,y+1] !=0 or segmented_img[x,y+1] !=0 or segmented_img[x-1,y] !=0 or segmented_img[x-1,y-1] !=0 or segmented_img[x+1,y-1] !=0 or segmented_img[x,y-1] !=0 or segmented_img[x-1,y+1] !=0):
                                radius = 2
                                for x_square in range (radius):   #Test that the area surrounding the square has more than 1% purple in it
                                    for y_square in range (radius):
                                        if segmented_img[x-x_square,y-y_square]==0:
                                            test_list.append(segmented_img[x,y])
                                        if segmented_img[x+x_square,y+y_square]==0:
                                            test_list.append(segmented_img[x,y])
                                if (len(test_list)/((2*radius)**2))>0.01: #if the purple part covers more than 1% of the test square
                                    segmented_img[x,y]=0
                            else:
                                segmented_img[x,y]=0
                    except:
                        pass
    return segmented_img
#Calculate cup to disk ratio
def calculate_c2d_ratio_one(segmented_img):
    disk_lst = []
    cup_lst = []
    for x in range(224):
        for y in range (224):
            if segmented_img[x,y] ==1:
                disk_lst.append ([x,y])
    for x in range(224):
        for y in range (224):
            if segmented_img[x,y] ==2:
                cup_lst.append ([x,y])
    for elt in [cup_lst, disk_lst]:
        if len(elt) == 0:
            print(str(elt)+ " is empty")
    y_disk_lst = []
    for coordinate in disk_lst:
        y_disk_lst.append(coordinate[0])
    y_cup_lst = []
    for coordinate in cup_lst:
        y_cup_lst.append(coordinate[0])
    return ((max(y_cup_lst)-min(y_cup_lst))/(max(y_disk_lst)-min(y_disk_lst)))
def prcss(segmented_img):
    all_coordinates = []
    disk_lst = []
    for x in range(224):
        for y in range (224):
            try:
                test_list = []
                all_coordinates.append([x,y])
                if segmented_img[x,y] !=0:
                    if (segmented_img[x+1,y] !=0 or segmented_img[x+1,y+1] !=0 or segmented_img[x,y+1] !=0 or segmented_img[x-1,y] !=0 or segmented_img[x-1,y-1] !=0 or segmented_img[x+1,y-1] !=0 or segmented_img[x,y-1] !=0 or segmented_img[x-1,y+1] !=0):
                        radius = 10
                        for x_square in range (radius):   #Test that the area surrounding the square has more than 30% purple in it
                            for y_square in range (radius):
                                if segmented_img[x-x_square,y-y_square]==0:
                                    test_list.append(segmented_img[x,y])
                                if segmented_img[x+x_square,y+y_square]==0:
                                    test_list.append(segmented_img[x,y])
                        if (len(test_list)/((2*radius)**2))>0.3: #if the purple part covers more than 30% of the test square
                            segmented_img[x,y]=0
                    else:
                        segmented_img[x,y]=0
            except: #if x+x_square is greater than 224 (dimensions of image)
                try:
                    test_list = []
                    all_coordinates.append([x,y])
                    if segmented_img[x,y] !=0:
                        if (segmented_img[x+1,y] !=0 or segmented_img[x+1,y+1] !=0 or segmented_img[x,y+1] !=0 or segmented_img[x-1,y] !=0 or segmented_img[x-1,y-1] !=0 or segmented_img[x+1,y-1] !=0 or segmented_img[x,y-1] !=0 or segmented_img[x-1,y+1] !=0):
                            radius = 5
                            for x_square in range (radius):   #Test that the area surrounding the square has more than 5% purple in it
                                for y_square in range (radius):
                                    if segmented_img[x-x_square,y-y_square]==0:
                                        test_list.append(segmented_img[x,y])
                                    if segmented_img[x+x_square,y+y_square]==0:
                                        test_list.append(segmented_img[x,y])
                            if (len(test_list)/((2*radius)**2))>0.05: #if the purple part covers more than 5% of the test square
                                segmented_img[x,y]=0
                        else:
                            segmented_img[x,y]=0
                except:
                    try:
                        test_list = []
                        all_coordinates.append([x,y])
                        if segmented_img[x,y] !=0:
                            if (segmented_img[x+1,y] !=0 or segmented_img[x+1,y+1] !=0 or segmented_img[x,y+1] !=0 or segmented_img[x-1,y] !=0 or segmented_img[x-1,y-1] !=0 or segmented_img[x+1,y-1] !=0 or segmented_img[x,y-1] !=0 or segmented_img[x-1,y+1] !=0):
                                radius = 2
                                for x_square in range (radius):   #Test that the area surrounding the square has more than 1% purple in it
                                    for y_square in range (radius):
                                        if segmented_img[x-x_square,y-y_square]==0:
                                            test_list.append(segmented_img[x,y])
                                        if segmented_img[x+x_square,y+y_square]==0:
                                            test_list.append(segmented_img[x,y])
                                if (len(test_list)/((2*radius)**2))>0.01: #if the purple part covers more than 1% of the test square
                                    segmented_img[x,y]=0
                            else:
                                segmented_img[x,y]=0
                    except:
                        pass
    return segmented_img
def load_keras_model(self):
    "load the keras model"
    with open ("/Users/leonardomilea/disk_model_newest.json", "r") as f:
        self.model = model_from_json(f.read())
    self.model.load_weights("/Users/leonardomilea/diskweights_newest.h5")
def make_numpy_array(self, image):
    "store image as a numpy array"
    image_raw = Image.open(image)
    image_raw = image_raw.resize((224, 224), Image.ANTIALIAS) #resize
    image_raw = np.array(image_raw)
    self.array = (image_raw).astype(np.uint8)
def prepare_seg ():
    "Predict the raw image's segmentation"
    self.make_numpy_array(self.filepath_raw)
    raw_img_equalized = np.zeros_like(self.array, dtype = np.float)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  #for histogram equalization
    b,g,r = cv2.split(self.array)
    raw_img_equalized[:,:,0] = clahe.apply(b) / 255.
    raw_img_equalized[:,:,1] = clahe.apply(g) / 255.
    raw_img_equalized[:,:,2] = clahe.apply(r) / 255.
    im = cv2.resize(raw_img_equalized,(224,224))
    im = np.reshape(im, (1,224,224,3))
    im = K.variable(im)
    prediction = self.model.predict(im, steps = 1)
    prediction = np.reshape(prediction, (224,224,3))
    prediction = np.argmax(prediction, axis=2)
    prediction = prcss(remove_segmented_background(prediction))
    self.parent.show_c2d_ratio(calculate_c2d_ratio_one(prediction))
    plt.imsave(self.filepath, prediction)
    self.load_image_seg(self.filepath)
