import string
import os
import getpass

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

from Functions import remove_segmented_background, calculate_c2d_ratio_one, prcss

class DropTarget(wx.FileDropTarget):
    "create a drop target"
    PhotoMaxSize = 400
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
    def OnDropFiles(self, x, y, filenames):
        image = Image.open(filenames[0])
        image.thumbnail((400, 400))
        image.save('thumbnail.png')
        pub.sendMessage('dnd', filepath='thumbnail.png')
        self.widget.reset_disk_pixels = True
        self.widget.disk_pixels = []
        return True
class ImagePanel (wx.Panel):
    def __init__(self, parent, image_size):
        super().__init__(parent)
        pub.subscribe(self.update_image_on_dnd, 'dnd')
        self.initialise_inputs(parent, image_size)
        self.layout(parent)
        self.load_keras_model()
        self.make_initial_background()
    def initialise_inputs(self, parent, image_size):
        self.max_size = 400
        self.parent = parent
        self.model = None #segmentation model
        self.reset_disk_pixels = True
        self.array = None
        self.disk_pixels = [] #List of coordinates which aren't always transparent (the disk+cup pixels)
        self.filepath = '/Users/'+getpass.getuser()+'/outfile.BMP'
        self.img_raw = wx.Image (*image_size)
        self.img_seg = wx.Image (*image_size)
    def layout(self, parent):
        self.img_ctrl_raw = wx.StaticBitmap(self, bitmap=wx.Bitmap(self.img_raw), pos = (200, 150))
        print ("bitmap")
        self.img_ctrl_seg = wx.StaticBitmap(self, bitmap=wx.Bitmap(self.img_seg), pos = (200, 150))
        browse_button = wx.Button (self, label = "Browse")
        browse_button.Bind (wx.EVT_BUTTON, self.on_button)
        self.alpha_button = wx.Slider (self)
        self.alpha_button.SetBackgroundColour((250,246,0))
        self.alpha_button.Bind (wx.EVT_SCROLL, self.alpha_func)
        self.current_percentage = "Transparency"
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.main_sizer.Add(browse_button, 0, wx.CENTER|wx.ALL, 15)
        self.main_sizer.Add(self.alpha_button, 0, wx.CENTER|wx.ALL, 15)
        self.SetSizer(self.main_sizer)
        filedroptarget = DropTarget(self)
        self.img_ctrl_raw.SetDropTarget(filedroptarget)
    def alpha_func (self, event):
        "changes the transparency of segmented image"
        self.load_image_seg(self.filepath, self.alpha_button.GetValue()*255//100) #change from slider's /100 to /255 for alpha
    def update_image_on_dnd(self, filepath):
        self.load_image_raw(filepath=filepath)
    def load_keras_model(self):
        "load the keras model"
        with open ("/Users/leonardomilea/disk_model_newest.json", "r") as f:
            self.model = model_from_json(f.read())
        self.model.load_weights("/Users/leonardomilea/diskweights_newest.h5")
    def on_button(self, event):
        "select raw image with browse button"
        with wx.FileDialog (None, "Choose a File", wildcard = "(*.jpg, *.pdf, *.png, *.tif)|*.jpg;*.pdf;*.png;*.tif") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.reset_disk_pixels, self.disk_pixels = True, [] #reset the disk pixels
                self.filepath_raw = dlg.GetPath()
                self.load_image_raw (self.filepath_raw)
    def load_image_raw(self, filepath):
        self.alpha_button.SetValue(0)
        self.filepath_raw = filepath
        img = wx.Image(filepath, wx.BITMAP_TYPE_ANY)
        W = img.GetWidth()
        H = img.GetHeight ()
        new_W = self.max_size
        new_H = self.max_size
        img = img.Scale(new_W, new_H)
        self.img_ctrl_raw.SetBitmap(wx.Bitmap(img))
        self.prepare_seg()
        self.Refresh()
    def make_numpy_array(self, image):
        "store image as a numpy array"
        image_raw = Image.open(image)
        image_raw = image_raw.resize((224, 224), Image.ANTIALIAS) #resize
        image_raw = np.array(image_raw)
        self.array = (image_raw).astype(np.uint8)
    def prepare_seg (self):
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
    def make_initial_background(self):
        im = wx.Image("/Users/leonardomilea/Desktop/Black.jpg", wx.BITMAP_TYPE_ANY)
        im = im.Scale(self.max_size, self.max_size)
        self.img_ctrl_seg.SetBitmap(wx.Bitmap(im))
        self.Refresh()
    def load_image_seg(self, image, alpha = 0):
        "load a segmented image"
        im = wx.Image(image, wx.BITMAP_TYPE_ANY)
        self.make_numpy_array(self.filepath)
        if self.reset_disk_pixels:
            self.sort_disk()
        for x in range (224):
            for y in range (224):
                im.SetAlpha(x, y, 0) # make everything transparent
        for e in self.disk_pixels:
            im.SetAlpha(e[0], e[1], alpha) #make the disk pixels the required transparency
        W = im.GetWidth()
        H = im.GetHeight ()
        if W >= H:
            new_W = self.max_size
            new_H = self.max_size*H/W
        else:
            new_H = self.max_size
            new_W = self.max_size*W/H
        im = im.Scale(new_W, new_H)
        self.img_ctrl_seg.SetBitmap(wx.Bitmap(im))
        self.Refresh()
    def sort_disk(self):
        "create a list of all disk pixel coordinates"
        self.reset_disk_pixels = False
        for x in range (224):
            for y in range (224):
                if self.array[y,x,0] != 68:
                    self.disk_pixels.append([x,y])
class MainFrame (wx.Frame):
    def __init__(self):
        super().__init__(None, title='Glaucoma Detector', size=(800, 650))
        self.panel = ImagePanel(self, image_size=(400, 400))
        self.change_text = False
        self.SetBackgroundColour('black')
        self.Show()
    def show_c2d_ratio(self, value = "Your Cup To Disk Ratio"): #Create static text showing cup to disk ratio
        if self.change_text == False:
            self.c2d_ratio = wx.StaticText(self, label = "Your Cup To Disk Ratio is "+ '{0:.3f}'.format(value), pos = (298, 550))
            self.c2d_ratio.SetForegroundColour((250,246,0)) # set text color
            self.change_text = True
        else:
            self.c2d_ratio.SetLabel("Your Cup To Disk Ratio is "+ '{0:.3f}'.format(value))
if __name__ == '__main__':
    app = wx.App(redirect=False)
    frame = MainFrame()
    app.MainLoop()
