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

class DropTarget(wx.FileDropTarget):
    "create a droptarget type"
    PhotoMaxSize = 400
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
    def OnDropFiles(self, x, y, filenames):
        image = Image.open(filenames[0])
        image.thumbnail((400, 400))
        image.save('thumbnail.png')
        pub.sendMessage('dnd', filepath='thumbnail.png')
        return True
class ImagePanel (wx.Panel):
    def __init__(self, parent, image_size):
        super().__init__(parent)
        pub.subscribe(self.update_image_on_dnd, 'dnd')
        self.max_size = 400
        self.converted = True
        self.alpha_out = True
        self.disk_pixels = []
        self.array = None
        self.model = None
        self.filepath = '/Users/leonardomilea/outfile.BMP'
        img_raw = wx.Image (*image_size)
        img_seg = wx.Image (*image_size)
        self.img_ctrl_raw = wx.StaticBitmap(self, bitmap=wx.Bitmap(img_raw), pos = (200, 150))
        self.img_ctrl_seg = wx.StaticBitmap(self, bitmap=wx.Bitmap(img_seg), pos = (200, 150))
        browse_button = wx.Button (self, label = "Browse")
        browse_button.Bind (wx.EVT_BUTTON, self.on_button)
        self.alpha_button = wx.Slider (self)
        self.alpha_button.Bind (wx.EVT_SCROLL, self.alpha_func)
        self.current_percentage = "Transparency"
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(browse_button, 0, wx.CENTER|wx.ALL, 15)
        main_sizer.Add(self.alpha_button, 0, wx.CENTER|wx.ALL, 15)
        self.SetSizer(main_sizer)
        self.load_keras_model()
        filedroptarget = DropTarget(self)
        self.img_ctrl_raw.SetDropTarget(filedroptarget)
    def alpha_func (self, event):
        "changes the transparency of segmented image"
        self.load_image_seg(self.filepath, self.alpha_button.GetValue()*255//100)
    def update_image_on_dnd(self, filepath):
        self.load_image_raw(filepath=filepath)
    def load_keras_model(self):
        with open ("/Users/leonardomilea/vesselweights.json", "r") as f:
            self.model = model_from_json(f.read())
        self.model.load_weights("/Users/leonardomilea/vesselweights.h5")
        print (self.model)
    def on_button(self, event):
        "load file with browse button"
        with wx.FileDialog (None, "Choose a File", wildcard = "(*.jpg, *.pdf, *.png, *.tif)|*.jpg;*.pdf;*.png;*.tif") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.filepath_raw = dlg.GetPath()
                self.load_image_raw (self.filepath_raw)
    def load_image_raw(self, filepath):
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
    def prepare_seg (self):
        "predict the raw image's segmentation"
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.make_numpy_array(self.filepath_raw)
        b,green_fundus,r = cv2.split(self.array)
        ax = clahe.apply(green_fundus) / 255.
        ax = np.clip(ax, 0., 1.)
        im = np.reshape(ax, (1,224,224,1))
        im = K.variable(im)
        prediction = self.model.predict(im, steps = 1)
        prediction = np.reshape(prediction, (224,224,3))
        prediction=np.argmax(prediction, axis=2)
        prediction = self.reduce_spluttering(prediction)
        plt.imsave(self.filepath, prediction)
        self.load_image_seg(self.filepath)
    def reduce_spluttering(self,seg_map, threshold = 50):
        "get rid of small connected component"
        ret, labels = cv2.connectedComponents(seg_map.astype(np.uint8))
        for k in range(ret):
            size_component = np.sum(labels==k)
            if size_component <= threshold:
                seg_map[labels==k] = 0
        return seg_map
    def make_numpy_array(self, image):
        "store image as a numpy array"
        image_raw = Image.open(image)
        image_raw = image_raw.resize((224, 224), Image.ANTIALIAS) #resize
        image_raw = np.array(image_raw)
        self.array = (image_raw).astype(np.uint8)
    def load_image_seg(self, image, alpha = 0):
        "load a segmented image"
        im = wx.Image(image, wx.BITMAP_TYPE_ANY)
        self.make_numpy_array(self.filepath)
        if self.alpha_out:
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
        for x in range (224):
            for y in range (224):
                if self.array[y,x,2] != 140 and self.array[y,x,2] != 84 :
                    self.disk_pixels.append([x,y]) # create a list of all vessel pixel coordinates
        self.alpha_out = False
class MainFrame (wx.Frame):
    def __init__(self):
        super().__init__(None, title='Glaucoma Detector', size=(800, 600))
        panel = ImagePanel(self, image_size=(400, 400))
        self.Show()
if __name__ == '__main__':
    app = wx.App(redirect=False)
    frame = MainFrame()
    app.MainLoop()
