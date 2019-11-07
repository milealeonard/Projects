#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import os as os
from os import environ
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Reshape, SpatialDropout2D, Activation
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import MaxPooling2D, ZeroPadding2D, UpSampling2D, Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import PReLU, ELU
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D, SeparableConv2D
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow as tf



#from os.path import dirname, abspath
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image

import random as random
import cv2 as cv2

# In[2]:

dim_x, dim_y = 224, 224  #read and resize images to this dimension

path_segmentation = ["/home/leo/Segmentation/Vessels/STARE",
                     "/home/leo/Segmentation/Vessels/HRF",
                     "/home/leo/Segmentation/Vessels/DRIVE"]



# In[4]:


#extractino of training data
segmentation_data = []
for path_current in path_segmentation:
    study = path_current.split("/")[-1]
    print("--->", study)
    path_raw = path_current + "/raw"
    image_files = [f for f in listdir(path_raw) if isfile(join(path_raw, f))
                                       and (f.endswith(".jpg")
                                            or f.endswith(".JPG")
                                            or f.endswith(".tif")
                                            or f.endswith(".ppm"))]
    path_seg = path_current + "/segmentation"
    seg_files = [f for f in listdir(path_seg) if isfile(join(path_seg, f))
                                         and (f.endswith(".jpg")
                                              or f.endswith(".JPG")
                                              or f.endswith(".tif")
                                              or f.endswith(".ppm")
                                              or f.endswith(".gif"))]
    for f in image_files:
        #check whether one can find a pair raw/segmentation image
        start_file_raw = f.split(".")[0] #extract the name of the file without extension
        #explore the segmentation folder and check if there is a corresponding file
        for f_seg in seg_files:
            start_file_seg = f_seg.split(".")[0] #extract the name of the file without extension
            if start_file_raw == start_file_seg:
                image_raw = Image.open(join(path_raw, f))
                image_raw = image_raw.resize((dim_x, dim_y), Image.ANTIALIAS) #resize
                image_raw = np.array(image_raw) #transform to Numpy

                image_seg = Image.open(join(path_seg, f_seg))
                image_seg = image_seg.resize((dim_x, dim_y), Image.ANTIALIAS) #resize
                image_seg = np.array(image_seg) #transform to Numpy

                print("raw file: {} \t \t segmentation file: {}".format(f, f_seg))
                segmentation_data.append([image_raw, image_seg, study])
                break





# In[5]:


def create_mask(study):
    """
    according to the study, work out the region of interest
    """
    X, Y = np.meshgrid(np.arange(224),np.arange(224))
    if study == "STARE":
        center_x, center_y = dim_x//2, dim_y // 2
        #print("STARE")
        radius = 104
        mask = (X-center_x)**2 + 0.7*(Y-center_y)**2 > radius**2
        mask[:10,:] = True
        mask[-10:,:] = True
    elif study == "HRF":
        #print("HRF")
        center_x, center_y = dim_x//2, dim_y // 2
        radius = 104
        mask = (X-center_x)**2 + 0.45*(Y-center_y)**2 > radius**2

    elif study == "DRIVE":
        #print("DRIVE")
        center_x, center_y = dim_x//2, dim_y // 2
        radius = 104
        mask = (X-center_x)**2 + 1.1*(Y-center_y)**2 > radius**2
    else:
        print("ERROR")
    return mask


# In[6]:


#shuffle the data
random.shuffle(segmentation_data) #shuffle method


# In[7]:


ncol = 3
nrow = 3
plt.rcParams['figure.figsize'] = (ncol*3*2, nrow*3)
for k in range(ncol*nrow):
    study = segmentation_data[k][2]

    plt.subplot(nrow, ncol*2,2*k+1)
    raw = segmentation_data[k][0]
    raw_green = raw[:,:,1].astype(float)
    raw_green = raw_green / np.max(raw_green)
    plt.imshow(raw_green)
    plt.title(study)
    plt.axis("Off")

    plt.subplot(nrow, ncol*2,2*k+2)
    data_seg = segmentation_data[k][1] / np.max(segmentation_data[k][1])
    data_seg_ = 1+(data_seg> 0.5).astype(int)
    data_seg = np.copy(data_seg_)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    data_seg = cv2.dilate(data_seg.astype(np.uint8),kernel,iterations = 1)

    mask = create_mask(study)
    data_seg[mask]=0

    plt.imshow(data_seg, cmap="gray", alpha=1)
    plt.axis("Off")


# In[9]:


#prepare dataset
# scale everything in between 0 and 1 for the raw images
# transofmr to 0-1 valued images for segmentation
n_total = len(segmentation_data)
n_train = int(0.7 * n_total)  #70% are used as training data
n_test = n_total - n_train

X_train = np.zeros((n_train, dim_x, dim_y, 3))
Y_train = np.zeros((n_train, dim_x, dim_y))
X_test = np.zeros((n_test, dim_x, dim_y, 3))
Y_test = np.zeros((n_test, dim_x, dim_y))

for k in range(n_train):
    study = segmentation_data[k][2]
    x_train, y_train = segmentation_data[k][0], segmentation_data[k][1]
    x_train = x_train / 255.
    y_seg = y_train / float(np.max(y_train))
    y_seg = (y_seg> 0.5).astype(int)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    data_seg = cv2.dilate(y_seg.astype(np.uint8),kernel,iterations = 1)
    X_train[k,:,:,:] = np.copy(x_train)
    Y_train[k,:,:] = 1 + data_seg
    mask = create_mask(study)
    Y_train[k,mask]=0

for k in range(n_test):
    study = segmentation_data[n_train + k][2]
    x_test, y_test = segmentation_data[n_train + k][0], segmentation_data[n_train + k][1]
    x_test = x_test / 255.
    y_seg = y_test / float(np.max(y_test))
    y_seg = (y_seg> 0.5).astype(int)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    data_seg = cv2.dilate(y_seg.astype(np.uint8),kernel,iterations = 1)
    X_test[k,:,:,:] = x_test
    Y_test[k,:,:] = 1+data_seg
    mask = create_mask(study)
    Y_train[k,mask]=0


# In[27]:


img = X_train[0]
raw_img = (255*img).astype(np.uint8)
b,g,r = cv2.split(raw_img)
contrast_enhanced_green_fundus = clahe.apply(g)
plt.imshow(contrast_enhanced_green_fundus)


# In[28]:


#sanity check
ncol = 3
nrow = 3
plt.rcParams['figure.figsize'] = (ncol*3*2, nrow*3)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
for k in range(ncol*nrow):
    plt.subplot(nrow, ncol*2,2*k+1)
    #plt.imshow(np.mean(segmentation_data[k][0], axis=2)) #plot the mean of RGB channels
    raw_img = (255*X_train[k]).astype(np.uint8)
    b,green_fundus,r = cv2.split(raw_img)
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)
    plt.imshow(contrast_enhanced_green_fundus)
    #plt.imshow(X_train[k,:,:,:])
    plt.axis("Off")

    plt.subplot(nrow, ncol*2,2*k+2)
    plt.imshow(Y_train[k,:,:], cmap="gray")
    plt.axis("Off")


# # Unet

# In[30]:


def DataGen(X, y, batch_sz):
    """
    DATA AUGMENTATION:
    elastic transformation, rotation, pixel inetensity change, and all that
    """
    #alpha, alpha2, sigma = 10, 15, 50

    c = list(zip(X, y))
    #croph, cropw = crop_H, crop_W

    dim_x, dim_y, n_channel = X[0].shape

    #x_mesh, y_mesh = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    Xpatches = np.ndarray((batch_sz, dim_x, dim_y, 1), dtype=np.float32)
    ypatches = np.ndarray((batch_sz, dim_x, dim_y, 3), dtype=np.float32)

    #create a mask for later purposes
    #black_dx = 60
    #black_dy = 20
    #dark_mask = np.zeros((black_dx, black_dy))
    #for k in range(black_dy):
    #    dark_mask[:,k] = (np.abs(k-black_dy//2) / (black_dy/2.))**2

    image_index = 0
    batch_index = 0
    while True:
        #if the batch is full, release the batch
        if batch_index == batch_sz:
            batch_index = 0
            yield Xpatches, ypatches

        #if all images have been seen, start again from the beginning
        if image_index >= len(X):
            image_index = 0

        #do the following at the start of each epoch
        if image_index == 0:
            #shuffle the data
            random.shuffle(c)
            X, y = zip(*c)

            #below is used once per epoch for the elastic deformation
            #g_1d = signal.gaussian(300, sigma)
            #kernel_deform = np.outer(g_1d, g_1d)
            #dx = signal.fftconvolve(np.random.rand(*shape) * 2 - 1, kernel_deform, mode='same')
            #dy = signal.fftconvolve(np.random.rand(*shape) * 2 - 1, kernel_deform, mode='same')
            #dx = alpha * (dx - np.mean(dx)) / np.std(dx)
            #dy = alpha2 * (dy - np.mean(dy))/ np.std(dy)
            #indices_x, indices_y = x_mesh+dx, y_mesh+dy
            #indices_x_clipped = np.clip(indices_x, a_min=0, a_max=shape[1]-1)
            #indices_y_clipped = np.clip(indices_y, a_min=0, a_max=shape[0]-1)


        #img_height, img_width = X[image_index].shape[:-1]
        ax = np.zeros((dim_x, dim_y, n_channel))
        ay = np.zeros((dim_x, dim_y))
        ax = np.copy(X[image_index])
        ay[:,:] = np.copy(y[image_index])

        #flip with probability 1/2
        if random.randint(0, 1):
            ax = ax[:, ::-1, :]
            ay = ay[:, ::-1]

        #rotation + zoom
        if True:
            angle = np.random.uniform(low=0., high=360.)
            zoom = np.random.uniform(low=0.9, high=1.3)
            transform_matrix = cv2.getRotationMatrix2D((dim_x/2.0,dim_y/2.0),angle,zoom)
            ax = cv2.warpAffine(ax,transform_matrix,(dim_x, dim_y), flags = cv2.INTER_NEAREST)
            ay = cv2.warpAffine(ay,transform_matrix,(dim_x, dim_y), flags = cv2.INTER_NEAREST)



        #Intensity nonlinear shift
        if True:
            for channel in range(3):
                p = np.random.uniform(low=0.6, high=1.4)
                ax[:,:,channel] = ax[:,:,channel]**p
                a = np.random.uniform(low=0, high=0.1)
                b = np.random.uniform(low=0, high=0.1)
                ax[:,:,channel] = -a + (1+a+b) * ax[:,:,channel]
                ax = np.clip(ax, 0., 1.)

#         #add black boxes
#         if True:
#             for k in range(20):
#                 black_dx = 60
#                 black_dy = 20
#                 black_x, black_y = randint(0, ax.shape[0] -  black_dx), randint(0, ax.shape[1] -  black_dy)
#                 #ax[black_x:(black_x+black_dx), black_y:(black_y+black_dy), 0] = 0
#                 intensity_dark = np.random.uniform(low=0.2, high=0.8)
#                 window_to_darken = ax[black_x:(black_x+black_dx), black_y:(black_y+black_dy), 0]
#                 ax[black_x:(black_x+black_dx), black_y:(black_y+black_dy), 0] = window_to_darken * (intensity_dark + (1.-intensity_dark)*dark_mask)

#         #elastic deformation
#         if True:
#             ax[:,:,0] = ax[indices_y_clipped.astype(int),indices_x_clipped.astype(int),0]
#             for k in range(nb_classes):
#                 ay[:,:,k] = ay[indices_y_clipped.astype(int),indices_x_clipped.astype(int),k]


        #add Gaussian **additive** noise with variable intensity
        if True:
            intensity_noise = np.random.uniform(low=0, high=0.01)
            ax[:,:,:] = ax[:,:,:]  + intensity_noise*np.random.normal(loc=0, scale=1, size=(dim_x, dim_y, n_channel))

        #make sure that pixel intensity is in [0,1] and labels are integers
        ax = np.clip(ax, 0., 1.)
        #ay = np.clip(ay, 0., 1.)
        ay = ay.astype(int)

        raw_img = (255*ax).astype(np.uint8)
        b,green_fundus,r = cv2.split(raw_img)
        ax = clahe.apply(green_fundus) / 255.
        ax = np.clip(ax, 0., 1.)

        Xpatches[batch_index,:,:,0] = ax

        ypatches[batch_index] = 0.
        ypatches[batch_index,ay==0,0] = 1
        ypatches[batch_index,ay==1,1] = 1
        ypatches[batch_index,ay==2,2] = 1



        batch_index = batch_index + 1
        image_index = image_index + 1



# In[33]:


# Generate augmentation
batch_sz = 10
training_gen, val_gen = DataGen(X_train, Y_train, batch_sz), DataGen(X_test, Y_test, batch_sz)
im,seg = next(training_gen)
print("Dim(image)=",np.shape(im))
print("Dim(Label)=",np.shape(seg))


# In[35]:


ncol, nrow = 3,3
#plt.rcParams['figure.figsize'] = (*6, 2*4)
plt.rcParams['figure.figsize'] = (ncol*3*2,nrow*3)


for k in range(ncol * nrow):
    #print(k)
    im,seg = next(training_gen)
    plt.subplot(nrow, ncol*2,2*k+1)
    #raw_img = (255*im[k]).astype(np.uint8)
    #b,green_fundus,r = cv2.split(raw_img)
    #contrast_enhanced_green_fundus = clahe.apply(green_fundus)
    #plt.imshow(contrast_enhanced_green_fundus)
    plt.imshow(im[k,:,:,0])
    plt.axis("off")

    plt.subplot(nrow, ncol*2,2*k+2)
    plt.imshow(seg[k,:,:,:], cmap='gray')
    plt.axis("off")


# In[38]:


seg[0,100,100]


# In[39]:


def standard_blocks(layer_input, nb_filters, conv_size, dillatation, nb_blocks):
    y = Conv2D(nb_filters, conv_size, dilation_rate = dillatation, padding='same')(layer_input)
    #y = BatchNormalization()(y)
    y = Activation('elu')(y)
    for _ in range(nb_blocks-1):
        y = Conv2D(nb_filters, conv_size, dilation_rate = dillatation, padding='same')(y)
        #y = BatchNormalization()(y)
        y = Activation('elu')(y)
    return y

def residual_blocks(layer_input, nb_filters, conv_size, dillatation, nb_blocks):
    residual = Conv2D(nb_filters, (1, 1), padding='same', use_bias=False)(layer_input)
    y = Conv2D(nb_filters, conv_size, dilation_rate = dillatation, padding='same')(layer_input)
    #y = BatchNormalization()(y)
    y = Activation('elu')(y)
    for _ in range(nb_blocks-1):
        y = Conv2D(nb_filters, conv_size, dilation_rate = dillatation, padding='same')(y)
        #y = BatchNormalization()(y)
        y = Activation('elu')(y)
    y = layers.add([y, residual])
    return y

def basic_unet(dim_x, dim_y, nb_classes):
    inputs = Input((dim_x, dim_y, 1))
    #
    conv1 = standard_blocks(layer_input = inputs, nb_filters = 16, conv_size = (3,3),
                                  dillatation = 1, nb_blocks=2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = residual_blocks(layer_input = pool1, nb_filters = 32, conv_size = (3,3),
                                  dillatation = 1, nb_blocks=2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = residual_blocks(layer_input = pool2, nb_filters = 64, conv_size = (3,3),
                                  dillatation = 1, nb_blocks=2)
    #pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #
    #conv5 = residual_blocks(layer_input = pool3, nb_filters = 64, conv_size = (3,3),
    #                              dillatation = 1, nb_blocks=2)

    #up7 = concatenate([UpSampling2D()(conv5), conv3], axis=3)
    #conv7 = residual_blocks(layer_input = up7, nb_filters = 32, conv_size = (3,3),
    #                             dillatation = 1, nb_blocks=2)

    up8 = concatenate([UpSampling2D()(conv3), conv2], axis=3)
    conv8 = residual_blocks(layer_input = up8, nb_filters = 16, conv_size = (3,3),
                                  dillatation = 1, nb_blocks=2)
    #
    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=3)
    conv9 = residual_blocks(layer_input = up9, nb_filters = 16, conv_size = (3,3),
                                  dillatation = 1, nb_blocks=2)
    #
    conv10 = Conv2D(nb_classes, (1, 1))(conv9)
    #out = Activation(softmax_custom) (conv10)
    out = Activation('softmax')(conv10)
    model = Model(inputs=[inputs], outputs=[out])
    return model


# In[40]:


def create_basic_unet():
    initial_LR = .001 # starting learning rate
    adam = Adam(lr=initial_LR)

    print('Compiling model...')
    #model = ResUnet_D_bilinear_nounet_v3(crop_W, crop_H, nb_classes)
    nb_classes = 3
    model = basic_unet(dim_x, dim_y, nb_classes)
    #model.compile(optimizer=adam, loss=jacc_coeff_loss)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    #model.compile(optimizer=sgd, loss=jacc_coeff_loss, metrics=['accuracy'],loss_weights=[0.4])
    print('Model compiled.')

    return model


# In[41]:


model = create_basic_unet()
model.summary()


# In[42]:


train_mini_batch_sz = 32
val_mini_batch_sz = 5
num_train_batches, num_val_batches = len(X_train) // train_mini_batch_sz, len(X_test) // val_mini_batch_sz
training_gen = DataGen(X_train, Y_train, train_mini_batch_sz)
val_gen = DataGen(X_test, Y_test, train_mini_batch_sz)




# In[43]:


epochs=200
#with tf.device('/gpu:3'):
fit_history = model.fit_generator(generator=training_gen,
                    steps_per_epoch=num_train_batches,
                    epochs=epochs,
                    validation_data=val_gen,
                    validation_steps=num_val_batches,
                    verbose=1, use_multiprocessing=True)


# In[44]:


plt.plot(fit_history.history["loss"], "b-")
plt.plot(fit_history.history["val_loss"], "r-")


# In[45]:


def reduce_spluttering(seg_map, threshold = 50):
    """
    get rid of small connected component
    """
    ret, labels = cv2.connectedComponents( seg_map.astype(np.uint8) )
    for k in range(ret):
        size_component = np.sum(labels==k)
        if size_component <= threshold:
            seg_map[labels==k] = 0
    return seg_map


# In[46]:


im,seg = next(val_gen)
seg_val = model.predict(im)


# In[47]:


ncol, nrow = 3,3
#plt.rcParams['figure.figsize'] = (*6, 2*4)
plt.rcParams['figure.figsize'] = (ncol*3*2,nrow*3)


for k in range(ncol * nrow):
    #print(k)
    plt.subplot(nrow, ncol*2,2*k+1)
    #raw_img = (255*im[k]).astype(np.uint8)
    #b,green_fundus,r = cv2.split(raw_img)
    #contrast_enhanced_green_fundus = clahe.apply(green_fundus)
    #plt.imshow(contrast_enhanced_green_fundus)
    plt.imshow(im[k,:,:,0])
    plt.axis("off")

    plt.subplot(nrow, ncol*2,2*k+2)
    seg_map = np.argmax(seg_val[k], axis=2)==2
    seg_map = reduce_spluttering(seg_map, threshold = 50)
    plt.imshow(seg_map, cmap='gray')
    plt.axis("off")


# In[71]:


ret, labels = cv2.connectedComponents( (np.argmax(seg_val[k], axis=2)==2).astype(np.uint8) )


# In[ ]:


plt.imshow(seg_val[k,:,:,1]>0.3)


# In[ ]:


from tensorflow.keras.models import model_from_json


# In[ ]:


model_json = model.to_json()


# In[ ]:


with open("fundusweights.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("fundusweights.h5")
print("Saved model to disk")


# In[ ]:


json_file = open('fundusweights.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("fundusweights.h5")
print("Loaded model from disk")


# In[ ]:
