import os
import random
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, Cropping2D, Dropout, Flatten, Reshape, Activation
from keras.models import Model, Sequential
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
#additional packages (in case native keras won't work)
#from scipy.misc import imsave, imread #imsave("image.png", some_array)
from PIL import Image

from google.colab import files


##########uncomment the lines below to check and ensure any GPU available
#!pip install -q Pillow
#!pip install -q tensorflow-gpu
#K.tensorflow_backend._get_available_gpus()
#tf.test.gpu_device_name()
#with tf.device("/gpu:0")
#with tf.device("/cpu:0")
#model.summary()

##########upload/download files from colab VM
#uploaded = files.upload()
#files.download('example.txt')


def FromImgConvert(img_path): #returns a numpy array
    return img_to_array(load_img(img_path)) / 255

def Sample(L, n):
    return np.array(random.sample(L, n))


work_dir = "/content/gdrive/My Drive/CloudAI"
dataY_dir = work_dir + "/FacesColor"
dataTrain_dir = work_dir + "/FacesGray"
nEpochs = 1000
Batch_size = 64
Alpha = 0.1
Beta = 0.15
Check_point = 10
limiter = 4900 #70 for test

os.chdir(dataY_dir)
data_list = os.listdir(os.curdir)
dataY = []
fcount = 0
for file in data_list:
    fcount += 1
    dataY.append(FromImgConvert(file))
    print(fcount)
    if fcount > limiter:
        break

datasetY = np.array(dataY)

os.chdir(dataTrain_dir)
data_list = os.listdir(os.curdir)
dataTrain = []
fcount = 0
for file in data_list:
    fcount += 1
    dataTrain.append(FromImgConvert(file))
    print(fcount)
    if fcount > limiter:
        break

datasetX = np.array(dataTrain)

os.chdir(work_dir)


#defining autoencoder model

AEmodel = Sequential()

AEmodel.add(Conv2D(filters = 16, input_shape = (128, 128, 3), kernel_size = (3, 3), padding = "same", activation = "relu"))
AEmodel.add(Dropout(Beta))
AEmodel.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu"))
AEmodel.add(Dropout(Beta))
AEmodel.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu"))
AEmodel.add(Dropout(Beta))
AEmodel.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu"))
AEmodel.add(Dropout(Beta))
AEmodel.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu"))
AEmodel.add(Dropout(Beta))
AEmodel.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu"))
AEmodel.add(Dropout(Beta))
AEmodel.add(Conv2D(filters = 16, kernel_size = (3, 3), padding = "same", activation = "relu"))
AEmodel.add(Dropout(Beta))
AEmodel.add(Conv2D(filters = 3, kernel_size = (3, 3), padding = "same", activation = "sigmoid"))

AEmodel.compile(optimizer = "adam", loss = "mse", metrics = ["accuracy"])
AEmodel.summary()

for E in range(nEpochs):
	AEmodel.fit(datasetX, datasetY, shuffle = True, epochs = 1, batch_size = Batch_size)
	print(E)

	if E % Check_point == 0:
		AEmodel.save_weights(str(E) + ".h5")
		save_img(str(E) + ".png", 255 * AEmodel.predict(Sample(dataTrain, 2))[0])