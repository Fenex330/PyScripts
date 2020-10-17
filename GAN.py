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

def GenerateNoise(dim, n):
    return np.random.normal(0, 1, size = (n, dim * 3)) #.tolist()

def Sample(L, n):
    return np.array(random.sample(L, n))


work_dir = "/content/gdrive/My Drive/CloudAI"
data_dir = work_dir + "/GlyphDatasetE/GlyphDatasetE/PreprocessedAutomated"
nEpochs = 1000
Batch_size = 32
noise_dim = 100 #must be square of an integer
Alpha = 0.1
Beta = 0.15
Check_point = 10
limiter = 200
#(75, 50, 3)

os.chdir(data_dir)
ds = os.listdir(os.curdir)
data = []
fcount = 0
for file in ds:
    fcount += 1
    data.append(FromImgConvert(file))
    print(fcount)
    if fcount > limiter:
        break
dataset = data
os.chdir(work_dir)


#defining generator
Generator = Sequential()

Generator.add(Dense(noise_dim * 3, input_dim = noise_dim * 3))
Generator.add(Reshape((int(noise_dim ** 0.5), int(noise_dim ** 0.5), 3)))
Generator.add(Conv2DTranspose(filters = 64, kernel_size = (3, 3), strides = (2, 2), padding = "same", activation = "relu"))
#Generator.add(LeakyRelu(Alpha))
#Generator.add(Dropout(Beta)
Generator.add(Conv2DTranspose(filters = 32, kernel_size = (3, 3), strides = (2, 2), padding = "same", activation = "relu"))
Generator.add(Conv2DTranspose(filters = 16, kernel_size = (3, 3), strides = (2, 2), padding = "same", activation = "sigmoid"))
Generator.add(Conv2D(filters = 3, kernel_size = (3, 3), padding = "same", activation = "relu"))
Generator.add(Cropping2D(((5, 0), (15, 15))))

Generator.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#defining discriminator
Discriminator = Sequential()

Discriminator.add(Conv2D(filters = 2, input_shape = (75, 50, 3), kernel_size = (3, 3), padding = "valid", activation = "relu"))
Discriminator.add(MaxPooling2D((2, 2)))
Discriminator.add(Conv2D(filters = 16, kernel_size = (3, 3), padding = "valid", activation = "relu"))
Discriminator.add(MaxPooling2D((2, 2)))
Discriminator.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = "valid", activation = "relu"))
Discriminator.add(MaxPooling2D((2, 2)))
Discriminator.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = "valid", activation = "relu"))
Discriminator.add(Flatten())
Discriminator.add(Dense(1, activation = "sigmoid"))

Discriminator.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


Discriminator.trainable = False

GAN = Sequential()
GAN.add(Generator)
GAN.add(Discriminator)

GAN.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

Generator.summary()
Discriminator.summary()
GAN.summary()


for epoch in range(nEpochs):
    cur_outs = []
    GN = GenerateNoise(noise_dim, Batch_size)
    cur_outs = Generator.predict(GN)
    cur_outs_np = np.array(cur_outs)
    cur_dataset = np.concatenate((Sample(dataset, Batch_size), cur_outs_np))
    cur_dataset_labels = np.concatenate((np.ones(Batch_size), np.zeros(Batch_size)))

    Discriminator.trainable = True
    Discriminator.fit(cur_dataset, cur_dataset_labels, shuffle = True, epochs = 1, batch_size = 1)
    Discriminator.trainable = False

    train_noise = GenerateNoise(noise_dim, Batch_size * 2)
    train_labels = np.ones(Batch_size * 2)
    GAN.train_on_batch(train_noise, train_labels)

    if epoch % Check_point == 0:
        save_img(str(epoch) + ".png", 255 * Generator(np.random.normal(size = (noise_dim, noise_dim, 3))))
        #plt.plot()
        #plt.show()
        Generator.save_weights(str(epochs) + ".h5")
