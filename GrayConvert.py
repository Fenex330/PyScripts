import os
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img

def Convert(nplist):
    a, b, c = nplist.shape
    outlist = np.ones((a, b, c), dtype = np.int16)
    for x in range(a):
        for y in range(b):
            outlist[x, y, 0] = (nplist[x, y, 0] + nplist[x, y, 1] + nplist[x, y, 2]) / 3
            outlist[x, y, 1] = (nplist[x, y, 0] + nplist[x, y, 1] + nplist[x, y, 2]) / 3
            outlist[x, y, 2] = (nplist[x, y, 0] + nplist[x, y, 1] + nplist[x, y, 2]) / 3
    return outlist

fileList = os.listdir(os.curdir)

counter = 1
for file in fileList:
    if not (file == "GrayConvert.py"):
        img = Convert(img_to_array(load_img(file)))
        print(img.shape)
        print(counter)
        save_img(file, img)
        counter = counter + 1