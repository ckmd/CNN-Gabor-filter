import cv2, os, glob, re, function
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def splitstring(word):
    x = re.split("_", word)
    return x[0]

def getName(word):
    if(word[0:4] == 'face'):
        return 0
    else:
        return 1

# read several image
img_dir = "data" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*png')
files = glob.glob(data_path)
detected = 0
data = []
label = []

for f1 in files:
    image = cv2.imread(f1)
    base = os.path.basename(f1)
    base = os.path.splitext(base)
    # title = splitstring(base[0])
    title = getName(base[0])
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_pyramid = [image]
    layer = image
    layer = cv2.resize(layer,(92,112))

    height = layer.shape[0]
    width = layer.shape[1]
    pad = 16

    new_layer = np.ones([height + 2*pad, width + 2*pad])
    new_layer += 128
    for i in range(height):
        for j in range(width):
            new_layer[i + pad][j + pad] = layer[i][j]
    # plt.imshow(new_layer, cmap='gray')
    # plt.show()
    # cv2.waitKey(1000)

    # for i in range(1):
    #     layer = cv2.pyrDown(layer)
    #     gaussian_pyramid.append(layer)
    
    data.append(new_layer)
    label.append(title)

data = np.array(data)
label = function.labelling(label , 2)
print("read data complete", data.shape, label.shape)
# print(label)
# cv2.imshow('data',data[0])
# cv2.waitKey(1000)