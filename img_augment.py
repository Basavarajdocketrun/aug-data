#!/usr/local/bin/python3
import numpy as np
import random
import cv2
import argparse


# parse command line
parser = argparse.ArgumentParser()
parser.add_argument("imgtype", type=str, default="squash", help="squash  or squeez")

opt = parser.parse_known_args()[0]


def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
image = np.zeros((512,512,3), np.uint8)# Only for grayscale image
noise_img = sp_noise(image,0.5)

im = cv2.imread('B1.jpg')
blk = np.zeros((512,512,3), np.uint8)
stretch = cv2.resize(im,(512,512))
h,w,c = im.shape
print(w,h,c)
if h  > 512:
    h1 =512
else:
    h1 = h
if w > 512:
    w1 =512
else:
    w1 =w

crop = im[0:0+h1, 0:0+w1]

im = crop
h,w,c = im.shape
print(w,h,c)
try:

    wid=int((512-w)/2)
    hig=int((512-h)/2)
    if opt.imgtype=='fillblack':
        blk[hig:hig+im.shape[0],wid:wid+im.shape[1]]=im
        cv2.imwrite("result4.png",blk)
    if opt.imgtype=='fillnoise':
        noise_img[hig:hig+im.shape[0],wid:wid+im.shape[1]]=im
        cv2.imwrite("result4.png",noise_img)
    if opt.imgtype=='stretch':
        cv2.imwrite("result4.png",stretch)

except:
    pass
