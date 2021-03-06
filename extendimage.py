#!/usr/local/bin/python3
import numpy as np
import random
import cv2

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

im = cv2.imread('I.jpg')
blk = np.zeros((512,512,3), np.uint8)
stretch = cv2.resize(im,(512,512))
#pad = cv2.copyMakeBorder( src, top, bottom, left, right, borderType)
h,w,c = im.shape
print(w,h,c)
wid=int((512-w)/2)
hig=int((512-h)/2)
noise_img[hig:hig+im.shape[0],wid:wid+im.shape[1]]=im
hsv=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
cv2.imshow("result",noise_img)

