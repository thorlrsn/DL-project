import cv2
import argparse
import numpy as np
from pathlib import Path

import time
import os

import random

from skimage import color, io, measure, img_as_ubyte
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join



path = ''
i = 0
while(True):
# for i in range(100):
    path = "./images" + str(i)
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        break
    i += 1


# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture('veo_example.mp4')
ret, frame = cap.read()


n_frames = 0

# while (True):
for i in range(10):
    # Capture frame-by-frame
    ret, new_frame = cap.read()
    if not ret:
        break
    n_frames = n_frames + 1
    predictions = []

    image = new_frame

    # temp = int(random.gauss(64, 10))
    w = 256
    h = 256
    # print(image.shape)

    x_ran = random.randint(0, image.shape[1]-w)
    y_ran = random.randint(0, image.shape[0]-h)

    # print(y_ran,y_ran+h, 'and', x_ran, x_ran+w)
    
    ran_part = image[y_ran:y_ran+h, x_ran:x_ran+w]


    img_path = path + "/image" + str(n_frames) + ".jpg"
    cv2.imwrite(img_path, ran_part)

    # display output image    
    # cv2.imshow("object detection", ran_part)


    # print('frame no:', n_frames)
    

    cv2.waitKey()

print('total frames:', n_frames)

# Delete all created frames
print("Deleting captures...")
# release the video capture object
cap.release()

# Closes all the windows currently opened.
cv2.destroyAllWindows()
