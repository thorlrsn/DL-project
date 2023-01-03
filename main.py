import cv2
import argparse
import numpy as np
from roboflow import Roboflow
import open3d as o3d; 

from skimage import color, io, measure, img_as_ubyte
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

import yolov7 

def show_in_moved_window(win_name, img, x, y, cmap=None):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    img = cv2.resize(img, (660, 440)) 
    if cmap == None:
        cv2.imshow(win_name, img)
    else:
        cv2.imshow(win_name, img, cmap=cmap)
        


# load yolov7 model
model = yolov7.createModel()


# frames = frames[80:]
# for i in range(len(color_img)):
for i in range(1):
    print('i: ', i)
    predictions = []

    # path = 'video_12_00_26_20_left001.png'
    path = 'yolo_image_for_model.png'
    # print(path)
    image = cv2.imread(path)
    b,g,r = cv2.split(image)       # get b,g,r
    image = cv2.merge([r,g,b])

    print('printing shape')
    print(image.shape)


    ### find with yolov7 
    yolo_pred = yolov7.predictwyolo(model, image)
    predictions = predictions + yolo_pred

    print('predictions: ', predictions)
    image = yolov7.drawClassifiedObjects(image, predictions)
    
    

    # Display the resulting frame
    show_in_moved_window('Input', image, 0, 50)

    cv2.imwrite('yolo_results.png', image) 
        
    cv2.waitKey()



# close all windows
cv2.destroyAllWindows()
