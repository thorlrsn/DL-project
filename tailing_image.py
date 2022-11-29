# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

import matplotlib.pyplot as plt
from PIL import Image 
import numpy as np
import math
# import random
from typing import List
from tensorflow.keras.models import Sequential, model_from_json
import tensorflow as tf
import sys, os
import heapq

# img = Image.open(r"Video_frames\left\video_12_00_26_20_left001.png")
img = Image.open(r"Video_frames\left\video_9_00_45_25_left001.png")
# img = Image.open(r"Video_frames\left\video_10_00_45_25_left001.png")
# img = Image.open(r"Video_frames\right\video_2_00_26_20_right001.png")


image = np.array(img)
# img.show()

frame_start_x = []
frame_start_y = []
frame_end_x = []
frame_end_y = []
prob = []
# 1261 331
# 1507 411
imgsize = [170 ,320]
grid_step = 20
print(image.shape)

image = image[200:880,300:1936]
img1 = Image.fromarray(image, 'RGB')
# img1.show()

path_to_dir = r"Models\outputs7"
path_to_model = path_to_dir + '\model.json'
path_to_weight = path_to_dir + '\model.h5'

json_file = open(path_to_model, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(path_to_weight)
print("Loaded model from disk")

### Compile the model
loaded_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
blockPrint()
for x in range(math.floor((image.shape[1]/grid_step))-math.ceil(((imgsize[1]/grid_step)))):
    for y in range(math.floor((image.shape[0]/grid_step))-math.ceil(((imgsize[0]/grid_step)))):
        # print(x,y)
        frame_start_x.append(grid_step*x)
        frame_start_y.append(grid_step*y)
        frame_end_x.append(grid_step*x+imgsize[1])
        frame_end_y.append(grid_step*y+imgsize[0])
        color = image[(grid_step*y):(grid_step*y+imgsize[0]), (grid_step*x):(grid_step*x+imgsize[1])]
        color=np.expand_dims(color, axis=0) #input shape needs to be (1,width,height,channels)
        # print(color.shape)
        if (color.shape[1]<150 or color.shape[2]<150):
            continue
        predictions = loaded_model.predict(color)
        prob.append(predictions[0][0])
enablePrint()
print(len(prob))
print("zero sample: ", prob[0])
print("max element", max(prob))
# maximum = heapq.nlargest(5, prob)
maximum = max(prob)
# for i in range(len(maximum)):
index = prob.index(maximum)
print(index)
print("dimensions: ",frame_start_y[index], frame_end_y[index], frame_start_x[index], frame_end_x[index])
# print(frame_start_y[index])
color = image[frame_start_y[index]:frame_end_y[index], frame_start_x[index]:frame_end_x[index]]
print("color shape", color.shape)
# color = image[321:421, 1261:1507 ]
img1 = Image.fromarray(color, 'RGB')
img1.show()
