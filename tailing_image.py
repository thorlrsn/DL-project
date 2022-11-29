import matplotlib.pyplot as plt
from PIL import Image 
import numpy as np
import math
import random
from typing import List


img = Image.open('field_frame.jpg')
image = np.array(img)

# img.show()

frame_start_x = []
frame_start_y = []
frame_end_x = []
frame_end_y = []
prob = []
# 1261 331
# 1507 411
imgsize = [260 ,110]
grid_step = 10
print(image.shape)
for x in range(math.floor(image.shape[0]/grid_step-imgsize[0]/grid_step)):
    for y in range(math.floor((image.shape[1])/grid_step-imgsize[1]/grid_step)):
        # print(x,y)
        frame_start_x.append(10*x)
        frame_start_y.append(10*y)
        frame_end_x.append(10*x+260)
        frame_end_y.append(10*y+110)
        prob.append (random.random())

index = prob.index(max(prob))
print(index)
# print(frame_start_y[index])
color = image[frame_start_y[index]:frame_end_y[index], frame_start_x[index]:frame_end_x[index]]
# print(color.shape)
# color = image[321:421, 1261:1507 ]
img1 = Image.fromarray(color, 'RGB')
img1.show()
