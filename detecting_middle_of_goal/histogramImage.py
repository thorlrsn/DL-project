
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import time
from scipy import sparse
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
import cv2

from PIL import Image, ImageFilter, ImageOps



def histogram_stretch(img_in):
    """
    Stretches the histogram of an image
    :param img_in: Input image
    :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255
    """
    # img_as_float will divide all pixel values with 255.0
    img_float = img_as_float(img_in)
    min_val = img_float.min()
    max_val = img_float.max()
    min_desired = 0.0
    max_desired = 1.0
    img_out = (img_float-min_val)*(1/(max_val-min_val))
    return img_as_ubyte(img_out)
    
def interval_fun (angle,val):
    return (angle>=val-0.05) & (angle<=val+0.05)


# path = "goal2.png"

# goal8
# x_start, x_end = 900, 1057 
# y_start, y_end = 348, 418

# goal9 1 left fram001
# x_start, x_end = 1272, 1558 
# y_start, y_end = 605, 728


# goal10 1 right fram001
x_start, x_end = 2069, 2410
y_start, y_end = 573, 705


# goal11 12 right fram001
# x_start, x_end = 2313, 2581
# y_start, y_end = 552, 678


# path = 'sample_image_from_ts.png'
# path = '../Video_frames/left/video_1_00_26_20_left001.png'
# path = '../Video_frames/left/video_1_00_26_20_left001.png'
# path = '../Video_frames/right/video_12_00_46_13_right001.png'

color = cv2.imread(path)
# color = io.imread(path)
# color = io.imread("frame10000.jpg")


print(np.shape(color))
color = color[y_start:y_end, x_start:x_end]
# cv2.imwrite('goal11.png', color)
# io.imshow(color)
# io.show()



image_org = Image.open(path)
image = image_org.convert("L")
image_stretch = np.array(image)
image_stretch = image_stretch[y_start:y_end, x_start:x_end]

image_stretch = histogram_stretch(image_stretch)


edges = cv2.Canny(image_stretch,70,255)


# n, b, patches = plt.hist(image1.ravel(), bins=100)
# plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 8))
ax = axes.ravel()

# ax[0].imshow(color, cmap='gray')

ax[0].imshow(image_stretch)
ax[0].set_title('Images stretched')
ax[0].axis('image')

ax[1].hist(image_stretch.ravel(), bins=100)
ax[1].set_title('Histogram')
ax[1].axis('image')


ax[2].imshow(edges)
ax[2].set_title('Edges')
ax[2].set_axis_off()

plt.tight_layout()
plt.show()

