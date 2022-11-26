import numpy as np
import math
# https://mayavan95.medium.com/3d-position-estimation-of-a-known-object-using-a-single-camera-7a82b37b326b

# https://math.stackexchange.com/questions/1683403/projecting-from-one-2d-plane-onto-another-2d-plane
# probaly need to work on the vertical pole to get the pixel density. ATM the position estimation is sketchy
int_left = [[1933.1962437237123, 0.0, 1836.8866309535802],
    [0.0, 1933.74682448549, 878.6808564823706], [0.0, 0.0, 1.0]]

Goalcorner_left = [3271, 723]
Goalcorner_right = [2766, 627]
Pim = [[3271],[723],[.1]]

world_coord = np.matmul(np.linalg.inv(np.matrix(int_left)),np.matrix(Pim))
print(world_coord)

fx = int_left[0][0]
fy = int_left[1][1]
cx = int_left[0][2]
cy = int_left[1][2]
wid = 10 #football goal height in pixels, substitute with object heigt
d_pix = np.sqrt((Goalcorner_left[0] - Goalcorner_right[0])**2 + (Goalcorner_left[1] - Goalcorner_right[1])**2)  # object height in pixels

print(d_pix)

Z = (fx * wid) / (d_pix)
X = ((Pim[0][0]-cx)*Z)/fx
Y= ((Pim[1][0]-cy)*Z)/fy

X_cam = [[X],[Y],[Z]]
print(X_cam)