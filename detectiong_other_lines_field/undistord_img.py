lens_left2camera = (0.737387190340703, 0.2162742559576276, -0.6399106013588293, -0.03387933088133182, -0.022444339259397095, 0.9546775475461002, 0.29679459537965447, 0.0016336334162659265, 0.6750973138419772, -0.20449016216272686, 0.7088211272401583, 0.0, 0.0, 0.0, 0.0, 1.0)
# "lens_right2camera": "0.7380913511385816, -0.2152812239869899, 0.6394334617245852, 0.03387933088133182, 0.023366849834891013, 0.9553167079176846, 0.29465908759464854, -0.0016336334162659265, -0.6742960386234051, -0.2025437784084729, 0.7101414437454011, 0.0, 0.0, 0.0, 0.0, 1.0"
distortion = [3.670680675357433, 1.1612862793879029, 0.0, 0.0, 0.010907748126931006, 4.029813052720811, 2.357017484561505, 0.16278370795452082, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
print(len(lens_left2camera))

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

#crop image in after undistort

x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
plt.figure(figsize=(10,10))
plt.imshow(dst[...,[2,1,0]])
