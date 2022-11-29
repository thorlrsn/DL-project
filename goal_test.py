import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
# import pathlib
# import requests
import cv2
# from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, model_from_json
# from tensorflow.keras.layers import Dense

### Load model from folders
path_to_dir = r"C:\Users\thorl\OneDrive - Danmarks Tekniske Universitet\thor\3. Semester\Deep learning\project\DP-project\Models\outputs5"
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

folder_dir = r"C:\Users\thorl\OneDrive - Danmarks Tekniske Universitet\thor\3. Semester\Deep learning\project\DP-project\test_data\test1"
class_names = ['Football goal', 'Not football goal']

for images in os.listdir(folder_dir):
    img_path = folder_dir + "\\" + images
    # print(img_path)
    ## Test on single image
    image=cv2.imread(img_path)
    image=cv2.resize(image, (100,100))
    image=np.expand_dims(image, axis=0) #input shape needs to be (1,width,height,channels)
    predictions = loaded_model.predict(image)
    class_index = np.argmax(predictions)
    # if class_index == 0:
    print(images)
    print("Model predictios :: ",predictions, " :: ",class_names[class_index])
