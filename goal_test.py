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
# path_to_dir = r"C:\Users\thorl\OneDrive - Danmarks Tekniske Universitet\thor\3. Semester\Deep learning\project\DP-project\Models\outputs4"
path_to_dir = r"Models/outputs5"
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

# folder_dir = r"C:\Users\thorl\OneDrive - Danmarks Tekniske Universitet\thor\3. Semester\Deep learning\project\DP-project\test_data\test2"
folder_dir = r"test_data/test2"
class_names = ['Football goal', 'Not football goal']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
ax = axes.ravel()

i = 1
for images in os.listdir(folder_dir):
    print(i, images)
    img_path = folder_dir + "\\" + images
    # print(img_path)
    ## Test on single image
    image=cv2.imread(img_path)
    image_plot=image.copy()
    image=cv2.resize(image, (150,150))
    image=np.expand_dims(image, axis=0) #input shape needs to be (1,width,height,channels)
    predictions = loaded_model.predict(image)
    class_index = np.argmax(predictions)
    # print('here')
    if class_index == 0:
        print(images)
        # print("Model predictios :: ",predictions, " :: ",class_names[class_index])



    ax[i-1].imshow(image_plot)
    ax[i-1].set_title(f'{class_names[class_index]}, values: {predictions}')
    ax[i-1].axis('image')


    if i % 6 == 0:
        plt.tight_layout()
        plt.show()
        cv2.waitKey()
        i = 0
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        ax = axes.ravel()

    i += 1




    

