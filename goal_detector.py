import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import requests
import cv2
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

### FLAGS ###
create_random_data_flag = False
show_sample_data_flag = False
show_results_flag = True
save_results_flag = False

batch_size = 2
img_height = 255
img_width = 255

data_dir = r"C:\Users\thorl\OneDrive - Danmarks Tekniske Universitet\thor\3. Semester\Deep learning\project\DP-project\Data"

if create_random_data_flag is True:
    ## Creating random data
    path = r"C:\Users\thorl\OneDrive - Danmarks Tekniske Universitet\thor\3. Semester\Deep learning\project\DP-project\Data\Not football goal"

    for i in range(200):
        url = "https://picsum.photos/200/200/?random"
        response = requests.get(url)
        if response.status_code == 200:
            file_name = 'not_nicolas_{}.jpg'.format(i)
            file_path = path + "/" + file_name
            with open(file_path, 'wb') as f:
                print("saving: " + file_name)
                f.write(response.content)

### Dividing data into training and validation
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = ['Football goal', 'Not football goal']

### Visualise some of the training data
if show_sample_data_flag is True:
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.show()

### Setting cache and prefetch, to optimise performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
  
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

### Test model on sample image
print("Testing on VEO DATA ")
img=cv2.imread(r"C:\Users\thorl\OneDrive - Danmarks Tekniske Universitet\thor\3. Semester\Deep learning\project\DP-project\sample_image_goal_from_ts.png")
image=cv2.resize(img, (img_height,img_width))
image=np.expand_dims(image, axis=0) #input shape needs to be (1,width,height,channels)
predictions = model.predict(image)
class_index = np.argmax(predictions)
print("Model predictios :: ",predictions, " :: ",class_names[class_index])


if show_results_flag is True:
    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    if save_results_flag is True:
        plt.savefig('results.png')
    plt.show()

### Saving model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
save = input("Want to save model? (y or n \n")
if save == 'n':
    os.remove("model.json")
    os.remove("model.h5")
elif save == 'y':
    print("Model files saved")
else:
    print("invalid input")
