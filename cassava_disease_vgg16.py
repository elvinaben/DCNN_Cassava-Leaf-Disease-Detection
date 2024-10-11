from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import cv2
plt.style.use("ggplot")


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm_notebook, tnrange
import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from skimage import io
import random
from sklearn.preprocessing import StandardScaler, normalize
from IPython.display import display


! kaggle datasets download nirmalsankalana/cassava-leaf-disease-classification/

!unzip '/content/cassava-leaf-disease-classification.zip' -d '/content/data/'



data_map_train = []
data_map_test = []
#for sub_dir_path in glob.glob("../lgg-mri-segmentation/kaggle_3m/"+"*"):


#train
path = "/content/data/data"
for lbl in os.listdir(path):
  if(lbl == ".DS_Store"):
    continue
  for imageName in os.listdir(path+"/"+lbl):
    imagePath = path+"/"+lbl+"/"+imageName;
    label = lbl
    data_map_train.extend([imagePath,label])


df_train = pd.DataFrame({"image_path" : data_map_train[::2], "label" : data_map_train[1::2]})
df_train


df_train = df_train.sample(frac=1).reset_index(drop=True)
df_train

"""# SPLIT DATASETS FOR TRAINING AND TESTING"""

train, valid = train_test_split(df_train, test_size=0.2)

test, valid = train_test_split(valid, test_size=0.5)


# datagen = ImageDataGenerator(rescale=1./255., validation_split=0.1)
datagen = ImageDataGenerator(rescale=1./255.)

train_generator = datagen.flow_from_dataframe(train,
                                              directory='./',
                                              x_col='image_path',
                                              y_col='label',
                                              class_mode='categorical',
                                              batch_size=16,
                                              shuffle=True,
                                              target_size=(224, 224)
                                             )
test_generator = datagen.flow_from_dataframe(test,
                                              directory='./',
                                              x_col='image_path',
                                              y_col='label',
                                              class_mode='categorical',
                                              batch_size=16,
                                              shuffle=False,
                                              target_size=(224, 224)
                                             )

valid_generator = datagen.flow_from_dataframe(valid,
                                                  directory='./',
                                                  x_col='image_path',
                                                  y_col='label',
                                                  class_mode='categorical',
                                                  batch_size=16,
                                                  shuffle=False,
                                                  target_size=(224, 224)
                                                 )

valid_generator.labels

"""# vgg PRETRAINED MODEL USING IMAGENET"""

from tensorflow.keras.applications.vgg16 import VGG16
vgg_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224,3)))

# Freeze the layers
for layer in vgg_model.layers:
    layer.trainable = False

"""# ADD THE HEAD USING VGG BASED CLASSIFIER"""

x = Flatten()(vgg_model.output)
prediction = Dense(5, activation='softmax')(x)
model = Model(inputs=vgg_model.input, outputs=prediction)
model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics= ["accuracy"]
             )
#model.summary()

earlystopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              verbose=1,
                              patience=5
                             )
checkpointer = ModelCheckpoint(filepath="vgg-weights.h5",
                               verbose=1,
                               save_best_only=True
                              )

callbacks = [checkpointer, earlystopping]

h = model.fit(train_generator,
              steps_per_epoch= train_generator.n // train_generator.batch_size,
              epochs = 10,
              validation_data= valid_generator,
              validation_steps= valid_generator.n // valid_generator.batch_size,
              callbacks=callbacks)

model_json = model.to_json()
with open("vgg-model.json", "w") as json_file:
    json_file.write(model_json)

h.history.keys()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(h.history['loss']);
plt.plot(h.history['val_loss']);
plt.title("CLASSIFICATION LOSS");
plt.ylabel("LOSS");
plt.ylim([0,1.1]);
plt.xlabel("EPOCHS");
plt.legend(['train', 'val']);

plt.subplot(1,2,2)
plt.plot(h.history['accuracy']);
plt.plot(h.history['val_accuracy']);
plt.title("CLASIFICATION ACCURACY");
plt.ylabel("ACCURACY");
plt.ylim([0,1.1]);
plt.xlabel("EPOCHS");
plt.legend(['train', 'val']);

model.load_weights('vgg-weights.h5')

_, acc = model.evaluate(test_generator)
print("Test accuracy : {} %".format(acc*100))

prediction = model.predict(test_generator)

pred = np.argmax(prediction, axis=1)
#pred = np.asarray(pred).astype('str')
original = test_generator.classes

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(original, pred)
print(accuracy)

cm = confusion_matrix(original, pred)

report = classification_report(original, pred, digits=4)
print(report)
plt.figure(figsize = (5,5))
sns.heatmap(cm, annot=True);

"""# FINE TUNING THE TOP PART OF THE VGG BASE MODEL"""

model.load_weights('vgg-weights.h5')

vgg_model.trainable = True

fine_tune_at = len(vgg_model.layers)-10

for layer in vgg_model.layers[:fine_tune_at]:
  layer.trainable = False

x = Flatten()(vgg_model.output)
prediction = Dense(5, activation='softmax')(x)
model = Model(inputs=vgg_model.input, outputs=prediction)
model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer= tf.keras.optimizers.Adam(learning_rate=0.00001),
              metrics= ["accuracy"]
             )


earlystopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              verbose=1,
                              patience=5
                             )
checkpointer = ModelCheckpoint(filepath="vgg-weights.h5",
                               verbose=1,
                               save_best_only=True
                              )

callbacks = [checkpointer, earlystopping]

h = model.fit(train_generator,
              steps_per_epoch= train_generator.n // train_generator.batch_size,
              epochs = 10,
              validation_data= valid_generator,
              validation_steps= valid_generator.n // valid_generator.batch_size,
              callbacks=callbacks)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(h.history['loss']);
plt.plot(h.history['val_loss']);
plt.title("CLASSIFICATION LOSS");
plt.ylabel("LOSS");
plt.ylim([0,1.1]);
plt.xlabel("EPOCHS");
plt.legend(['train', 'val']);

plt.subplot(1,2,2)
plt.plot(h.history['accuracy']);
plt.plot(h.history['val_accuracy']);
plt.title("CLASIFICATION ACCURACY");
plt.ylabel("ACCURACY");
plt.ylim([0,1.1]);
plt.xlabel("EPOCHS");
plt.legend(['train', 'val']);

model.load_weights('vgg-weights.h5')

_, acc = model.evaluate(test_generator)
print("Test accuracy : {} %".format(acc*100))

prediction = model.predict(test_generator)

pred = np.argmax(prediction, axis=1)
original = test_generator.classes

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(original, pred)
print(accuracy)

cm = confusion_matrix(original, pred)

report = classification_report(original, pred, digits=4)
print(report)
plt.figure(figsize = (5,5))
sns.heatmap(cm, annot=True);