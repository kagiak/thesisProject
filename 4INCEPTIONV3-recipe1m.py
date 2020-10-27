import numpy as np
import os
import json
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import keras

currentPath = os.getcwd()
os.chdir("../..")
newPath = os.getcwd()
image_size = 299
trainFile = '/home/eleni/code/Recipe1M-602/train.json'
valFile = '/home/eleni/code/Recipe1M-602/val.json'
batch_size = 64

train_datagen = ImageDataGenerator(featurewise_center=True,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)

def openJson(file):
    with open(file) as File:
        dict = json.load(File)
    return dict

def train_generator():
    with open(trainFile) as trainfile:
        dict_train = json.load(trainfile)

    train = pd.DataFrame.from_dict(dict_train, orient='index')
    train.reset_index(level=0, inplace=True)
    train.columns = ['Id', 'Ingredients', 'Binary']
    nb_samples = len(train)

    while True:
        for start in range(0, nb_samples, batch_size):
            train_image =[]
            y_batch = []
            end = min(start + batch_size, nb_samples)
            for i in range(start, end):
                name = train['Id'][i]
                img = image.load_img("/home/shared/data/Recipe1M-602/train/" + name[0] + "/" + name[1] + "/" + name[2] + "/" + name[3] + "/" + name, target_size=(image_size, image_size, 3))
                img = image.img_to_array(img)
                img = (img / 255)
                train_image.append(img)
                aaaaaaa = np.array(train_image)

                y_batch.append(train['Binary'][i])


            # return np.array(train_image), np.array(y_batch)
            yield (np.array(train_image), np.array(y_batch))

def val_generator():
    with open(valFile) as valfile:
        dict_val = json.load(valfile)

    val = pd.DataFrame.from_dict(dict_val, orient='index')
    val.reset_index(level=0, inplace=True)
    val.columns = ['Id', 'Ingredients', 'Binary']

    nb_samples = len(val)

    while True:
        for start in range(0, nb_samples, batch_size):
            val_image = []
            y_batch = []
            end = min(start + batch_size, nb_samples)
            for i in range(start, end):
                name = val['Id'][i]
                img = image.load_img("/home/shared/data/Recipe1M-602/val/" + name[0] + "/" + name[1] + "/" + name[2] + "/" + name[3] + "/" + name, target_size=(image_size, image_size, 3))
                img = image.img_to_array(img)
                img = (img / 255)
                val_image.append(img)

                y_batch.append(val['Binary'][i])

            yield (np.array(val_image), np.array(y_batch))
            # return np.array(val_image), np.array(y_batch)

if __name__ == "__main__":
    nb_train_samples =  len(openJson(trainFile))
    nb_valid_samples = len(openJson(valFile))

    print("TRAIN LEN", nb_train_samples)
    print("VALID LEN", nb_valid_samples)

    train_gen = train_generator()
    val_gen = val_generator()

    with tf.device('/gpu:1'):

        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(602, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        # model.summary()

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(optimizer=Adam(lr=1e-05), loss='categorical_crossentropy', metrics=['acc']) #1e-05, 0.01

    checkpoint1 = ModelCheckpoint ('/home/eleni/code/models/Recipe1M602.h5', save_weights_only=False, monitor='val_loss', save_best_only=True, verbose=1, mode='min')
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # my_lr_scheduler = LearningRateScheduler(adapt_learning_rate) # def adapt_learning_rate()
    model.fit_generator(train_gen, epochs=100, steps_per_epoch= nb_train_samples // batch_size+ 1, validation_data=val_gen, validation_steps = nb_valid_samples // batch_size+ 1,callbacks=[es_callback,checkpoint1], verbose=1, use_multiprocessing=True)
