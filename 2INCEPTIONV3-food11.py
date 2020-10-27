import numpy as np
import json
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import keras

image_size = 299
trainFile = 'food-11/train.json'
valFile = 'food-11/val.json'
batch_size = 64
categories = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']

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
                img = image.load_img('food-11/training/' + train['Id'][i], target_size=(image_size, image_size, 3))
                img = image.img_to_array(img)
                img = (img / 255)# - 0.5
                train_image.append(img)

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
                img = image.load_img('food-11/validation/' + val['Id'][i], target_size=(image_size, image_size, 3))
                img = image.img_to_array(img)
                img = (img / 255)# - 0.5
                val_image.append(img)

                y_batch.append(val['Binary'][i])

            yield (np.array(val_image), np.array(y_batch))
            # return np.array(val_image), np.array(y_batch)

if __name__ == "__main__":
    nb_train_samples =  len(openJson(trainFile))
    nb_valid_samples = len(openJson(valFile))

    print("TRAIN LEN", nb_train_samples)
    print("VALID LEN", nb_valid_samples)

    # train_gen = train_datagen.flow(x_train,y_train, batch_size=batch_size)
    # val_gen = train_datagen.flow(x_val,y_val, batch_size=batch_size)
    train_gen = train_generator()
    val_gen = val_generator()

    with tf.device('/gpu:2'):

        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(11, activation='sigmoid')(x) #tf.nn.softmax
        model = Model(inputs=base_model.input, outputs=predictions)
        # model.summary()

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(optimizer=Adam(lr=1e-05), loss='binary_crossentropy', metrics=['acc'])

    checkpoint1 = ModelCheckpoint ('models/InceptionV3.h5', save_weights_only=False, monitor='val_loss', save_best_only=True, verbose=1, mode='min')
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.fit_generator(train_gen, epochs=50, steps_per_epoch= nb_train_samples // batch_size+ 1, validation_data=val_gen, validation_steps = nb_valid_samples // batch_size+ 1,callbacks=[es_callback,checkpoint1], verbose=1)
