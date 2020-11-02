import numpy as np
import json
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Add, BatchNormalization, Activation, Concatenate, AveragePooling2D
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

def Dilation(filters,convLayer, n1, n2, name):
    a = Conv2D(filters=filters, kernel_size=(n1, n2), dilation_rate=1, name=str(name)+'_1')(convLayer)
    b = Conv2D(filters=filters, kernel_size=(n1, n2), dilation_rate=2, name=str(name)+'_2')(convLayer)
    c = Conv2D(filters=filters, kernel_size=(n1, n2), dilation_rate=3, name=str(name)+'_3')(convLayer)

    a_b = BatchNormalization(axis=bn_axis, scale=False)(a)
    b_b = BatchNormalization(axis=bn_axis, scale=False)(b)
    c_b = BatchNormalization(axis=bn_axis, scale=False)(c)

    a_new = Activation('relu')(a_b)
    b_new = Activation('relu')(b_b)
    c_new = Activation('relu')(c_b)

    a_new = GlobalAveragePooling2D()(a_new)
    b_new = GlobalAveragePooling2D()(b_new)
    c_new = GlobalAveragePooling2D()(c_new)

    merged = Add()([a_new, b_new, c_new])
    return merged

if __name__ == "__main__":
    nb_train_samples =  len(openJson(trainFile))
    nb_valid_samples = len(openJson(valFile))
    print("TRAIN LEN", nb_train_samples)
    print("VALID LEN", nb_valid_samples)

    # train_gen = train_datagen.flow(x_train,y_train, batch_size=batch_size)
    # val_gen = train_datagen.flow(x_val,y_val, batch_size=batch_size)
    train_gen = train_generator()
    val_gen = val_generator()
    bn_axis = 3

    with tf.device('/gpu:2'):
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
        x = base_model.output
        base_model.layers.pop()
        activation_94 = base_model.layers[-1].output

        mixed9 = base_model.get_layer('mixed9').output
        conv2d_94 = base_model.get_layer('conv2d_94').output
        merged_94 = Dilation(192, conv2d_94, 1, 1, 'merged_94')

        conv2d_90 = Conv2D(448, (1,1), strides=(1,1), padding='same', use_bias=False, name='conv2d_90')(mixed9)
        batch_normalization_90 = BatchNormalization(axis=bn_axis, scale=False, name='batch_normalization_90')(conv2d_90)
        activation_90 = Activation('relu', name='activation_90')(batch_normalization_90)

        conv2d_87 = Conv2D(384, (1,1), strides=(1, 1), padding='same', use_bias=False, name='conv2d_87')(mixed9)
        batch_normalization_87 = BatchNormalization(axis=bn_axis, scale=False, name='batch_normalization_87')(conv2d_87)
        activation_87 = Activation('relu', name='activation_87')(batch_normalization_87)

        conv2d_91 = Conv2D(384, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv2d_91')(activation_90)
        batch_normalization_91 = BatchNormalization(axis=bn_axis, scale=False, name='batch_normalization_91')(conv2d_91)
        activation_91 = Activation('relu', name='activation_91')(batch_normalization_91)

        conv2d_88 = Conv2D(384, (1, 3), strides=(1, 1), padding='same', use_bias=False, name='conv2d_88')(activation_87)
        conv2d_89 = Conv2D(384, (3, 1), strides=(1, 1), padding='same', use_bias=False, name='conv2d_89')(activation_87)

        conv2d_92 = Conv2D(384, (1, 3), strides=(1, 1), padding='same', use_bias=False, name='conv2d_92')(activation_91)
        conv2d_93 = Conv2D(384, (1, 3), strides=(1, 1), padding='same', use_bias=False, name='conv2d_93')(activation_91)

        merged_88 = Dilation(384, conv2d_88, 1, 3, 'merged_88')
        merged_89 = Dilation(384, conv2d_89, 3, 1, 'merged_89')
        merged_92 = Dilation(384, conv2d_92, 1, 3, 'merged_92')
        merged_93 = Dilation(384, conv2d_93, 3, 1, 'merged_93')

        conv2d_86 = Conv2D(320, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='conv2d_86')(mixed9)
        merged_86 = Dilation(320, conv2d_86, 1, 1, 'merged_86')

        # batch_normalization_88 = BatchNormalization(axis=bn_axis, scale=False, name='batch_normalization_88')(merged_88)
        # batch_normalization_89 = BatchNormalization(axis=bn_axis, scale=False, name='batch_normalization_89')(merged_89)
        # batch_normalization_92 = BatchNormalization(axis=bn_axis, scale=False, name='batch_normalization_92')(merged_92)
        # batch_normalization_93 = BatchNormalization(axis=bn_axis, scale=False, name='batch_normalization_93')(merged_93)
        #
        # batch_normalization_86 = BatchNormalization(axis=bn_axis, scale=False, name='batch_normalization_86')(merged_86)
        #
        # activation_88 = Activation('relu', name='activation_88')(batch_normalization_88)
        # activation_89 = Activation('relu', name='activation_89')(batch_normalization_89)
        # activation_92 = Activation('relu', name='activation_92')(batch_normalization_92)
        # activation_93 = Activation('relu', name='activation_93')(batch_normalization_93)
        #
        # activation_86 = Activation('relu', name='activation_86')(batch_normalization_86)

        mixed9_1 = Concatenate(name='mixed9_1')([merged_88,merged_89])

        concatenate_2 = Concatenate(name='concatenate2')([merged_92,merged_93])

        mixed10 = Concatenate(name='mixed10')([merged_86,mixed9_1,concatenate_2,merged_94])
        print (mixed10.shape)
        # x = GlobalAveragePooling2D()(mixed10)
        predictions = Dense(11, activation='sigmoid')(mixed10)
        model = Model(inputs=base_model.input, outputs=predictions)

    model.summary()

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(optimizer=Adam(lr=1e-05), loss='binary_crossentropy', metrics=['acc'])

    checkpoint1 = ModelCheckpoint ('models/InceptionV3_dilated.h5', save_weights_only=False, monitor='val_loss', save_best_only=True, verbose=1, mode='min')
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.fit_generator(train_gen, epochs=50, steps_per_epoch= nb_train_samples // batch_size+ 1, validation_data=val_gen, validation_steps = nb_valid_samples // batch_size+ 1,callbacks=[es_callback,checkpoint1], verbose=1)
