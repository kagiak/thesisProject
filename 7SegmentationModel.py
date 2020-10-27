import os
import numpy as np
import matplotlib.pyplot as plt
from keras_segmentation.models.fcn import fcn_32,fcn_8, fcn_8_vgg, fcn_32_vgg
from keras_segmentation.models.segnet import segnet, vgg_segnet
from keras_segmentation.data_utils.data_loader import \
    verify_segmentation_dataset
import tensorflow as tf
from skimage.io import imread
import cv2
from keras.preprocessing import image
from keras_segmentation.predict import predict_multiple
# from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.models.pspnet import pspnet_50
from tqdm import tqdm
from keras_segmentation.models.unet import resnet50_unet

trainImage = os.listdir("""food-11/training""")
valImage = os.listdir("""food-11/validation""")
trainMask = os.listdir("""food-11/segmentation/train_masks_INCEPTIONV3""")
valMask = os.listdir("""food-11/segmentation/val_masks_INCEPTIONV3""")
nb_train_samples = len(trainImage)

def loadImage(img_path):
    # img = image.load_img(img_path) #, target_size=(299, 299, 1))
    img = cv2.imread(img_path)
    img = cv2.resize(img, (299, 299))
    img = image.img_to_array(img)
    return img

def labels():
    Bread = 4227136 #[64, 128, 64] , HEX:4227136
    Dairy = 12583040 #[192, 0, 128] HEX:C00080
    Dessert = 32960 #[0, 128, 192] HEX:0080C0
    Egg = 8388608 #[128, 0, 0] HEX:	800000
    Fried = 4194432 #[64, 0, 128] HEX:400080
    Meat = 4194496 #[64, 0, 192] HEX:4000C0
    Noodles = 12615744 #[192, 128, 64] HEX:C08040
    Rice = 12632192 #[192, 192, 128] HEX:C0C080
    Seafood = 4210816 #[64, 64, 128] HEX:404080
    Soup = 32832 #[0, 128, 64] HEX:008040
    Vegetable = 8388800 #[128, 0, 192] HEX:8000C0

    COLOR_DICT = np.array([Bread, Dairy, Dessert, Egg, Fried, Meat, Noodles, Rice, Seafood, Soup, Vegetable])
    return COLOR_DICT

def modifyValData(image_height, image_width, channels):
    for n, id in (enumerate(valImage)):
        image_1 = cv2.imread('food-11/validation/' + valImage[n])
        image_1 = cv2.resize(image_1, (image_height, image_width), interpolation=cv2.INTER_AREA) #INTER_NEAREST
        # height, width, channels = image_1.shape
        plt.imsave('food-11/segmentation/val_images/' + valImage[n].replace('.jpg','.png'), image_1)

        for m, mask_file in (enumerate(valMask)):
            name = (valMask[m]).replace('png','jpg')
            if (name==id):
                id = id.replace('jpg', 'png')
                mask_1 = cv2.imread('food-11/segmentation/val_masks_INCEPTIONV3/'+ id) #'food-11/training_segm_gt/'
                mask_1 = cv2.resize(mask_1, (image_height, image_width), interpolation=cv2.INTER_AREA) #INTER_NEAREST or INTER_AREA
                # height, width, channels = mask_1.shape
                mask_1[mask_1 > 0] = 1
                # colored = np.where(mask_ > 0, colors[label], mask_)
                plt.imsave('food-11/segmentation/val_masks/' + valMask[m], mask_1)

def modifyTrainData(image_height, image_width,channels):
    for n, id in (enumerate(trainImage)):
        image_1 = cv2.imread('food-11/training/' + trainImage[n])
        image_1 = cv2.resize(image_1, (image_height, image_width), interpolation=cv2.INTER_AREA)
        plt.imsave('food-11/segmentation/train_images/' + trainImage[n].replace('.jpg','.png'), image_1)

        for m, mask_file in (enumerate(trainMask)):
            name = (trainMask[m]).replace('png','jpg')
            if (name==id):
                id = id.replace('jpg', 'png')
                mask_1 = cv2.imread('food-11/segmentation/train_masks_INCEPTIONV3/'+ id) #'food-11/training_segm_gt/'
                mask_1 = cv2.resize(mask_1, (image_height, image_width), interpolation=cv2.INTER_AREA) #INTER_NEAREST or INTER_AREA
                mask_1[mask_1 > 0] = 1
                plt.imsave('food-11/segmentation/train_masks/' + trainMask[m], mask_1)

def check():
    for m, mask_file in (enumerate(trainMask)):
        # mask_ = imread('food-11/training_segm_gt/' +trainMask[m])
        mask_ = imread('food-11/segmentation/train_masks_INCEPTIONV3/' + trainMask[m])
        if np.argmax(mask_)> 9:
            print (trainMask[m])
            print (np.unique(mask_))

def dice_coef(y_true, y_pred, smooth=1):
    intersection = np.sum(y_true * y_pred, axis=-1)
    union = np.sum(y_true, axis=-1) + np.sum(y_pred, axis=-1)
    dice = np.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def test_verify():
    verify_segmentation_dataset(trainImage, trainMask, 50)

def metrics():
    testMaskManual = os.listdir("""food-11/segmentation/val_masks_INCEPTIONV3""") #ground truth
    testMask = os.listdir("""food-11/validationSegmentationOutputs-segnet""") #mask created from segmentation model
    for n, id in (enumerate(testMask)):
        image_test2 = cv2.imread('food-11/validationSegmentationOutputs-segnet/' + testMask[n])
        image_test2 = cv2.resize(image_test2, (299, 299), interpolation=cv2.INTER_AREA)
        plt.imsave('food-11/segmentation/evaluationMetric-segnet/' + testMask[n].replace('.jpg','.png'), image_test2)

    for t in range(len(testMaskManual)):
        img_path = 'food-11/segmentation/val_masks_INCEPTIONV3/' + str(testMaskManual[t])
        newt = testMaskManual[t].replace('.png', '.png')
        for s in range(len(testMask)):
            mask_path = 'food-11/segmentation/validationSegmentationOutputs-segnet/' + str(testMask[s])
            if (testMask[s] == newt): # and s == 3):
                image_ = loadImage(img_path)
                gray_image = cv2.cvtColor(image_,cv2.COLOR_BGR2GRAY)
                gray_image = gray_image / 255
                gray_image[gray_image > 0] = 3
                gray_image[gray_image == 0] = 0
                gray_image[gray_image == 3] = 1

                mask_ = loadImage(mask_path)
                gray_mask = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
                gray_mask = gray_mask / 255
                gray_mask[gray_mask > 0] = 5
                gray_mask[gray_mask == 0] = 0
                gray_mask[gray_mask == 5] = 1

                DICE = dice_coef(gray_image, gray_mask)

def transfer_weights(m1, m2, verbose=True):

    assert len(m1.layers) == len(
        m2.layers), "Both models should have same number of layers"

    nSet = 0
    nNotSet = 0

    if verbose:
        print("Copying weights ")
        bar = tqdm(zip(m1.layers, m2.layers))
    else:
        bar = zip(m1.layers, m2.layers)

    for l, ll in bar:

        if not any([w.shape != ww.shape for w, ww in zip(list(l.weights),
                                                         list(ll.weights))]):
            if len(list(l.weights)) > 0:
                ll.set_weights(l.get_weights())
                nSet += 1
        else:
            nNotSet += 1

    if verbose:
        print("Copied weights of %d layers and skipped %d layers" %
              (nSet, nNotSet))

def test_model():
    h =  299
    w =  299
    n_c = 11
    modifyTrainData(h,w,3)
    modifyValData(h, w, 3)

    with tf.device('/gpu:3'):

        model = fcn_32(n_classes=n_c, input_height=h, input_width=w)
        # pretrained_model = pspnet_50_ADE_20K()
        # model = pspnet_50(n_classes=n_c, input_height=h, input_width=w)
        # transfer_weights(pretrained_model, model)

        model.train(
            train_images="food-11/segmentation/train_images/",
            train_annotations="food-11/segmentation/train_masks/",validate=True, val_images="food-11/segmentation/val_images/",val_annotations="food-11/segmentation/val_masks/",
            epochs=50, optimizer_name='adadelta', verify_dataset=True, checkpoints_path="food-11/segmentationCheckpoints-fcn_32/fcn_32"
        )

        predict_multiple(
            checkpoints_path="food-11/segmentationCheckpoints-fcn_32/fcn_32",
            inp_dir="food-11/evaluation/",
            out_dir="food-11/segmentation/testPredict-fc32/"
        )
        print ("DONE")

if __name__ == "__main__":
    test_model()
    # metrics()

