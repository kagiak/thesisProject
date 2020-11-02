from keras.models import load_model
from keras import backend as K
from keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
import json
import os
import dense_crf
import tensorflow as tf

def openJson(file):
    with open(file) as File:
        dict = json.load(File)
    return dict

def getLayerOutput(model, layer):
    conv_layer = model.get_layer(layer)
    out = conv_layer.output
    return out

def CAM(model, output, img_path, layer, rangeNo, outputName):
    """
        This function is the one that creates and returns heatmap from the output of the given convolution layer
    """
    last_conv_layer = model.get_layer(layer)
    grads = K.gradients(output, last_conv_layer.output)[0] # grads: tensor that has 4 axes,
    # like the output of the conv layer, but it has the gradients of the output tensor
    pooled_grads = K.mean(grads, axis=(0, 1, 2)) # the K.mean returns
    # a tensor with the mean of elements of grads. So a tensor like this (512,). 512 channels with the means of the grads.
    # axis=(0,1,2): means that the mean runs through the 0 to 2 axes of the tensor
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]]) # K.function returns: output values as Numpy arrays.
    # first argument is input to this model
    # and second is set of 2 outputs - one for convolution and second for softmax output at the last layer.
    pooled_grads_value, conv_layer_output_value = iterate([x]) # both arrays
    # print (conv_layer_output_value.shape)
    # print (pooled_grads_value.shape)
    # We multiply each channel in the feature map array by "how important this channel is" with regard to the class

    for i in range(rangeNo):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # plotting the heatmap
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if (np.max(heatmap)!= 0):
        heatmap /= np.max(heatmap)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (299, 299))

    # heatmap or CAM or localization map
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap1 = heatmap
    # heatmap = np.uint8(255 * heatmap)  # Convert heatmap to RGB (224,224)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply heatmap to original Image
    # hif = .8
    # superimposed_img = heatmap * hif + img  # Apply heatmap intensity factor
    # cv2.imwrite('food11_output/' + outputName, superimposed_img)  # produce the output.jpeg to vizualize the map
    return heatmap1

def localizationMap(model, output, img_path, s):
    heatmap1 = CAM(model, output, img_path, 'conv2d_88', 384, str(s)+'_output1.jpeg')
    heatmap2 = CAM(model, output, img_path, 'conv2d_89', 384, str(s)+'_output2.jpeg')
    heatmap3 = CAM(model, output, img_path, 'conv2d_92', 384, str(s)+'_output3.jpeg')
    heatmap4 = CAM(model, output, img_path, 'conv2d_93', 384, str(s)+'_output4.jpeg')
    heatmap5 = CAM(model, output, img_path, 'conv2d_86', 320, str(s)+'_output5.jpeg')
    heatmap6 = CAM(model, output, img_path, 'conv2d_94', 190, str(s)+'_output6.jpeg')

    superimposed = np.average((np.array([heatmap1, heatmap2, heatmap3, heatmap4, heatmap5, heatmap6])), axis=0)
    return superimposed

def foregroundMap(model, output, img_path, s):
    heatmap5 = CAM(model, output, img_path, 'conv2d_95', 2048, str(s)+'_output-dilat1.jpeg')
    heatmap6 = CAM(model, output, img_path, 'conv2d_96', 2048, str(s)+'_output5_dilat2.jpeg')
    heatmap7 = CAM(model, output, img_path, 'conv2d_97', 2048, str(s) + '_output5_dilat3.jpeg')
    heatmap8 = CAM(model, output, img_path, 'conv2d_98', 2048, str(s) + '_output5_dilat4.jpeg')

    finalMap = np.add(heatmap6, heatmap5)
    finalMap1 = np.add(finalMap, heatmap7)
    finalMap2 = np.add(finalMap1,heatmap8)
    # scale the final map between 0 and 1 and get a foreground probability map Pf
    finalMap2 = finalMap2 / np.ptp(finalMap2)

    return finalMap2

def plotHeatmap(heatmap, outputName, img_path):
    """
    This function is used like CAM but it does not take the output of a convolution layer.
    It takes an already created heatmap and saves it in a folder for its vizualization
    and also returns the heatmap
    :param heatmap: the map that we want to plot
    :param outputName: the name of the map we want to plot
    :param img_path: the path to the original image
    :return: the heatmap
    """
    heatmap = (heatmap - heatmap.min())/ np.ptp(heatmap)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (299, 299))
    # plotting the heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap1 = heatmap
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # Apply heatmap to original Image
    # hif = .8
    # superimposed_img = heatmap * hif + img # Apply heatmap intensity factor
    # cv2.imwrite('food11_output_inc/'+outputName, superimposed_img) # produce the output.jpeg
    return heatmap1

def loadImage(img_path):
    img = image.load_img(img_path, target_size=(299, 299, 3))
    # plt.imshow(img)
    # plt.show()
    return img

def editImage(img):
    img = image.img_to_array(img)
    img = img / 255
    X = np.array(img)
    x = np.expand_dims(X, axis=0)

    return x

def test_dense_crf(img_path,heatmaps, name, prediction):
    """
    :param img_path: the path of the original images for which it wants to segment
    :param heatmaps: the final heatmap created by localization and fg/bg map
    :param name: name of the image in order to give the same name to the mask
    :param prediction: the class to which the image belongs
    :return: the final ground truth mask
    """
    I = Image.open(img_path)
    I = I.resize((299, 299), Image.ANTIALIAS)
    Iq = np.asarray(I.convert('L'), np.uint8)
    P = np.asarray(heatmaps, np.float32)
    fP = P/255.0
    totalLab = []

    w1    = 5.0  # weight of bilateral term
    alpha = 10   # spatial std
    beta  = 10   # rgb  std
    w2    = 3.0  # weight of spatial term
    gamma = 10   # spatial std
    it    = 5.0  # iteration
    param = (w1, alpha, beta, w2, gamma, it)

    lab = dense_crf.dense_crf(Iq, fP, param)
    # superimposed_img = P + Iq
    # fig = plt.figure()
    # ax1 = fig.add_subplot(231); ax1.imshow(I); ax1.set_title('Image')
    # ax2 = fig.add_subplot(232); ax2.imshow(P); ax2.set_title('Heatmap')
    # ax3 = fig.add_subplot(233); ax3.imshow(superimposed_img); ax3.set_title('Superimosed')
    P[P < (0.3 * P.max())] = 0
    P[P >= (0.3*P.max())] = 1
    # ax4 = fig.add_subplot(234); ax4.imshow(P); ax4.set_title('Segmentation')
    # mask3d = np.where(P[:,:,None]==1,I,[0,0,255])
    # ax5 = fig.add_subplot(235); ax5.imshow(mask3d); ax5.set_title('Cutted Image')
    # fig.subplots_adjust(hspace=2.0, wspace=0.3)
    # trueclass = name.split('_')
    # fig.text(0.8, 0.25, 'Prediction:'+str(prediction)+'\nTrue class:'+trueclass[0], ha='center', va='center')
    # name = str(name.replace('.jpg', '.png'))
    # plt.tight_layout()
    # plt.savefig('food11_output_inc/'+ name)
    cv2.imwrite('food-11/segmentation/train_masks_INCEPTIONV3/' + name, P)
    return lab

def concatenations(model, output, img_path, s):
    heatmap6 = CAM(model, output, img_path, 'mixed6', 768, str(s) + '_output_conc6.jpeg')
    heatmap7 = CAM(model, output, img_path, 'mixed7', 768, str(s) + '_output_conc7.jpeg')
    heatmap8 = CAM(model, output, img_path, 'mixed8', 1280, str(s) + '_output_conc8.jpeg')
    heatmap9 = CAM(model, output, img_path, 'mixed9', 2048, str(s) + '_output_conc9.jpeg')

    conc = np.average((np.array([heatmap6, heatmap7, heatmap8, heatmap9])), axis=0)
    return conc

if __name__ == "__main__":
    with tf.device('/gpu:1'):
        K.clear_session()
        model = load_model('models/InceptionV3.h5')
        trainFile = 'food-11/train.json'
        train = os.listdir("""food-11/training""")
        predictions = []
        truth = []
        expected = []
        nb_test_samples = len(openJson(trainFile))
        test_true = openJson(trainFile)
        values = (test_true.values())
        keys = test_true.keys()
        categories = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']

        for t in range(len(train)):
            print(t)
            img_path = 'food-11/training/'+str(train[t])
            img = loadImage(img_path)
            x = editImage(img)
            a = 0.5

            s = str(train[t]).replace('.jpg', ' ')
            s1 = str(train[t]).replace('.jpg', '.png')
            folderElements = os.listdir("""food-11/segmentation/train_masks_INCEPTIONV3""")
            # print (len(folderElements))
            if str(s1) not in folderElements:
                classes = categories
                preds = model.predict(x)
                argmax = np.argmax(preds[0])
                ar = model.output[:, argmax]
                concat = concatenations(model, ar, img_path,s)
                # plotHeatmap(concat, str(s) + 'cocnats.jpeg', img_path)
                Mc = localizationMap(model, ar, img_path,s)
                # plotHeatmap(Mc, str(s) + 'local.jpeg', img_path)
                Qc = np.multiply(Mc, concat)
                # plotHeatmap(Qc, str(s) + 'Qc.jpeg', img_path)
                newMc = localizationMap(model, ar, img_path,s)
                Pc = a * Qc + (1-a) * newMc
                Ptest = plotHeatmap(Pc, str(s) + 'heatmap.png', img_path)
                Dcrf = test_dense_crf(img_path, Ptest, s1, argmax)
