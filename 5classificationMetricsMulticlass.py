import pandas as pd
import json
import csv
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from sklearn import metrics
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
import matplotlib.pyplot as plt

image_size = 299
model = load_model('models/InceptionV3.h5')
testFile = 'food-11/eval.json'
categories = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']
dataset = 'recipe1M/ingredients-reduc.csv'
batch_size = 64

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = categories # getIngredientsHeader() # #classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    return ax

def openJson(file):
    with open(file) as File:
        dict = json.load(File)
    return dict

def getIngredientsHeader():
    mylist = []
    with open(dataset, mode='r') as infile:
        reader = csv.reader(infile)
        ingrsSet = {rows[0] for rows in reader}

    for i in ingrsSet: # create the list with all the ingredients from the .csv file and sort it. Also create a same-size binary list
        mylist.append(i)
    mylist.sort()
    return mylist

def hamming_score(y_true, y_pred):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    set_finalpred = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        # print('set_true: {0}'.format(set_true))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            # print (y_pred[i])
            top_n = np.argsort(y_pred[i])
            for j in range(len(set_true)):
                set_finalpred.append(top_n[len(top_n)-1-j])
            # print ('top_n:::',top_n)
            # print('set_finalpred: {0}'.format(set_finalpred))
            inter = set_true.intersection(set_finalpred)
            tmp_a = len(set_true.intersection(set_finalpred))/\
                    float( len(set_true.union(set_finalpred)) )
        # print('tmp_a: {0}'.format(tmp_a))
        del set_finalpred[:]
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def precision(y_true, y_pred):
    acc_list = []
    set_finalpred = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            top_n = np.argsort(y_pred[i])
            for j in range(len(set_true)):
                set_finalpred.append(top_n[len(top_n) - 1 - j])
                tmp_a = len(set_true.intersection(set_finalpred)) / \
                        float(len(set_finalpred))
        del set_finalpred[:]
        acc_list.append(tmp_a)
    return np.mean(acc_list)\

def recall(y_true, y_pred):
    acc_list = []
    set_finalpred = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            top_n = np.argsort(y_pred[i])
            for j in range(len(set_true)):
                set_finalpred.append(top_n[len(top_n) - 1 - j])
                tmp_a = len(set_true.intersection(set_finalpred)) / \
                        float(len(set_true))
        del set_finalpred[:]
        acc_list.append(tmp_a)
    return np.mean(acc_list)

noIngr = []
nb_test_samples = len(openJson(testFile))
test_true = openJson(testFile)
values = (test_true.values())
keys = test_true.keys()
y_true = np.array([item[1] for item in values])
predictions = []
truth = []
expected = []
np.set_printoptions(precision=2)

def showTagging(values, keys, pred, path, folder, image_path):
    """
    :param values: the correct values for the labels
    :param keys: the name/id of the image in the test folder
    :param pred: the predictions of labels that the model gave
    :param path: the path of the folder where the image with the tags written on it will be saved
    :param folder: the folder of the test images
    :param image_path: the path of the test image
    :return: nothing. It saves the images with the tags in a folder
    """
    labels = categories #getIngredientsHeader() in case of Recipe1M-602 DB
    y_truelabels = np.array([item[0] for item in values])
    y_true = np.array([item[1] for item in values])

    for i in range(len(pred)):
        top_n = np.argsort(pred[i])
        set_true = set(np.where(y_true[i])[0])
        for j in range(len(set_true)):
            predictions.append(top_n[len(top_n)-1-j])
        truth.append(y_truelabels[i])
        # truth = [t.replace('u', '') for t in truth]
        # print ('predictions:',predictions)
        for p in predictions:
            expected.append(labels[p])
        # print ('expected:',expected)

        test = os.listdir(folder)
        for jpg in test:
            if (jpg == keys[i]):
                img = Image.open(image_path + jpg)
                # img = img.resize((224, 224), Image.ANTIALIAS)
                fontsize = 15  # starting font size
                # portion of image width you want text width to be
                img_fraction = 0.50
                font = ImageFont.truetype("alef/Alef-Bold.ttf", fontsize)
                while font.getsize('true labels:' + str(truth) + '\npredictions:' + str(expected))[0] < img_fraction * img.size[0]:
                    # iterate until the text size is just larger than the criteria
                    fontsize += 1
                    font = ImageFont.truetype("alef/Alef-Bold.ttf", fontsize)

                # optionally de-increment to be sure it is less than criteria
                fontsize -= 1
                font = ImageFont.truetype("alef/Alef-Bold.ttf", fontsize)
                draw = ImageDraw.Draw(img)
                draw.text((0, 0), 'true labels:' + str(truth) + '\npredictions:' + str(expected), font=font, fill="blue")
                img.save(path+jpg)

        del predictions[:]
        del truth[:]
        del expected[:]

def test():
    """
    In this function we prepare the test data in order for the model to predict the labels
    :return: the test images in form of arrays in order to feed them to the model and get predictions
    """
    with open(testFile) as testfile:
        dict_test = json.load(testfile)

    test = pd.DataFrame.from_dict(dict_test, orient='index')
    test.reset_index(level=0, inplace=True)
    test.columns = ['Id', 'Ingredients', 'Binary']

    testImages = os.listdir("""food-11/evaluation""") # newDATAbiggerDB-20/test
    test_image = []
    for i in range(len(testImages)):
        img = image.load_img('food-11/evaluation/' + test['Id'][i], target_size=(image_size, image_size, 3)) #newDATAbiggerDB-20/test
        img = image.img_to_array(img)
        img = (img / 255)
        test_image.append(img)
    return  (np.array(test_image))

with tf.device('/gpu:2'):
    x_test = test()
    pred = model.predict(x_test)
    y_pred = []
    true = []

    # this is for the multi-class problem and not the multi-label. It returns one prediction
    # for a multi-label problem it will return a list with the n-highest predictions.
    # Where n is equal to the number of the correct labels for each image
    for i in range(len(pred)):
        predmax = np.argmax(pred[i])
        truemax = np.argmax(y_true[i])
        y_pred.append(predmax)
        true.append(truemax)

    print ('Metrics: accuracy_score: {0}'.format(metrics.accuracy_score(true, y_pred)))
    # print ('Metrics: CategoricalAccuracy: {0}'.format(k.metrics.top_k_categorical_accuracy(true, y_pred,1)))
    print ('Metrics: cohen_kappa_score: {0}'.format(metrics.cohen_kappa_score(true,y_pred)))
    # print('Subset accuracy: {0}'.format(metrics.accuracy_score(y_true, pred, normalize=True, sample_weight=None)))
    print('\nHamming score: {0}'.format(hamming_score(y_true, pred))) #label-based accuracy

    log = metrics.log_loss(y_true, pred)
    print ('Metrics: LogLoss: {0}'.format(log))
    r = recall(y_true, pred)
    p = precision(y_true, pred)
    print ('Metrics: recall: {0}'.format(r))
    print ('Metrics: precision: {0}'.format(p))
    f1 = 2*(p*r)/(p+r)
    print ('Metrics: f1score: {0}'.format(f1))

    showTagging(values, keys, pred, 'tagging/InceptionV3-Food11_evaluation/', """food-11/evaluation""", 'food-11/evaluation/')

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(true, y_pred, classes=categories, title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    plot_confusion_matrix(true, y_pred, classes=categories, normalize=True, title='Normalized confusion matrix')
    plt.show()