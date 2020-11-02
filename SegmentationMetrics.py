import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing import image

def loadImage(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (299, 299))
    img = image.img_to_array(img)
    return img

def dice_coef(y_true, y_pred, smooth=1):
    intersection = np.sum(y_true * y_pred, axis=-1)
    union = np.sum(y_true, axis=-1) + np.sum(y_pred, axis=-1)
    dice = np.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def iou(target, prediction):
    intersection = np.sum(np.abs(target* prediction))
    union = np.sum(np.maximum(np.abs(target), np.abs(prediction)))
    iou = float(intersection) / (union)
    return iou

def metrics():
        groundTruth = os.listdir("""food-11/segmentation/test_masks_INCEPTIONV3""") #ground truth in .png
        testMask = os.listdir("""food-11/segmentation/testPredict-fc32""") #predicted mask in .jpg
        testMaskPNG = os.listdir("""food-11/segmentation/testPredictPNG-fc32""")

        # convert .jpg mask to .png mask
        for n, id in (enumerate(testMask)):
            jpg2png = cv2.imread('food-11/segmentation/testPredict-fc32/' + testMask[n])
            jpg2png = cv2.resize(jpg2png, (299, 299), interpolation=cv2.INTER_AREA)
            plt.imsave('food-11/segmentation/testPredictPNG-fc32/' + testMask[n].replace('.jpg', '.png'),jpg2png)

        count0 = count1 = count2 = count3 = count4 = count5 = count6 = count7 = count8 = count9 = count10 = 0
        add0 = add1 = add2 = add3 = add4 = add5 = add6 = add7 = add8 = add9 = add10 = 0

        count0d = count1d = count2d = count3d = count4d = count5d = count6d = count7d = count8d = count9d = count10d = 0
        add0d = add1d = add2d = add3d = add4d = add5d = add6d = add7d = add8d = add9d = add10d = 0


        for t in range(len(groundTruth)):
            img_path = 'food-11/segmentation/test_masks_INCEPTIONV3/' + str(groundTruth[t]) # ground truth images
            newt = groundTruth[t].replace('.png', '.png')
            for s in range(len(testMaskPNG)):
                mask_path = 'food-11/segmentation/testPredictPNG-fc32/' + str(testMaskPNG[s])
                # convert masks, and prediction in a correct form with only 0 and 1 values for the metrics
                if (testMaskPNG[s] == newt):
                    image_ = loadImage(img_path)
                    gray_image = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
                    gray_image = gray_image / 255
                    gray_image[gray_image > 0] = 3
                    gray_image[gray_image == 0] = 0
                    gray_image[gray_image == 3] = 1

                    mask_ = loadImage(mask_path)
                    gray_mask = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
                    gray_mask = gray_mask / 255
                    gray_mask[gray_mask > 0] = 3
                    gray_mask[gray_mask == 0] = 0
                    gray_mask[gray_mask == 3] = 1

                    DICE = dice_coef(gray_image, gray_mask)
                    # calculate the mean value per class by the sum of dice metric and the total number of images/class
                    category = testMask[s]
                    if (category[0] == '0'):
                        count0d = count0d + DICE
                        add0d = add0d + 1
                    elif (category[0] == '1' and category[1] == '_'):
                        count1d = count1d + DICE
                        add1d = add1d + 1
                    elif (category[0] == '2'):
                        count2d = count2d + DICE
                        add2d = add2d + 1
                    elif (category[0] == '3'):
                        count3d = count3d + DICE
                        add3d = add3d + 1
                    elif (category[0] == '4'):
                        count4d = count4d + DICE
                        add4d = add4d + 1
                    elif (category[0] == '5'):
                        count5d = count5d + DICE
                        add5d = add5d + 1
                    elif (category[0] == '6'):
                        count6d = count6d + DICE
                        add6d = add6d + 1
                    elif (category[0] == '7'):
                        count7d = count7d + DICE
                        add7d = add7d + 1
                    elif (category[0] == '8'):
                        count8d = count8d + DICE
                        add8d = add8d + 1
                    elif (category[0] == '9'):
                        count9d = count9d + DICE
                        add9d = add9d + 1
                    elif (category[0] == '1' and category[1] == '0'):
                        count10d = count10d + DICE
                        add10d = add10d + 1

                    IOU = iou(gray_image,gray_mask)
                    # calculate the mean value per class by the sum of IOU metric and the total number of images/class
                    if (category[0] == '0'):
                        count0 = count0 + IOU
                        add0 = add0 + 1
                    elif (category[0] == '1' and category[1] == '_'):
                        count1 = count1 + IOU
                        add1 = add1 + 1
                    elif (category[0] == '2'):
                        count2 = count2 + IOU
                        add2 = add2 + 1
                    elif (category[0] == '3'):
                        count3 = count3 + IOU
                        add3 = add3 + 1
                    elif (category[0] == '4'):
                        count4 = count4 + IOU
                        add4 = add4 + 1
                    elif (category[0] == '5'):
                        count5 = count5 + IOU
                        add5 = add5 + 1
                    elif (category[0] == '6'):
                        count6 = count6 + IOU
                        add6 = add6 + 1
                    elif (category[0] == '7'):
                        count7 = count7 + IOU
                        add7 = add7 + 1
                    elif (category[0] == '8'):
                        count8 = count8 + IOU
                        add8 = add8 + 1
                    elif (category[0] == '9'):
                        count9 = count9 + IOU
                        add9 = add9 + 1
                    elif (category[0] == '1' and category[1] == '0'):
                        count10 = count10 + IOU
                        add10 = add10 + 1

        print ('Total IOU for class 0', count0/add0, 'Total DICE for class 0', count0d/add0d)
        print ('Total IOU for class 1', count1/add1, 'Total DICE for class 1', count1d/add1d)
        print ('Total IOU for class 2', count2/add2, 'Total DICE for class 2', count2d/add2d)
        print ('Total IOU for class 3', count3/add3, 'Total DICE for class 3', count3d/add3d)
        print ('Total IOU for class 4', count4/add4, 'Total DICE for class 4', count4d/add4d)
        print ('Total IOU for class 5', count5/add5, 'Total DICE for class 5', count5d/add5d)
        print ('Total IOU for class 6', count6/add6, 'Total DICE for class 6', count6d/add6d)
        print ('Total IOU for class 7', count7/add7, 'Total DICE for class 7', count7d/add7d)
        print ('Total IOU for class 8', count8/add8, 'Total DICE for class 8', count8d/add8d)
        print ('Total IOU for class 9', count9/add9, 'Total DICE for class 9', count9d/add9d)
        print ('Total IOU for class 10', count10/add10, 'Total DICE for class 10', count10d/add10d)

if __name__ == "__main__":
    metrics()
