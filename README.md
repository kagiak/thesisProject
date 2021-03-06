# ThesisProject scripts

The thesis contains 3 parts. 
- First the classification
- Second the creation of localization maps (or segmentation masks) 
- and third the segmentation procedure. 

*The following work is mainly done on the multiclass dB Food-11.*

*In the link [here](https://www.kaggle.com/trolukovich/food11-image-dataset) there are the images of the Food-11 dB, divided into 3 sub-folders, training, validation, evaluation.*

*Inside the segmentation folder there are the localization masks for training, validation and testinf sets.*
*And inside the 110 folder there are 110 test images that have been manually segmented and used for evaluating the performance of the created segmentation masks for the second part of the thesis*

1) The **getIngredients.py** script is used in order to get the correct ingredients for each image from the original dataset (Recipe1m) that contains all the information retrieved from the web.
2) The **INCEPTIONV3-food11.py** script uses the Inception v3 model (modified for my problem) on the Food-11 dataset for classification purposes.
3) The **INCEPTIONV3-food11-dilated.py** adds the dilation technique to the last convolution module of the Inception v3 architecture and applied again on Food-11 DB for classification purposes.
4) The **INCEPTIONV3-recipe1m.py** script uses the Inception v3 model (modified for my problem) for classification purposes again but his time applied on the Recipe1M DB.
5) The **classificationMetricsMulticlass.py** script calculates the metrics for evaluiation for the multiclass DB Food-11.
6) The **camVizualINCEPTIONV3.py** script is the one that uses the best classification model (Inception v3 in this case) in order to create segmentation masks of the classes, that will be used afterwards as ground truths for the segmentation algorithm.
7) The **SegmentationModel.py** script contains the segmentation architecture that is being used for training.
8) While the **segmentationMetrics.py** contains the functions for calculating the metrics and evaluating the performance of the previous segmentation model.
9) Finally the script **VizualizeMaskGTImage.py** is used for vizualization purposes of the results from the camVizualINCEPTIONV3.py script (6).
