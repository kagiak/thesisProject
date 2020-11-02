import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
import cv2

img_path = 'food-11/segmentation/test_masks_INCEPTIONV3/10_223.png'
img = image.load_img(img_path, target_size=(299, 299, 3))
img = image.img_to_array(img)
img = (img - img.min()) / np.ptp(img)
# plotting the heatmap
heatmap = np.uint8(255 * img)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # Apply heatmap to original Image
hif = .8
superimposed_img = heatmap * hif + img # Apply heatmap intensity factor
plt.imshow(superimposed_img)
plt.show()

img_path2 = 'food-11/segmentation/testPredict-fc32/10_223.jpg'
img2 = image.load_img(img_path2, target_size=(299, 299, 3))
img2 = image.img_to_array(img2)
img2 = (img2 - img2.min()) / np.ptp(img2)
# plotting the heatmap
heatmap2 = np.uint8(255 * img2)
heatmap2 = cv2.applyColorMap(heatmap2, cv2.COLORMAP_JET) # Apply heatmap to original Image
hif2 = .8
superimposed_img2 = heatmap2 * hif2 + img2 # Apply heatmap intensity factor
plt.imshow(superimposed_img2)
plt.show()

img_path3 = 'food-11/evaluation/10_223.jpg'
img3 = image.load_img(img_path3, target_size=(299, 299, 3))
plt.imshow(img3)
plt.show()
