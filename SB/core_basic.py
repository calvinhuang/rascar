# Accessing and Modifying pixel values
#


import cv2
import numpy as np

# load a color image:
img = cv2.imread('fox.jpg')
# You can access a pixel value by its row and column coordinates. For BGR image, it returns an array of Blue, Green, Red values. For grayscale image, just corresponding intensity is returned.

px = img[100,100]
print px
# [157 166 200]

# accessing only blue pixel
blue = img[100,100,0]
print blue
# 157
# You can modify the pixel values the same way.

img[100,100] = [255,255,255]
print img[100,100]

#---------------------------------------

for i in range(0,10):
    for j in range(0,10):
        img[i,j] = [255,0,255]

# - - - - Portions of the image....

# This is numpy indexing:
#   from this row to this row and from this column to this column:

portion  = img[100:300, 200:600]
img[300:500, 100:500] = portion

cv2.imwrite('fox_bad.jpg', img)
#create new image:
image_chunk = np.zeros((500,500,3), np.uint8)
image_chunk[0:200, 0:400] = portion
cv2.imwrite("fox_chunk.jpg", image_chunk);



cv2.imshow("image_chunk", image_chunk)
cv2.waitKey(0)
#---------------------------------------

# [255 255 255]
# Warning Numpy is a optimized library for fast array calculations. So simply accessing each and every pixel values and modifying it will be very slow and it is discouraged.
# Note Above mentioned method is normally used for selecting a region of array, say first 5 rows and last 3 columns like that. For individual pixel access, Numpy array methods, array.item() and array.itemset() is considered to be better. But it always returns a scalar. So if you want to access all B,G,R values, you need to call array.item() separately for all.
# Better pixel accessing and editing method :

# accessing RED value
img.item(10,10,2)
# 59

# modifying RED value
img.itemset((10,10,2),255)
img.item(10,10,2)
# 100
# ------- Accessing Image Properties
#
# Image properties include number of rows, columns and channels, type of image data, number of pixels etc.
#
# Shape of image is accessed by img.shape. It returns a tuple of number of rows, columns and channels (if image is color):

print "image shape: ", img.shape
# (342, 548, 3)

# Note If image is grayscale, tuple returned contains only number of rows and columns. So it is a good method to check if loaded image is grayscale or color image.
# Total number of pixels is accessed by img.size:

print "image size: ", img.size
# 562248
# Image datatype is obtained by img.dtype:

print "image data type: ", img.dtype
# uint8

# Note img.dtype is very important while debugging because a large number of errors in OpenCV-Python code is caused by invalid datatype.


# --------------- Image ROI
#
# Sometimes, you will have to play with certain region of images. For eye detection in images, first face detection is done all over the image and when face is obtained, we select the face region alone and search for eyes inside it instead of searching whole image. It improves accuracy (because eyes are always on faces :D ) and performance (because we search for a small area)
#
# ROI is again obtained using Numpy indexing. Here I am selecting the ball and copying it to another region in the image:

ball = img[280:340, 330:390]
img[273:333, 100:160] = ball