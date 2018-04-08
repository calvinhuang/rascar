# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#geometric-transformations

import cv2
import numpy as np
# import matplotlib.pyplot as plt
def scaling():
    # Scaling: Resizing of the image.
    img = cv2.imread('1.jpg')

    res = cv2.resize(img,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
    cv2.imshow("image_chunk", res)
    cv2.waitKey(0)

    #OR
    # must use whole numbers for resizing with this method:
    # height, width = img.shape[:2]
    # res = cv2.resize(img,(4*width, 4*height), interpolation = cv2.INTER_AREA)
    # cv2.imshow("image_chunk", res)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

def translation():
    # translation: Sifting of object in x, y: Shifts the entire image?

    # Preferable interpolation methods are cv2.INTER_AREA
    # for shrinking and cv2.INTER_CUBIC(slow) &
    #  cv2.INTER_LINEAR for zooming.
    # By default, interpolation method used is cv2.INTER_LINEAR
    #   for all resizing purposes.

    img = cv2.imread('1.jpg',0)
    rows,cols = img.shape

    img = cv2.resize(img,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
    dst = img
    # trnaslation matrix M: {{1, 0, tx},{0,1,ty}}
    # making it inot a numpy array:
    M = np.float32([[1,0,100],[0,1,50]])

    # Pass the matrix to warpAffine: a-fine
    dst = cv2.warpAffine(img,M,(cols,rows))

    cv2.imshow('img',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotation():

    img = cv2.imread('fox.jpg', 0)
    img = cv2.resize(img,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
    rows, cols = img.shape
    # rotate 90 degrees centered at cols/3, rows/3 and scale by 0.5:
    M = cv2.getRotationMatrix2D((cols / 3, rows / 3), 90, 0.5)
    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow('img', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def affine():
    #     proniounced: a-fine
    #
    img = cv2.imread('1.jpg')
    rows, cols, ch = img.shape

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    M = cv2.getAffineTransform(pts1, pts2)

    dst = cv2.warpAffine(img, M, (cols, rows))


    # Where is plt from?!!
    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()

def perspective():
    img = cv2.imread('sudokusmall.png')
    rows, cols, ch = img.shape

    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (300, 300))

    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()

# translation()
# perspective()
rotation()