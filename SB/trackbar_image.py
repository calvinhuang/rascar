import cv2
import numpy as np
from trackbar import trackbars
debug = False
image = cv2.imread("test_images/solidWhiteCurve.jpg")

if image is None: raise ValueError("no image given to mark_lanes")
# grayscale the image to make finding gradients clearer
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# Define our parameters for Canny and apply
# low_threshold = 50
# high_threshold = 150
# grayscale the image to make finding gradients clearer

t = trackbars(['low', 'high'],[500,500])
font = cv2.FONT_HERSHEY_SIMPLEX

while (1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    low_threshold = t._pos('low')
    high_threshold = t._pos('high')
    edges_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)




    cv2.putText(edges_img, str(low_threshold)+" "+str(high_threshold), (20, 20), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("window", edges_img)
cv2.destroyAllWindows()
