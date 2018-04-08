# https://github.com/DavidAwad/Lane-Detection/blob/master/detection.py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
import math
import cv2


def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    # reshape lines to a 2d matrix
    lines = lines.reshape(lines.shape[0], lines.shape[2])
    # create array of slopes
    slopes = (lines[:,3] - lines[:,1]) /(lines[:,2] - lines[:,0])
    # remove junk from lists
    lines = lines[~np.isnan(lines) & ~np.isinf(lines)]
    slopes = slopes[~np.isnan(slopes) & ~np.isinf(slopes)]
    # convert lines into list of points
    lines.shape = (lines.shape[0]//2,2)

    # Right lane
    # move all points with negative slopes into right "lane"
    right_slopes = slopes[slopes < 0]
    right_lines = np.array(list(filter(lambda x: x[0] > (img.shape[1]/2), lines)))
    max_right_x, max_right_y = right_lines.max(axis=0)
    min_right_x, min_right_y = right_lines.min(axis=0)

    # Left lane
    # all positive  slopes go into left "lane"
    left_slopes = slopes[slopes > 0]
    left_lines = np.array(list(filter(lambda x: x[0] < (img.shape[1]/2), lines)))
    max_left_x, max_left_y = left_lines.max(axis=0)
    min_left_x, min_left_y = left_lines.min(axis=0)

    # Curve fitting approach
    # calculate polynomial fit for the points in right lane
    right_curve = np.poly1d(np.polyfit(right_lines[:,1], right_lines[:,0], 2))
    left_curve  = np.poly1d(np.polyfit(left_lines[:,1], left_lines[:,0], 2))

    # shared ceiling on the horizon for both lines
    min_y = min(min_left_y, min_right_y)

    # use new curve function f(y) to calculate x values
    max_right_x = int(right_curve(img.shape[0]))
    min_right_x = int(right_curve(min_right_y))

    min_left_x = int(left_curve(img.shape[0]))

    r1 = (min_right_x, min_y)
    r2 = (max_right_x, img.shape[0])
    print('Right points r1 and r2,', r1, r2)
    cv2.line(img, r1, r2, color, thickness)

    l1 = (max_left_x, min_y)
    l2 = (min_left_x, img.shape[0])
    print('Left points l1 and l2,', l1, l2)
    cv2.line(img, l1, l2, color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    print "hough lines: img.shape:  ", img.shape
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    print "hough lines: lines: ", lines
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if len(lines) == 0:
        #666: line is empty if no line was found.
        draw_lines(line_img, lines)
    return line_img


def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def mark_lanes(image):
    if image is None: raise ValueError("no image given to mark_lanes")
    # grayscale the image to make finding gradients clearer
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)


    imshape = image.shape
    vertices = np.array([[(0, imshape[0]),
                          (450, 320),
                          (490, 320),
                          (imshape[1], imshape[0]) ]],
                          dtype=np.int32)

    masked_edges = region_of_interest(edges_img, vertices )


    # Define the Hough transform parameters
    rho             = 2           # distance resolution in pixels of the Hough grid
    theta           = np.pi/180   # angular resolution in radians of the Hough grid
    threshold       = 15       # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20       # minimum number of pixels making up a line
    max_line_gap    = 20       # maximum gap in pixels between connectable line segments

    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw the lines on the edge image
    # initial_img * alfa + img * beta + lambda
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return lines_edges


def video():
    cap = cv2.VideoCapture(0)

    # sometimes cap is not opened (?) use cap.open() to open it
    if not cap.isOpened():
        cap.open()
    while (True):
        # Capture frame-by-frame
        # if read returns false, that's the end of the video.
        ret, frame = cap.read()

        # Our operations on the frame come here:
        # cvtColor will return an image that gets stroed in gray

        # ------------------------- Video Features ------------------
        # acess video features: cap.get(propId)
        # frame width and height: props 3 and 4:
        # print "frame size: ", cap.get(3), 'X', cap.get(4)
        # -----------------------------------------------------------
        # set features by cap.set(propId, value):

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # gray = frame
        # # see img_proc_canny_edge_detection.py:
        # edges = cv2.Canny(gray, 100, 200)
        # gray = cv2.add(gray, edges)

        gray = mark_lanes(frame)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0XFF == ord('1'):
            ret = cap.set(3, cap.get(3) / 2)
            ret = cap.set(4, cap.get(4) / 2)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def feed_image():
    img = cv2.imread('road3.jpg')
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    lined_image = mark_lanes(img)
    cv2.imshow("xxx", lined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# feed_image()
video()