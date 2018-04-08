'''
	functions from mark_all_lines 
	and lines and lines3

'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys #for reading command line arguments etc.
import glob
import matplotlib.image as mpimg
import math


#--------------------------------------

#--------------------------------------

def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


def movingAverage(avg, new_sample, N=20):
    if (avg == 0):
        return new_sample
    avg -= avg / N
    avg += new_sample / N
    return avg


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # state variables to keep track of most dominant segment
    largestLeftLineSize = 0
    largestRightLineSize = 0
    largestLeftLine = (0, 0, 0, 0)
    largestRightLine = (0, 0, 0, 0)
    global avgLeft
    global avgRight

    if lines is None:
        # avgx1, avgy1, avgx2, avgy2 = avgLeft
        # cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw left line
        # avgx1, avgy1, avgx2, avgy2 = avgRight
        # cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw right line
        return

    for line in lines:
        for x1, y1, x2, y2 in line:
            size = math.hypot(x2 - x1, y2 - y1)
            slope = ((y2 - y1) / (x2 - x1))
            # Filter slope based on incline and
            # find the most dominent segment based on length
            if (slope > 0.5):  # right
                if (size > largestRightLineSize):
                    largestRightLine = (x1, y1, x2, y2)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            elif (slope < -0.5):  # left
                if (size > largestLeftLineSize):
                    largestLeftLine = (x1, y1, x2, y2)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


    return img 
    # Define an imaginary horizontal line in the center of the screen
    # and at the bottom of the image, to extrapolate determined segment
    imgHeight, imgWidth = (img.shape[0], img.shape[1])
    upLinePoint1 = np.array([0, int(imgHeight - (imgHeight / 3))])
    upLinePoint2 = np.array([int(imgWidth), int(imgHeight - (imgHeight / 3))])
    downLinePoint1 = np.array([0, int(imgHeight)])
    downLinePoint2 = np.array([int(imgWidth), int(imgHeight)])

    # Find the intersection of dominant lane with an imaginary horizontal line
    # in the middle of the image and at the bottom of the image.
    p3 = np.array([largestLeftLine[0], largestLeftLine[1]])
    p4 = np.array([largestLeftLine[2], largestLeftLine[3]])
    upLeftPoint = seg_intersect(upLinePoint1, upLinePoint2, p3, p4)
    downLeftPoint = seg_intersect(downLinePoint1, downLinePoint2, p3, p4)
    if (math.isnan(upLeftPoint[0]) or math.isnan(downLeftPoint[0])):
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw left line
        avgx1, avgy1, avgx2, avgy2 = avgRight
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw right line
        return
    cv2.line(img, (int(upLeftPoint[0]), int(upLeftPoint[1])), (int(downLeftPoint[0]), int(downLeftPoint[1])),
             [0, 0, 255], 8)  # draw left line

    # Calculate the average position of detected left lane over multiple video frames and draw
    avgx1, avgy1, avgx2, avgy2 = avgLeft
    avgLeft = (
    movingAverage(avgx1, upLeftPoint[0]), movingAverage(avgy1, upLeftPoint[1]), movingAverage(avgx2, downLeftPoint[0]),
    movingAverage(avgy2, downLeftPoint[1]))
    avgx1, avgy1, avgx2, avgy2 = avgLeft
    cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw left line

    # Find the intersection of dominant lane with an imaginary horizontal line
    # in the middle of the image and at the bottom of the image.
    p5 = np.array([largestRightLine[0], largestRightLine[1]])
    p6 = np.array([largestRightLine[2], largestRightLine[3]])
    upRightPoint = seg_intersect(upLinePoint1, upLinePoint2, p5, p6)
    downRightPoint = seg_intersect(downLinePoint1, downLinePoint2, p5, p6)
    if (math.isnan(upRightPoint[0]) or math.isnan(downRightPoint[0])):
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw left line
        avgx1, avgy1, avgx2, avgy2 = avgRight
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw right line
        return
    cv2.line(img, (int(upRightPoint[0]), int(upRightPoint[1])), (int(downRightPoint[0]), int(downRightPoint[1])),
             [0, 0, 255], 8)  # draw left line

    # Calculate the average position of detected right lane over multiple video frames and draw
    avgx1, avgy1, avgx2, avgy2 = avgRight
    avgRight = (movingAverage(avgx1, upRightPoint[0]), movingAverage(avgy1, upRightPoint[1]),
                movingAverage(avgx2, downRightPoint[0]), movingAverage(avgy2, downRightPoint[1]))
    avgx1, avgy1, avgx2, avgy2 = avgRight
    cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw left line


def hough_lines(img, rho=2, theta=np.pi/180, threshold=5, min_line_len=10, max_line_gap=20):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    # lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
    #                         maxLineGap=max_line_gap)
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    # line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, a=0.8, b=1., l=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alfa + img * beta + lambda
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, l)


def mark_lanes(image, lo = 50, hi = 150):
    debug = False

    if image is None: raise ValueError("no image given to mark_lanes")
    
    # grayscale the image to make finding gradients clearer
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    if debug:
        save_image("1_gray.jpg", gray)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    if debug:
        save_image("2_blur_gray.jpg", blur_gray)

    # Define our parameters for Canny and apply
    low_threshold = lo
    high_threshold = hi
    edges_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)
    if debug:
        save_image("3_edges.jpg", edges_img)

    imshape = image.shape

    #-------- R O I -------------------------------
    vertices = np.array([[(0, imshape[0]),
                          (450, 320),
                          (490, 320),
                          (imshape[1], imshape[0]) ]],
                          dtype=np.int32)

    masked_edges = region_of_interest(edges_img, vertices )
    if debug:
        save_image("4_masked_image.jpg", masked_edges)

    # Define the Hough transform parameters
    rho             = 2           # distance resolution in pixels of the Hough grid
    theta           = np.pi/180   # angular resolution in radians of the Hough grid
    threshold       = 5        # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10       # minimum number of pixels making up a line
    max_line_gap    = 20       # maximum gap in pixels between connectable line segments

    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    if debug:
        save_image("5_line_image.jpg", line_image)

    # Draw the lines on the image
    # initial_img * a+ img * b + l
    # marked_lanes = cv2.addWeighted(image, 0.8, edges_img, 1, 0)
    color_edges = cv2.cvtColor(edges_img, cv2.COLOR_GRAY2RGB)

    image = cv2.addWeighted(image, 0.4, color_edges, 1, 0)
    marked_lanes = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    # marked_lanes = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
    if debug:
        save_image("6_marked_lanes.jpg", marked_lanes)

    return marked_lanes



#--------------------------------------

#--------------------------------------
def test_line_one_video():
    cap = cv2.VideoCapture(0)

    #sometimes cap is not opened (?) use cap.open() to open it
    if not cap.isOpened():
        cap.open()
    while(True):
        # Capture frame-by-frame
        #if read returns false, that's the end of the video.
        ret, frame = cap.read()

        # Our operations on the frame goes here:
        # cvtColor will return an image that gets stroed in gray

        #------------------------- Video Features ------------------
        #access video features: cap.get(propId)
        #frame width and height: props 3 and 4:
        # print "frame size: ", cap.get(3), 'X', cap.get(4)
        #-----------------------------------------------------------
        # set features by cap.set(propId, value):

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = frame
        # see img_proc_canny_edge_detection.py:
        # gray = cv2.Canny(gray, 100, 200)
        # gray = cv2.add(gray, gray)

        # gray = hough_lines(frame)
        gray = mark_lanes(frame)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0XFF == ord('1'):
            ret = cap.set(3, cap.get(3)/2)
            ret = cap.set(4, cap.get(4)/2)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def test_on_images():
    #give this function the source, the destination 
    #   and the function to run on the images
    source_path = 'test_images/source_images/'
    paths = glob.glob(source_path + "*.jpg")
    print "Paths: ", paths
    for i, image_path in enumerate(paths):
        print " --- ", image_path, "----------------------"
        image = cv2.imread(image_path)

        # result = hough_linesP(image)
        result = mark_lanes(image)
        # plt.subplot(2, 3, i + 1)
        # plt.imshow(result)
        # mpimg.imsave('test_images/marked/' + image_path[12:-4] + '_detected.jpg', result)
        print 'test_images/marked/' + image_path[len(source_path):-4]
        cv2.imwrite('test_images/marked/' + image_path[len(source_path):-4] + '_detected.jpg', result)
    print ". . . . . . . . . . . . . . . . . . . . ."
    print "Batch processed. "


#--------------------------------------
#	m a i n ( )
#--------------------------------------

def main():
    print "------------------------------------------------"
    print "number of command line args: ", len(sys.argv)
    print "------------------------------------------------"
    print ""
    if len(sys.argv) >1:
        print "here is the commandline arguments: "
        # argv[0] is the name of this file
        # the rest are the command line arguments:
        for i in range(len(sys.argv)):
            print  sys.argv[i]
        print "------------------------------------------------" 
    print ""
    print ""
    # test_line_one_video()
    test_on_images()

if __name__ == '__main__':
    main()
