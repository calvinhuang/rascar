# https://www.youtube.com/watch?v=a6pDdS6sY2E
# https://github.com/paramaggarwal/CarND-LaneLines-P1/blob/master/P1.ipynb
# https://medium.com/computer-car/my-lane-detection-project-for-the-self-driving-car-nanodegree-by-udacity-36a230553bd3

#project submission:
# https://github.com/DavidAwad/Lane-Detection/blob/master/notebook.ipynb

# ====================================================
# you need these two for test_mark_lanes()
import imageio
# ====================================================
# imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import math
import glob
import cv2
from trackbar import trackbars


def test_imread():
    #reading in an image
    image = mpimg.imread('road3.jpg')
    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimesions:', image.shape)
    # plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image
    cv2.imshow("image", image)
    cv2.waitKey(0)

    image = cv2.imread('road3.jpg')
    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimesions:', image.shape)
    # plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image
    cv2.imshow("image", image)
    cv2.waitKey(0)

def show_image(img):
    cv2.imshow("image", img);
    cv2.waitKey(0)

def save_image_seq(img):
    count = count + 1
    cv2.imwrite(str(count)+".jpg", img)

def save_image(file_name, img):
    cv2.imwrite(file_name, img)

def imread_cv(file_name):
    img = cv2.imread(file_name)
    return img

def imread_mp(file_name):
    img = mpimg.imread(file_name)
    return img

def write_text(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    row, col, junk = img.shape

    cv2.putText(img, text, (0, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img

def video():
    cap = cv2.VideoCapture(0)

    #sometimes cap is not opened (?) use cap.open() to open it
    if not cap.isOpened():
        cap.open()
    while(True):
        # Capture frame-by-frame
        #if read returns false, that's the end of the video.
        ret, frame = cap.read()
        # Our operations on the frame come here:
        # cvtColor will return an image that gets stroed in gray

        #------------------------- Video Features ------------------
        #acess video features: cap.get(propId)
        #frame width and height: props 3 and 4:
        # print "frame size: ", cap.get(3), 'X', cap.get(4)
        #-----------------------------------------------------------
        # set features by cap.set(propId, value):

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # gray = frame
        # # see img_proc_canny_edge_detection.py:
        # edges = cv2.Canny(gray, 100, 200)
        # gray = cv2.add(gray, edges)

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





def play_video(v):
    pass
# * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *

# * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *  * *

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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


def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    msg = ""
    if (lines is None):
        return img

    if len(lines) == 0:
        print "          zero length lines"
        return img
    else:
        print len(lines)
        msg = msg + chr(13) + str(len(lines))
    # reshape lines to a 2d matrix
    lines = lines.reshape(lines.shape[0], lines.shape[2])
    # create array of slopes
    slopes = (lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0])
    # remove junk from lists
    lines = lines[~np.isnan(lines) & ~np.isinf(lines)]
    slopes = slopes[~np.isnan(slopes) & ~np.isinf(slopes)]
    # convert lines into list of points
    lines.shape = (lines.shape[0] // 2, 2)

    # Right lane
    # move all points with negative slopes into right "lane"
    right_slopes = slopes[slopes < 0]
    right_lines = np.array(list(filter(lambda x: x[0] > (img.shape[1] / 2), lines)))
    print "right_lanes: ", len(right_lines)
    if (len(right_lines)>0):
        max_right_x, max_right_y = right_lines.max(axis=0)
        min_right_x, min_right_y = right_lines.min(axis=0)
    else:
        msg  = msg + chr(13) + "no right lane. "
        return
    # Left lane
    # all positive  slopes go into left "lane"
    left_slopes = slopes[slopes > 0]
    left_lines = np.array(list(filter(lambda x: x[0] < (img.shape[1] / 2), lines)))
    print "left_lanes: ", len(left_lines)
    if (len(left_lines) > 0):
        max_left_x, max_left_y = left_lines.max(axis=0)
        min_left_x, min_left_y = left_lines.min(axis=0)
    else:
        msg  = msg + chr(13) + "no left lane. "
        return

    # Curve fitting approach
    # calculate polynomial fit for the points in right lane
    if right_lines is not None:
        right_curve = np.poly1d(np.polyfit(right_lines[:, 1], right_lines[:, 0], 2))
    else:
        print "right lines is none "

    if left_lines is not None:
        left_curve = np.poly1d(np.polyfit(left_lines[:, 1], left_lines[:, 0], 2))
    else:
        print "left lines is none: "

    # shared ceiling on the horizon for both lines
    min_y = min(min_left_y, min_right_y)

    # use new curve function f(y) to calculate x values
    max_right_x = int(right_curve(img.shape[0]))
    min_right_x = int(right_curve(min_right_y))

    min_left_x = int(left_curve(img.shape[0]))

    r1 = (min_right_x, min_y)
    r2 = (max_right_x, img.shape[0])
    print('Right points r1 and r2,', r1, r2)
    msg = msg + chr(13) + "Right: "+str(r1) + " " + str(r2)

    cv2.line(img, r1, r2, color, thickness)

    l1 = (max_left_x, min_y)
    l2 = (min_left_x, img.shape[0])
    print('Left points l1 and l2,', l1, l2)
    msg = msg + chr(13) + "left: "+str(l1) + " " + str(l2)
    cv2.line(img, l1, l2, color, thickness)
    # write_text(img, msg)
def draw_hough_lines(img, lines):
    if lines is None:
        return img
    if len(lines)==0:
        return img
    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # print "Hough Lines: ", lines
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # color_lines = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
    # img = cv2.addWeighted(img, 0.4, color_lines, 1, 0)
    # return img

    draw_hough_lines(line_img, lines)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, a=0.8, b=1., l=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * a + img * b + l
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

def objects(filename):
    img_filt = cv2.medianBlur(cv2.imread(filename, 0), 5)
    img_th = cv2.adaptiveThreshold(img_filt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print "contours: ", contours
    # print "hierarchy: ", hierarchy
    return img_th

def do_nothing(img):
    return img

def test_read_image():
    gray = imread_mp("road4.jpg")
    gray = grayscale(gray)
    show_image(gray)

def test_mark_lanes():
    img = imread_cv("road3.jpg")
    line_edges = mark_lanes(img)
    show_image(line_edges)


def test_on_images():
    paths = glob.glob('test_images/*.jpg')

    for i, image_path in enumerate(paths):
        image = cv2.imread(image_path)
        result = mark_lanes(image)

        # plt.subplot(2, 3, i + 1)
        # plt.imshow(result)
        # mpimg.imsave('test_images/marked/' + image_path[12:-4] + '_detected.jpg', result)

def double_check_mark_lanes():
    image = cv2.imread("test_images/solidWhiteRight.jpg")
    result = mark_lanes(image)
    show_image(result)
def test_videos():
    global count
    count = 0
    white_output = 'test_images/white.mp4'
    clip1 = VideoFileClip("test_images/road2.mov")

    # white_clip = clip1.fl_image(mark_lanes) #NOTE: this function expects color images!!
    white_clip = clip1.fl_image(do_nothing) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

def trackbar_Canny():
    t = trackbars(['low', 'high'], [500, 500])
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = imread_cv("test_images/solidWhiteRight.jpg")

    while (1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        lo = t._pos('low')
        hi = t._pos('high')
        line_edges = mark_lanes(img, lo, hi)

        cv2.putText(line_edges, str(lo) + " " + str(hi), (20, 20), font, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow("window", line_edges)
    cv2.destroyAllWindows()

def test_find_objects():
    img = objects("test_images/solidWhiteRight.jpg")
    show_image(img)

# test_on_images()
# double_check_mark_lanes()
# test_videos()
# trackbar_Canny()
# test_find_objects()
# video()
test_mark_lanes()
