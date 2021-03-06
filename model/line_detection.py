import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
This is the traditional, old-fashioned way of detecting road lines using OpenCV.
Pros: no need for additional libraries, easy to implement and use
Cons: need to set area, where the lanes are, not as accurate as state-of-the-art neural nets.
"""


# Implemented using https://www.kdnuggets.com/2017/07/road-lane-line-detection-using-computer-vision-models.html/2 as base


# 1. Preprocessing the image (a single frame)
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


# 2. Canny edge detection
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


# 3. Select points from the region of interest
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with
    # depending on the input image
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


# 4. Apply Hough transform
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    #processed_lines = process_lines(img, lines)
    return lines


def is_crossing(image, coordinates, sensitivity=0.5):
    """
    Detects, whether the vehicle is crossing a lane or not.
    @param image: video frame
    @param coordinates: lane corner coordinates
    @param sensitivity: when the function starts telling about lane change
    @return: Boolean, is the car between two lanes or not.
    """
    # [[(upper_left_x, ymin_global), (lower_left_x, ymax_global)],
    # [(upper_right_x, ymin_global), (lower_right_x, ymax_global)]]

    if not coordinates:  # We want little false positives
        return True

    if len(coordinates) < 2:
        print("Found only a single line.")
        return True
    x_left = coordinates[0][1][0]
    x_right = coordinates[1][1][0]

    # print("is_crossing", x_left, x_right)

    height, width, _ = image.shape

    # print("sensitivity*width", sensitivity * width)
    # print("Relative placement of the left lane", x_left / width)
    # print("Relative placement of the right lane", x_right / width)


    if x_left < 400:  # Changing lane to the left
        print("Left lane change")
        return False
    if x_right > 350:  # Changing lane to the right
        print("Right lane change")
        return False

    # When no lane change detected
    return True


def process_lines(img, lines):
    # these variables represent the y-axis coordinates to which
    # the line will be extrapolated to
    ymin_global = img.shape[0]
    ymax_global = img.shape[0]

    # left lane line variables
    all_left_grad = []
    all_left_y = []
    all_left_x = []

    # right lane line variables
    all_right_grad = []
    all_right_y = []
    all_right_x = []
    for line in lines:
        #print("line in process_lines", (line))
        for x1, y1, x2, y2 in line:
            gradient, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            ymin_global = min(min(y1, y2), ymin_global)

            if (gradient > 0):
                all_left_grad += [gradient]
                all_left_y += [y1, y2]
                all_left_x += [x1, x2]
            else:
                all_right_grad += [gradient]
                all_right_y += [y1, y2]
                all_right_x += [x1, x2]

    left_mean_grad = np.mean(all_left_grad)
    left_y_mean = np.mean(all_left_y)
    left_x_mean = np.mean(all_left_x)
    left_intercept = left_y_mean - (left_mean_grad * left_x_mean)

    right_mean_grad = np.mean(all_right_grad)
    right_y_mean = np.mean(all_right_y)
    right_x_mean = np.mean(all_right_x)
    right_intercept = right_y_mean - (right_mean_grad * right_x_mean)

    # Make sure we have some points in each lane line category
    if (len(all_left_grad) > 0) and (len(all_right_grad) > 0):
        #print("ymin_global", all_left_grad)
        upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
        upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)

        #print("returning", [[(upper_left_x, ymin_global), (lower_left_x, ymax_global)],
        #        [(upper_right_x, ymin_global), (lower_right_x, ymax_global)]])

        if upper_right_x > 1080 or lower_right_x < 0: #impossible case
            return None
        if upper_left_x < 0 or lower_left_x > 1080:
            return None
        return [[(upper_left_x, ymin_global), (lower_left_x, ymax_global)],
                [(upper_right_x, ymin_global), (lower_right_x, ymax_global)]]


# 5. Drawing lines
def draw_lines(img, processed_lines, color=(0, 0, 255), thickness=5):
    """
    This function draws `lines` with `color` and `thickness`.
    """
    #processed_lines = process_lines(img, lines)
    if lines is None:
        return
    (upper_left_x, ymin_global), (lower_left_x, ymax_global) = lines[0]
    (upper_right_x, ymin_global), (lower_right_x, ymax_global) = lines[1]

    #print("draw_lines", lines)

    cv2.line(img, (upper_left_x, ymin_global),
             (lower_left_x, ymax_global), color, thickness)
    cv2.line(img, (upper_right_x, ymin_global),
             (lower_right_x, ymax_global), color, thickness)


# 6. Add lines to the input image
def weighted_img(img, initial_img, alpha=0.8, beta=1., llambda=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, llambda)


# 7. Combine everything
def find_lines(image):
    # grayscale the image
    grayscaled = grayscale(image)

    # apply gaussian blur
    kernelSize = 5
    gaussianBlur = gaussian_blur(grayscaled, kernelSize)

    # canny
    minThreshold = 100
    maxThreshold = 200
    edgeDetectedImage = canny(gaussianBlur, minThreshold, maxThreshold)

    #plt.imshow(edgeDetectedImage)
    #plt.show()

    # apply mask
    lowerLeftPoint = [250, 1000] #x, y
    upperLeftPoint = [500, 600]
    upperRightPoint = [1000, 600]
    lowerRightPoint = [1500, 1080]

    pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint,
                     lowerRightPoint]], dtype=np.int32)

    masked_image = region_of_interest(edgeDetectedImage, pts)

    #plt.imshow(masked_image)
    #plt.show()

    # hough lines
    rho = 1
    theta = np.pi / 180
    threshold = 30
    min_line_len = 10
    max_line_gap = 10

    lines = hough_lines(masked_image, rho, theta, threshold, min_line_len,
                        max_line_gap)

    #print("from hough_lines", lines, ".")

    processed_lines = process_lines(image, lines)

    #print("from process_lines in find_lines", processed_lines)

    is_crossing_flag = is_crossing(image, processed_lines)

    #print("From hough_lines", lines)

    return processed_lines, is_crossing_flag


if __name__ == '__main__':

    """
    IMAGE_FILE = 'road-line-detection-0.jpeg'
    image = cv2.imread(IMAGE_FILE)
    lines, is_crossing = find_lines(image)
    draw_lines(image, lines)
    cv2.imshow("", image)
    cv2.waitKey(0)
    print(lines)
    """

    VIDEO_FILE = "4-line-crossing.mp4"

    cap = cv2.VideoCapture(VIDEO_FILE)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_counter = 0

    out = cv2.VideoWriter("output.avi", fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), fps=30,
                          frameSize=(width, height))

    while cap.isOpened():
        frame_counter += 1
        print("Examining frame", frame_counter)
        ret, frame = cap.read()
        if not ret: #end of the video
            break

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        lines, is_crossing_flag = find_lines(frame)

        #print("lines from find_lines", lines)

        draw_lines(frame, lines)

        #cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
