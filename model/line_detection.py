import cv2
import numpy as np

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
    processed_lines = process_lines(img, lines)
    return processed_lines


def process_lines(img, lines):
    # these variables represent the y-axis coordinates to which
    # the line will be extrapolated to
    if lines is None:
        return
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
        upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
        upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)

        return [[(upper_left_x, ymin_global), (lower_left_x, ymax_global)],
                [(upper_right_x, ymin_global), (lower_right_x, ymax_global)]]


# 5. Drawing lines
def draw_lines(img, lines, color=[0, 0, 255], thickness=12):
    """
    This function draws `lines` with `color` and `thickness`.
    """
    processed_lines = process_lines(img, lines)
    if processed_lines is None:
        return
    (upper_left_x, ymin_global), (lower_left_x, ymax_global) = processed_lines[0]
    (upper_right_x, ymin_global), (lower_right_x, ymax_global) = processed_lines[1]

    cv2.line(img, (upper_left_x, ymin_global),
             (lower_left_x, ymax_global), color, thickness)
    cv2.line(img, (upper_right_x, ymin_global),
             (lower_right_x, ymax_global), color, thickness)


# 6. Add lines to the input image
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


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

    # apply mask
    lowerLeftPoint = [130, 540]
    upperLeftPoint = [410, 350]
    upperRightPoint = [570, 350]
    lowerRightPoint = [915, 540]

    pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint,
                     lowerRightPoint]], dtype=np.int32)

    masked_image = region_of_interest(edgeDetectedImage, pts)

    # hough lines
    rho = 1
    theta = np.pi / 180
    threshold = 30
    min_line_len = 20
    max_line_gap = 20

    lines = hough_lines(masked_image, rho, theta, threshold, min_line_len,
                        max_line_gap)

    return lines


if __name__ == '__main__':
    IMAGE_FILE = 'road-line-detection-0.jpeg'
    image = cv2.imread(IMAGE_FILE)
    lines = find_lines(image)
    print(lines)
