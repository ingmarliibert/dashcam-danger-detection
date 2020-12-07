import cv2
import numpy as np
from scipy.stats import itemfreq
from model.traffic_sign import traffic_sign_factory


def get_dominant_color(image, n_colors):
    pixels = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    flags, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)

    return palette[np.argmax(itemfreq(labels)[:, -1])]


font = cv2.FONT_HERSHEY_COMPLEX

traffic_sign = traffic_sign_factory()

def detect_circles(frame, original):
    rows = frame.shape[0]

    # https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=40, minRadius=20, maxRadius=80)

    if circles is not None:
        circles = np.uint16(np.around(circles))


        i = 0
        for circle in circles[0, :]:

            x = circle[0]
            y = circle[1]
            r = circle[2]

            # given x,y are circle center and r is radius
            rectX = (x - r)
            rectY = (y - r)

            print(f'rectX={rectX}')
            x_padding = int(rectX / 10)
            y_padding = int(rectY / 10)

            crop_img = original[(rectY - y_padding):(rectY + 2 * r) + y_padding,
                       (rectX - x_padding):(rectX + 2 * r) + x_padding]
            # cv2.imshow("circle",crop_img)
            result = traffic_sign.inception(crop_img)
            print(result)

            if result.score > .30:
                # cv2.imshow(f'circle traffic sign {i}', crop_img)

                cv2.circle(original, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                cv2.circle(original, (circle[0], circle[1]), 2, (255, 255, 0), 8)

                cv2.putText(original, result.class_name, (x, y), font, 1, (0, 0, 0))

            i = i + 1


def shapes(cnts, img, original):
    triagleNB = 0

    crop_img = img
    for cnt in cnts:
        epsilon = 0.26 * cv2.arcLength(cnt, True) #0.1, 0.25 2 triangles (overlap), 0.26 1 triangle
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        area = cv2.contourArea(approx, False)

        if area < 200:
            continue

        #print(f'approx={approx} \n area={area}')

        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx) == 3:
            triagleNB += 1
            cv2.drawContours(img, [approx], 0, (255, 0, 0), 5)
            # cv2.putText(img, "Triangle", (x, y), font, 1, (255, 0, 0))

            # Triangle extraction
            flat_approx = approx.ravel()
            xs = approx.ravel()[0:len(flat_approx):2]
            ys = approx.ravel()[1:len(flat_approx):2]

            # paddings
            x_padding = 10
            y_padding = 10

            # wrapping rectangle
            lower_x = (np.amin(xs) - x_padding)
            upper_x = (np.amax(xs) + x_padding)
            lower_y = (np.amin(ys) - y_padding)
            upper_y = (np.amax(ys) + y_padding)

            # rectangle : x -> from min x point to max x (paddings ignored)
            crop_img = original[lower_y:upper_y,lower_x:upper_x]
            cv2.imshow("detect", img)
            # cv2.imshow("cropped triangle", crop_img)

            result = traffic_sign.inception(crop_img)

            if result.score > .30:
                print(result)
                cv2.putText(img, result.class_name, (x, y), font, 1, (255, 0, 0))

        elif len(approx) == 4:
            cv2.putText(img, "Rectangle", (x, y), font, 1, (0))

    #print(triagleNB, " triangles found")

# frame = cv2.imread('./france-paris-30-kph-sign.jpg')
frame = cv2.imread('./traffic_sign2.jpg')

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
img = cv2.medianBlur(gray, 5)

kernel = np.ones((4, 4), np.uint8)
dilation = cv2.dilate(gray, kernel, iterations=2)
blur = cv2.GaussianBlur(dilation, (5, 5), 0)

_, threshold = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY)

cv2.imshow('cleaned image', blur)

detect_circles(blur, frame)

# _, threshold = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY)
# cv2.imshow('threshold', threshold)
contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
shapes(contours, frame, img)

while True:
    cv2.imshow('detected circles', frame)
    k = cv2.waitKey(0)
    if k == 27:  # Esc key to stop
        break

cv2.destroyAllWindows()
