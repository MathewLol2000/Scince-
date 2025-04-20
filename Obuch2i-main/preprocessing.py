import cv2
import numpy as np


def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)

def erosion(image):
    kernel = np.ones((5, 5),np.uint8)
    erosion = cv2.erode(image, kernel, iterations = 1)
    return erosion

def dilation(image):
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(image, kernel, iterations=1) 
    return img_dilation

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def threshold(image):
    return cv2.threshold(image, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

def distance(image):
    dist = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    dist = (dist * 255).astype("uint8")
    return dist

def opening(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening

def gauss(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    return blur

def median(image):
    median = cv2.medianBlur(image, 5)
    return median

def bilateral(image):
    blur = cv2.bilateralFilter(image, 9, 75, 75)
    return blur

def histogram_equalization(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe = clahe.apply(image)
    return clahe


def hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([100, 43, 20])
    upper_green = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(image, image, mask=mask)

    gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    ret, generator = cv2.threshold(gray, 1,255,cv2.THRESH_BINARY)
    return generator

def lines_removal(image):
    # Remove border
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    temp2 = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, horizontal_kernel)
    result = cv2.add(temp2, image)

    # Convert to grayscale and Otsu's threshold
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    _,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return thresh

image = cv2.imread('test_img.jpg')
image = lines_removal(image)

cv2.imwrite('delete_me.jpg', image)