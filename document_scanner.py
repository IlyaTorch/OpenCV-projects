import cv2
import numpy as np
from ConsoleParser import get_arguments_from_console


IMG_WIDTH = 720
IMG_HEIGHT = 640
IMG_BRIGHTNESS = 150


def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_cany = cv2.Canny(img_blur, 200, 200)
    kernel = np.ones((5, 5))
    img_dial = cv2.dilate(img_cany, kernel=kernel, iterations=2)
    img_threshold = cv2.erode(img_dial, kernel=kernel, iterations=1)
    return img_threshold


def get_contours(img, img_contour):
    biggest_contour = np.array([])
    max_area = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:
            peri = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if area > max_area and len(approximation) == 4:
                biggest_contour = approximation
                max_area = area

    cv2.drawContours(img_contour, biggest_contour, -1, (255, 0, 0), 20)
    return biggest_contour


def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)
    x_y_sum = points.sum(1)

    new_points[0] = points[np.argmin(x_y_sum)]
    new_points[3] = points[np.argmax(x_y_sum)]

    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]

    return new_points


def get_warp(img, biggest_contour):
    biggest_contour = reorder(biggest_contour)
    points1 = np.float32(biggest_contour)
    points2 = np.float32([[0, 0], [IMG_WIDTH, 0], [0, IMG_HEIGHT], [IMG_WIDTH, IMG_HEIGHT]])
    matrix = cv2.getPerspectiveTransform(points1, points2)
    img_warped = cv2.warpPerspective(img, matrix, (IMG_WIDTH, IMG_HEIGHT))

    img_cropped = img_warped[20:img_warped.shape[0]-20, 20:img_warped.shape[1]-20]
    img_cropped = cv2.resize(img_cropped, (img_warped.shape[1::-1]))

    return img_cropped


def stack_images(scale, img_arr):
    rows = len(img_arr)
    cols = len(img_arr[0])
    rows_available = isinstance(img_arr[0], list)
    width = img_arr[0][0].shape[1]
    height = img_arr[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_arr[x][y].shape[:2] == img_arr[0][0].shape [:2]:
                    img_arr[x][y] = cv2.resize(img_arr[x][y], (0, 0), None, scale, scale)
                else:
                    img_arr[x][y] = cv2.resize(img_arr[x][y], (img_arr[0][0].shape[1], img_arr[0][0].shape[0]), None, scale, scale)
                if len(img_arr[x][y].shape) == 2: img_arr[x][y]= cv2.cvtColor(img_arr[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank]*rows
        hor_con = [image_blank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_arr[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_arr[x].shape[:2] == img_arr[0].shape[:2]:
                img_arr[x] = cv2.resize(img_arr[x], (0, 0), None, scale, scale)
            else:
                img_arr[x] = cv2.resize(img_arr[x], (img_arr[0].shape[1], img_arr[0].shape[0]), None, scale, scale)
            if len(img_arr[x].shape) == 2: img_arr[x] = cv2.cvtColor(img_arr[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_arr)
        ver = hor
    return ver


def show_document(img, is_video=True):
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_contour = img.copy()

    img_threshold = preprocess(img)
    biggest = get_contours(img_threshold, img_contour)
    if biggest.size != 0:
        img_warped = get_warp(img, biggest)
        img_array = [[img, img_contour], [img_threshold, img_warped]]
    else:
        img_array = [[img, img_contour], [img, img]]

    result = stack_images(0.5, img_array)
    cv2.imshow('before/after', result)
    if not is_video:
        cv2.waitKey(0)


def main():
    console_arguments = get_arguments_from_console()

    if console_arguments.img:
        img = cv2.imread(console_arguments.img)
        show_document(img, False)
    else:
        cap = cv2.VideoCapture(0)
        cap.set(3, IMG_WIDTH)
        cap.set(4, IMG_HEIGHT)
        cap.set(10, IMG_BRIGHTNESS)
        while True:
            _, img = cap.read()
            show_document(img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
