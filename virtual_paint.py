import cv2
import numpy as np


# web-cam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # frame width
cap.set(4, 480)  # frame height
cap.set(10, 100)  # brightness

colors_dict = {
    'green': [37, 199, 67, 179, 255, 255],  # hue_min, saturation_min, value_min, hue_max, saturation_max, value_max
    'rose': [142, 69, 117, 179, 153, 255],
}

colors_values_dict = {  # color values in bgr format
    'green': [69, 255, 48],
    'rose': [212, 0, 255],
}

points = []  # x, y, color


def find_color(img, colors_dict, colors_values_dict):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    points = []

    for color in colors_dict:
        # lower and upper limits to detect a color
        lower = np.array(colors_dict[color][:3])
        upper = np.array(colors_dict[color][3:])
        mask = cv2.inRange(img_hsv, lower, upper)
        x, y = get_contours(mask)
        # cv2.imshow(color, mask)
        if x != 0 and y != 0:
            points.append([x, y, color])
    return points


def get_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            # cv2.drawContours(img_res, cnt, -1, (0, 0, 255), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x + w // 2, y


def draw_one_canvas(points, colors_values_dict, img_res):
    for point in points:
        cv2.circle(img_res, (point[0], point[1]), 10, colors_values_dict[point[2]], cv2.FILLED)


def main():
    while True:
        success, img = cap.read()
        img_res = img.copy()

        new_points = find_color(img, colors_dict, colors_values_dict)
        if len(new_points) != 0:
            for point in new_points:
                points.append(point)
        if len(points) != 0:
            draw_one_canvas(points, colors_values_dict, img_res)

        cv2.imshow('video', img_res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
