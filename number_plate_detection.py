import cv2
from ConsoleParser import get_arguments_from_console


DEVICE_INDEX = 0  # default: web-cam
IMG_WIDTH = 640
IMG_HEIGHT = 480
MIN_AREA = 500
COLOR = (0, 0, 255)
number_plate_cascade = cv2.CascadeClassifier('resources/haarcascade_russian_plate_number.xml')


def show_number_plate(img, is_video=True):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    number_plates = number_plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in number_plates:
        area = w * h
        if area > MIN_AREA:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'Number Plate', (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, COLOR, 2)

            region_of_interest = img[y:y + h, x:x + w]
            cv2.imshow('Number Plate Region', region_of_interest)
            if not is_video:
                cv2.waitKey(0)


def main():
    console_arguments = get_arguments_from_console()
    if console_arguments.img:
        img = cv2.imread(console_arguments.img)
        show_number_plate(img, False)
    else:
        cap = cv2.VideoCapture(DEVICE_INDEX)
        cap.set(3, IMG_WIDTH)  # width
        cap.set(4, IMG_HEIGHT)  # height
        cap.set(10, 100)  # brightness

        while True:
            success, img = cap.read()
            show_number_plate(img)

            cv2.imshow('video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
