import cv2
from ConsoleParser import get_arguments_from_console

DEVICE_INDEX = 0  # default: web-cam
IMG_WIDTH = 640
IMG_HEIGHT = 480
IMG_BRIGHTNESS = 100

face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')


def show_face(img, is_video=True):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('image', img)
    if not is_video:
        cv2.waitKey(0)


def main():
    console_arguments = get_arguments_from_console()

    if console_arguments.img:
        img = cv2.imread(console_arguments.img)
        show_face(img, False)
    else:
        cap = cv2.VideoCapture(DEVICE_INDEX)
        cap.set(3, IMG_WIDTH)
        cap.set(4, IMG_HEIGHT)
        cap.set(10, IMG_BRIGHTNESS)

        while True:
            success, img = cap.read()
            show_face(img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
