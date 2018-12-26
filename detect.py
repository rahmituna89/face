import cv2
import matplotlib.pyplot as plt

haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')


def convert_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def plt_show(img):
    plt.imshow(img)
    plt.show()


def detect_faces(f_cascade, colored_img, scale_factor=1.2):
    img_copy = colored_img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (244, 244, 66), 10)
    return img_copy


baby = cv2.imread('data/baby1.jpg')
faces_detected = detect_faces(haar_face_cascade, baby)
plt_show(convert_to_rgb(faces_detected))
