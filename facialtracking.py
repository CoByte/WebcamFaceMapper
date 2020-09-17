import cv2
import numpy as np


net = cv2.dnn.readNetFromCaffe(
    "C:/Users/Owen/Programming/Python/WebcamScrewery/model/deploy.prototxt.txt",
    "C:/Users/Owen/Programming/Python/WebcamScrewery/model/res10_300x300_ssd_iter_140000.caffemodel")


def get_face(img):
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    detected_faces = []

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            detected_faces.append(box.astype("int"))

    return detected_faces