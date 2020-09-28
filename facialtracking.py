import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils


net = cv2.dnn.readNetFromCaffe(
    "C:/Users/Owen/Programming/Python/WebcamScrewery/model/deploy.prototxt.txt",
    "C:/Users/Owen/Programming/Python/WebcamScrewery/model/res10_300x300_ssd_iter_140000.caffemodel")

predictor = dlib.shape_predictor(
    "C:/Users/Owen/Programming/Python/WebcamScrewery/model/shape_predictor_68_face_landmarks.dat")


def get_face(img, getDetections=False):
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    if getDetections:
        return detections

    detected_faces = []

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            detected_faces.append(box.astype("int"))

    return detected_faces


def get_face_keypoints(img):
    faces = get_face(img)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEyePositions = []
    rightEyePositions = []

    for face in faces:
        rectangle = dlib.rectangle(
            face[0],
            face[1],
            face[2],
            face[3]
        )
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        shape = predictor(gray, rectangle)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rectangle)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        leftPos = map(sum, zip(*shape[lStart:lEnd]))
        leftPos = [x/6 for x in leftPos]
        leftEyePositions.append(leftPos)

        rightPos = map(sum, zip(*shape[rStart:rEnd]))
        rightPos = [x / 6 for x in rightPos]
        rightEyePositions.append(rightPos)

    return zip(leftEyePositions, rightEyePositions)
