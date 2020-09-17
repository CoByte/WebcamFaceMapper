import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer


def add_overlay(image, overlay, y1, x1):
    y1Slice = int(min(y1, 0) * -1)
    x1Slice = int(min(x1, 0) * -1)
    y2Slice = int(min(y1 + overlay.shape[0], image.shape[0]) - y1)
    x2Slice = int(min(x1 + overlay.shape[1], image.shape[1]) - x1)

    print(f"y1Slice: {y1Slice}, x1Slice: {x1Slice}, y2Slice: {y2Slice}, x2Slice: {x2Slice}")

    if y1Slice > overlay.shape[0] or \
        x1Slice > overlay.shape[1] or \
        y2Slice <= 0 or \
        x2Slice <= 0:

        return image

    y1 = max(y1, 0)
    x1 = max(x1, 0)
    y2 = y1 - y1Slice + y2Slice
    x2 = x1 - x1Slice + x2Slice

    overlayAlpha = overlay[y1Slice:y2Slice, x1Slice:x2Slice, 3] / 255.0
    imageAlpha = 1.0 - overlayAlpha

    for c in range(0, 3):
        image[y1:y2, x1:x2, c] = (overlayAlpha * overlay[y1Slice:y2Slice, x1Slice:x2Slice, c] +
                                  imageAlpha * image[y1:y2, x1:x2, c])

    return image


class RunningAverage:

    def __init__(self, targetSize):
        self.targetSize = targetSize
        self.list = []

    def add(self, value):
        if len(self.list) > self.targetSize:
            del self.list[0]
        self.list.append(value)

        return self.get()

    def get(self):
        return sum(self.list) / len(self.list)


class MJPEGServer(BaseHTTPRequestHandler):
    """
    A simple mjpeg server that either publishes images directly from a camera
    or republishes images from another pygecko process.
    """

    net = cv2.dnn.readNetFromCaffe(
        "C:/Users/Owen/Programming/Python/WebcamScrewery/model/deploy.prototxt.txt",
        "C:/Users/Owen/Programming/Python/WebcamScrewery/model/res10_300x300_ssd_iter_140000.caffemodel")

    mask = cv2.imread("C:/Users/Owen/Programming/Python/WebcamScrewery/myface.png", -1)
    upscale = .55
    runningAverageScale = RunningAverage(5)
    runningAverageX = RunningAverage(2)
    runningAverageY = RunningAverage(2)

    camera = None

    def get_frame(self):
        ret, frame = MJPEGServer.camera.read()

        maskRatio = MJPEGServer.mask.shape[0] / MJPEGServer.mask.shape[1]

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        MJPEGServer.net.setInput(blob)
        detections = MJPEGServer.net.forward()

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                width = endX - startX
                width = MJPEGServer.runningAverageScale.add(width)

                widthAddition = width * MJPEGServer.upscale / 2
                width = int(width * (1 + MJPEGServer.upscale))

                height = int(width * maskRatio)

                y1 = startY + ((endY - startY - height) / 2)
                x1 = startX - widthAddition

                y1 = int(MJPEGServer.runningAverageY.add(y1))
                x1 = int(MJPEGServer.runningAverageX.add(x1))

                overlay = cv2.resize(MJPEGServer.mask, (width, height))

                frame = add_overlay(frame, overlay, y1, x1)

                # width = endX - startX
                #
                # width = MJPEGServer.runningAverageScale.add(width)
                #
                # widthAddition = width * (MJPEGServer.upscale / 2)
                # width = int(width * (1 + MJPEGServer.upscale))
                #
                # height = int(width * maskRatio)
                #
                # mid = ((endY - startY) / 2) + startY
                #
                # x1 = startX - widthAddition
                # y1 = mid - (height / 2)
                #
                # x1 = int(MJPEGServer.runningAverageX.add(x1))
                # y1 = int(MJPEGServer.runningAverageY.add(y1))
                #
                # x2 = int(x1 + width)
                # y2 = int(y1 + height)
                #
                # tempMask = cv2.resize(MJPEGServer.mask, (width, height))
                #
                # maskAlpha = tempMask[:, :, 3] / 255.0
                # frameAlpha = 1.0 - maskAlpha
                #
                # for c in range(0, 3):
                #     frame[y1:y2, x1:x2, c] = (maskAlpha * tempMask[:, :, c] +
                #                               frameAlpha * frame[y1:y2, x1:x2, c])

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        return ret, frame

    def do_GET(self):
        print('connection from:', self.address_string())

        if self.path == '/mjpeg':
        #if self.path == '/':
            self.send_response(200)
            self.send_header(
                'Content-type',
                'multipart/x-mixed-replace; boundary=--jpgboundary'
            )
            self.end_headers()

            while True:
                if MJPEGServer.camera:
                    # print('cam')
                    ret, img = self.get_frame()

                else:
                    raise Exception('Error, camera not setup')

                if not ret:
                    print('no image from camera')
                    continue

                ret, jpg = cv2.imencode('.jpg', img)
                #print(cv2.imwrite('C:/Users/Owen/Programming/Python/WebcamScrewery/testing.jpg', img))

                # print 'Compression ratio: %d4.0:1'%(compress(img.size,jpg.size))
                self.wfile.write("--jpgboundary\r\n".encode("utf-8"))
                self.send_header('Content-type', 'image/jpeg')
                # self.send_header('Content-length',str(tmpFile.len))
                self.send_header('Content-length', str(jpg.size))
                self.end_headers()
                self.wfile.write(jpg.tobytes())
                # time.sleep(0.05)

        elif self.path == '/':

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            with open("html-request.html") as file:
                fileContent = file.read()
                fileContent = fileContent.replace("\n", "")
                fileContent = fileContent.encode("utf-8")

                self.wfile.write(fileContent)

        else:
            print('error', self.path)
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head></head><body>')
            self.wfile.write(bytes('<h1>{0!s} not found</h1>'.format(self.path), 'utf-8'))
            self.wfile.write(b'</body></html>')


address = ("", 8080)

webcam = cv2.VideoCapture(0)

with HTTPServer(address, MJPEGServer) as server:
    MJPEGServer.camera = webcam
    print(f"Serving at localhost, port {address[1]}")
    server.serve_forever()
