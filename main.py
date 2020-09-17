import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
import facialtracking


def add_overlay(image, overlay, y1, x1):
    y1Slice = int(min(y1, 0) * -1)
    x1Slice = int(min(x1, 0) * -1)
    y2Slice = int(min(y1 + overlay.shape[0], image.shape[0]) - y1)
    x2Slice = int(min(x1 + overlay.shape[1], image.shape[1]) - x1)

    # print(f"y1Slice: {y1Slice}, x1Slice: {x1Slice}, y2Slice: {y2Slice}, x2Slice: {x2Slice}")

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


class MapMaskToFace:

    def __init__(self, mask, upscale, a, b, c):
        self.mask = mask
        self.upscale = upscale

        self.runningAverageScale = RunningAverage(a)
        self.runningAverageY = RunningAverage(b)
        self.runningAverageX = RunningAverage(c)

    def map(self, img, faceBox):
        startX, startY, endX, endY = faceBox

        maskRatio = self.mask.shape[0] / self.mask.shape[1]

        width = endX - startX
        width = self.runningAverageScale.add(width)

        widthAddition = width * self.upscale / 2
        width = int(width * (1 + self.upscale))

        height = int(width * maskRatio)

        y1 = startY + ((endY - startY - height) / 2)
        x1 = startX - widthAddition

        y1 = int(self.runningAverageY.add(y1))
        x1 = int(self.runningAverageX.add(x1))

        overlay = cv2.resize(self.mask, (width, height))

        img = add_overlay(img, overlay, y1, x1)

        return img


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

    mapMaskToFace = MapMaskToFace(
        cv2.imread("C:/Users/Owen/Programming/Python/WebcamScrewery/myface.png", -1),
        .55, 5, 2, 2
    )

    camera = None

    def get_frame(self):
        ret, frame = MJPEGServer.camera.read()

        faces = facialtracking.get_face(frame)
        for box in faces:
            frame = MJPEGServer.mapMaskToFace.map(frame, box)

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
