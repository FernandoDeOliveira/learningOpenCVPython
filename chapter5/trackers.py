import cv2
from chapter5 import utils, rects


class Face:
    def __init__(self):
        self.faceRect = None
        self.leftEyeRect = None
        self.rightEyeRect = None
        self.noseRect = None
        self.mouthRect = None


class FaceTracker:
    def __init__(self, scaleFactor=1.2, minNeighbors=2, flags=cv2.CASCADE_SCALE_IMAGE):
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.flags = flags

        self._faces = []

        self._faceClassifier = cv2.CascadeClassifier('../cascades/haarcascade_frontalface_alt.xml')
        self._eyeClassifier = cv2.CascadeClassifier('../cascades/haarcascade_eye.xml')
        self._noseClassifier = cv2.CascadeClassifier('../cascades/haarcascade_mcs_nose.xml')
        self._mouthClassifier = cv2.CascadeClassifier('../cascades/haarcascade_mcs_mouth.xml')

    @property
    def faces(self):
        return self._faces

    def update(self, image):
        self._faces = []

        if utils.isGray(image):
            image = cv2.equalizeHist(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(image, image)

        minSize = utils.widthHeightDividedBy(image, 8)

        facesRects = self._faceClassifier.detectMultiScale(image, self.scaleFactor,
                                                           self.minNeighbors, self.flags)
        if facesRects is not None:
            for faceRect in facesRects:

                face = Face()
                face.faceRect = faceRect

                x, y, w, h = faceRect

                # Seek an eye in the upper-left part of the face.
                searchRect = (int(x + w / 7), y, int(w * 2 / 7), int(h / 2))
                face.leftEyeRect = self._detectOneObject(self._eyeClassifier, image, searchRect, 64)

                # Seek an eye in the upper-right part of the face.
                searchRect = (int(x + w * 4 / 7), int(y), int(w * 2 / 7), int(h / 2))
                face.rightEyeRect = self._detectOneObject(self._eyeClassifier, image, searchRect, 64)

                # Seek a nose in the middle part of the face.
                searchRect = (int(x + w / 4), int(y + h / 4), int(w / 2), int(h / 2))
                face.noseRect = self._detectOneObject(self._noseClassifier, image, searchRect, 32)

                # Seek a mouth in the lower-middle part of the face.
                searchRect = (int(x + w / 6), int(y + h * 2 / 3), int(w * 2 / 3), int(h / 3))
                face.mouthRect = self._detectOneObject(self._mouthClassifier, image, searchRect, 16)

                self._faces.append(face)

    def _detectOneObject(self, classifier, image, rect, imageSizeToMinSizeRatio):
        x, y, w, h = rect
        minSize = utils.widthHeightDividedBy(image, imageSizeToMinSizeRatio)

        subImage = image[y:y+h, x:x+w]

        subRects = classifier.detectMultiScale(subImage, self.scaleFactor, self.minNeighbors, self.flags, minSize)

        if len(subRects) == 0:
            return None

        subX, subY, subW, subH = subRects[0]
        return (x+subX, y+subY, subW, subH)

    def drawDebugRects(self, image):

        if utils.isGray(image):
            faceColor = 255
            leftRyeColor = 255
            rightEyeColor = 255
            noseColor = 255
            mouthColor = 255
        else:
            faceColor = (255, 255, 255)
            leftRyeColor = (0, 0, 255)
            rightEyeColor = (0, 255, 255)
            noseColor = (0, 255, 0)
            mouthColor = (255, 0, 0)

        for face in self.faces:
            rects.outlineRect(image, face.faceRect, faceColor)
            rects.outlineRect(image, face.leftEyeRect, leftRyeColor)
            rects.outlineRect(image, face.rightEyeRect, rightEyeColor)
            rects.outlineRect(image, face.noseRect, noseColor)
            rects.outlineRect(image, face.mouthRect, mouthColor)
