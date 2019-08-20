import cv2
import numpy as np
import os.path as path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", help="m (or nothing) for meanShift and c for camshift")
args = vars(parser.parse_args())

font = cv2.FONT_HERSHEY_SIMPLEX


def center(points):
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    return np.array([np.float(x), np.float32(y)], np.float32)


class Pedestrian:
    def __init__(self, id, frame, track_window):
        self.id = int(id)
        self.track_window = track_window
        x, y, w, h = track_window
        self.roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        roi_hist = cv2.calcHist([self.roi], [0], None, [16], [0, 180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               np.float32) * 0.03

        self.measurement = np.array((2, 1), np.float32)
        self.prediction = np.zeros((2, 1), np.float32)
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.center = None
        self.update(frame)

    def __del__(self):
        print("pedestrian {} destroyed".format(self.id))

    def update(self, frame, algorithm='m'):
        print("updating {}".format(self.id))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_project = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        if algorithm == "c":
            ret, self.track_window = cv2.CamShift(back_project, self.track_window, self.term_crit)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            self.center = center(pts)
            cv2.polylines(frame, [pts], True, 255, 1)

        else:
            ret, self.track_window = cv2.meanShift(back_project, self.track_window, self.term_crit)
            x, y, w, h = self.track_window
            self.center = center([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

        self.kalman.correct(self.center)
        prediction = self.kalman.predict()
        cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 4, (0, 255, 0), -1)

        text = "ID: {} -> {}".format(self.id, self.center)
        cv2.putText(frame, text, (11, (self.id + 1) * 25 + 1), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, text, (10, (self.id + 1) * 25), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)


def main():
    algoritm = str(input('with algorithm? '))
    algoritm = algoritm if algoritm=='c' else 'm'
    camera = cv2.VideoCapture('768x576.avi')
    history = 20
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    bs.setHistory(history)

    cv2.namedWindow("surveillance")
    pedestrians = {}
    firstFrame = True
    frames = 0
    while True:
        print("----------------------- FRAME {} -----------------------".format(frames))

        grabbed, frame = camera.read()
        if grabbed is False:
            print("failed to grab frame")
            break

        fgmask = bs.apply(frame)
        if frames < history:
            frames += 1
            continue

        th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,3)), iterations=2)
        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter = 0
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                if firstFrame:
                    pedestrians[counter] = Pedestrian(counter, frame, (x, y, w, h))
                counter +=1
        for i, p in pedestrians.items():
            p.update(frame, algoritm)

        firstFrame = False
        frames +=1

        cv2.imshow('surveillance', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()
    camera.release()

if __name__ == '__main__':
    main()

