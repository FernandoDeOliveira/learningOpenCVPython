import os
import sys
import cv2
import numpy as np
# from chapter5 .face_recognition import read_images


def read_images(path, sz=None):
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if filename == ".directory":
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    if im is None:
                        print("image " + filepath + " is none")
                    else:
                        print(filepath)
                    # resize to given size (if given)
                    if sz is not None:
                        im = cv2.resize(im, (200, 200))

                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError:
                    print("I/O error({})".format(IOError))
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            print(c)
            c += 1

    print(y)
    return [X, y]

def face_rec():
    names = ['Luis Fernando', 'Não eu', "outrem"]
    if len(sys.argv) < 2:
        print("USAGE: facerec_demo.py </path/to/images> [</path/to/store/images/at>]")
        sys.exit()

    [X, y] = read_images(sys.argv[1])
    y = np.asarray(y, dtype=np.int32)

    if len(sys.argv) == 3:
        out_dir = sys.argv[2]

    model = cv2.face.EigenFaceRecognizer_create()
    model.train(np.asarray(X), np.asarray(y))
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('../cascades/haarcascade_frontalface_default.xml')

    while True:
        _, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[x:x+w, y:y+h]

            try:
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi)
                print("label: {}, Confidence: {}".format(params[0], params[1]))
                cv2.putText(img, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                if params[0] == 0:
                    cv2.imwrite('face_rec.jpg', img)
            except:
                continue
        cv2.imshow("camera", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_rec()

