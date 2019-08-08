import os
import sys

import cv2
import numpy as np


def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    X = X - float(minX)
    X = X / float((maxX - minX))
    X = X * (high - low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)


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
                        # resize to given size (if given)
                    if sz is not None:
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError:
                    print("I/O error({})".format(IOError))
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c + 1
    return [X, y]


if __name__ == "__main__":
    out_dir = None

    if len(sys.argv) < 2:
        print("USAGE: facerec_demo.py </path/to/images> [</path/to/store/images/at>]")
        sys.exit()

    [X, y] = read_images(sys.argv[1])
    y = np.asarray(y, dtype=np.int32)

    if len(sys.argv) == 3:
        out_dir = sys.argv[2]

    model = cv2.face.EigenFaceRecognizer_create()
    model.train(np.asarray(X), np.asarray(y))

    [p_label, p_confidence] = model.predict(np.asarray(X[0]))
    print("Predicted label = {0} (confidence={1:.2f})".format(p_label, p_confidence))

    mean = model.getMean()
    eigenvectors = model.getEigenVectors()

    mean_norm = normalize(mean, 0, 255, dtype=np.uint8)
    mean_resized = mean_norm.reshape(X[0].shape)
    if out_dir is None:
        cv2.imshow("mean", mean_resized)
    else:
        cv2.imwrite("{}/mean.png".format(out_dir), mean_resized)

    for i in range(min(len(X), 16)):
        eigenvector_i = eigenvectors[:, i].reshape(X[0].shape)
        eigenvector_i_norm = normalize(eigenvector_i, 0, 255, dtype=np.uint8)

        if out_dir is None:
            cv2.imshow("{}/eigenface_{}".format(out_dir, i), eigenvector_i_norm)
        else:
            cv2.imwrite("{}/eigenface_{}.png".format(out_dir, i), eigenvector_i_norm)

    if out_dir is None:
        cv2.waitKey(0)
