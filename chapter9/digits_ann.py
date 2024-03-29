import _pickle as cPickle
import gzip

import cv2
import numpy as np


def load_data():
    mnist = gzip.open('mnist.pkl.gz', 'rb')
    training_data, classification_data, test_data = cPickle.load(mnist, encoding='latin1')
    mnist.close()
    return training_data, classification_data, test_data


def vectorized_results(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def wrap_data():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_results(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    return training_data, validation_data, test_data


def create_ANN(hidden=20):
    ann = cv2.ml.ANN_MLP_create()
    ann.setLayerSizes(np.array([784, hidden, 10]))
    ann.setTrainMethod(cv2.ml.ANN_MLP_SIGMOID_SYM)
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
    ann.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1))
    return ann


def train(ann, samples=10000, epochs=1):
    tr, val, test = wrap_data()

    for x in range(epochs):
        counter = 0
        for img in tr:

            if counter > samples:
                break
            if counter % 1000 == 0:
                print("epoch {}: trained {}/{}".format(x, counter, samples))
            counter += 1

            data, digit = img
            ann.train(np.array([data.ravel()], dtype=np.float32), cv2.ml.ROW_SAMPLE,
                      np.array([digit.ravel()], dtype=np.float32))
        print("epoch {} complete".format(x))
    return ann, test


def test(ann, test_data):
    sample = np.array(test_data[0][0].ravel(), dtype=np.float32).reshape(28, 28)
    cv2.imshow("sample", sample)
    cv2.waitKey()
    print(ann.predict(np.array([test_data[0][0].ravel()], dtype=np.float32)))


def predict(ann, sample):
    resized = sample.copy()
    rows, cols = resized.shape

    if (rows != 28 or cols != 28) and rows * cols > 0:
        resized = cv2.resize(resized, (28, 28), interpolation=cv2.INTER_CUBIC)

    return ann.predict(np.array([resized.ravel()], dtype=np.float32))
