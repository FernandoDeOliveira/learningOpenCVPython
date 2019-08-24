from random import randint

import cv2
import numpy as np

RECORDS = 100000
TESTS = 100
EPOCHS = 5


animals_net = cv2.ml.ANN_MLP_create()
animals_net.setTrainMethod(cv2.ml.ANN_MLP_RPROP | cv2.ml.ANN_MLP_UPDATE_WEIGHTS)
animals_net.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
animals_net.setLayerSizes(np.array([3, 8, 4]))
animals_net.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))


# input arrays: weight, lenght, teeth
#  output arrays: dog, eagle, dolphin and dragon


def dog_sample():
    return [randint(5, 20), 1, randint(38, 42)]


def dog_class():
    return [1, 0, 0, 0]


def condor_sample():
    return [randint(3, 13), 3, 0]


def condor_class():
    return [0, 1, 0, 0]


def dolphin_sample():
    return [randint(30, 1900), randint(5, 15), randint(80, 100)]


def dolphin_class():
    return [0, 0, 1, 0]


def dragon_sample():
    return [randint(1200, 1800), randint(15, 40), randint(110, 180)]


def dragon_class():
    return [0, 0, 0, 1]


def record(sample, classification):
    return np.array([sample], dtype=np.float32), np.array([classification], dtype=np.float32)


records = []
for x in range(RECORDS):
    records.append(record(dog_sample(), dog_class()))
    records.append(record(condor_sample(), condor_class()))
    records.append(record(dolphin_sample(), dolphin_class()))
    records.append(record(dragon_sample(), dragon_class()))

for e in range(EPOCHS):
    print("epoch {}".format(e))
    for t, c in records:
        animals_net.train(t, cv2.ml.ROW_SAMPLE, c)

dog_results = 0
for x in range(0, TESTS):
    clas = int(animals_net.predict(np.array([dog_sample()], dtype=np.float32))[0])
    print("class: {}".format(clas))
    if clas == 0:
        dog_results += 1

condor_results = 0
for x in range(0, TESTS):
    clas = int(animals_net.predict(np.array([condor_sample()], dtype=np.float32))[0])
    print("class: {}".format(clas))
    if clas == 1:
        condor_results += 1

dolphin_results = 0
for x in range(0, TESTS):
    clas = int(animals_net.predict(np.array([dolphin_sample()], dtype=np.float32))[0])
    print("class: {}".format(clas))
    if clas == 2:
        dolphin_results += 1

dragon_results = 0
for x in range(0, TESTS):
    clas = int(animals_net.predict(np.array([dragon_sample()], dtype=np.float32))[0])
    print("class: {}".format(clas))
    if clas == 3:
        dragon_results += 1

print("dog accuracy: {}".format(dog_results))
print("condor accuracy: {}".format(condor_results))
print("dolphin accuracy: {}".format(dolphin_results))
print("dragon accuracy: {}".format(dragon_results))
