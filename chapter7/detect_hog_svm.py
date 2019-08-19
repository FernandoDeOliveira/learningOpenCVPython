import cv2
import numpy as np
from os.path import join

# dir with train data gitignored
# to download: http://l2r.cs.uiuc.edu/~cogcomp/Data/Car/CarData.tar.gz
datapath = "trainImages/"


def path(cls, i):
    return "{datapath}/{cls}{count}.pgm".format(datapath=datapath, cls=cls, count=i+1)


pos, neg = "pos-", "neg-"

detect = cv2.xfeatures2d.SIFT_create()
extract = cv2.xfeatures2d.SIFT_create()

flann_params = dict(algorithm=1, trees=5)
flann = cv2.FlannBasedMatcher(flann_params, {})

bow_kmean_trainer = cv2.BOWKMeansTrainer(40)
extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann)


def extract_sift(fn):
    im = cv2.imread(fn, 0)
    return extract.compute(im, detect.detect(im))[1]


for i in range(8):
    bow_kmean_trainer.add(extract_sift(path(pos, i)))
    bow_kmean_trainer.add(extract_sift(path(neg, i)))

voc = bow_kmean_trainer.cluster()
extract_bow.setVocabulary(voc)


def bow_features(fn):
    im = cv2.imread(fn, 0)
    return extract_bow.compute(im, detect.detect(im))


traindata, trainlabels = [],  []

for i in range(20):
    traindata.extend(bow_features(path(pos, i)))
    trainlabels.append(1)
    traindata.extend(bow_features(path(neg, i)))
    trainlabels.append(-1)

svm = cv2.ml.SVM_create()
svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))


def predict(fn):
    f = bow_features(fn)
    p = svm.predict(f)
    print ("fn={} \t {}".format(fn, p[1][0][0]))
    return p


car, notcar = '../images/car.jpg', '../images/bb.jpg'
car_img = cv2.imread(car)
notcar_img = cv2.imread(notcar)

car_predict = predict(car)
not_car_predict = predict(notcar)

font = cv2.FONT_HERSHEY_SIMPLEX

if car_predict[1][0][0] == 1.0:
    cv2.putText(car_img, 'Car detected', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

if not_car_predict[1][0][0] == -1.0:
    cv2.putText(notcar_img, 'Car not detected', (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

cv2.imshow('bow + svm success', car_img)
cv2.imshow('bow + svm failure', notcar_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

