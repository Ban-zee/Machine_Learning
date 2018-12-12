import numpy as np
import cv2
import mahotas
import matplotlib.image as img
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os


def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    np.shape(hist)
    return hist.flatten()
def feature_extractor(path):
    image = img.imread(path)
    image = cv2.resize(image,(200,200))
    image_hu = fd_hu_moments(image)
    image_haralick = fd_haralick(image)
    image_histogram = fd_histogram(image)
    features = np.hstack([image_histogram,image_hu,image_haralick])
    return features
def training_data_generator(classifier):
    train_path = '/home/banzee/Desktop/ML/CV/dataset/training_set'
    train_labels = os.listdir(train_path)
    global_features = []
    labels = []
    for x in train_labels:
        image_path = str(train_path)+'/'+str(x)
        for images in os.listdir(image_path):
            image_parameter = image_path+'/'+str(images)
            features = feature_extractor(path=image_parameter)
            global_features.append(features)
            labels.append(x)
    classifier.fit(global_features,labels)
    return classifier
def tester():
    bayesian_predictor = GaussianNB()
    forest_predictor = RandomForestClassifier(n_estimators=300)
    support_vector = SVC()
    x = training_data_generator(bayesian_predictor)
    y = training_data_generator(forest_predictor)
    z = training_data_generator(support_vector)
    test_path = '/home/banzee/Desktop/ML/CV/dataset/test_set'''
    for image in os.listdir(test_path):
        testpath = test_path+'/'+str(image)
        for file in os.listdir(testpath):
            image_path = testpath+'/'+str(file)
            feature = feature_extractor(image_path)
            feature = np.reshape(feature, (1, -1))
            print('The Bayesian gives: ', x.predict(feature))
            print('The Ensemble gives:',y.predict(feature) )
            print('The SVM gives: ', z.predict(feature))
            print('True value: ', image)
tester()

