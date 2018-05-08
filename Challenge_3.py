import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import data, exposure, img_as_float, io
from skimage.transform import rescale, resize
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.externals import joblib
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
        "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
        "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


def run_classifiers(filepath):
    x_train, x_test, y_train, y_test = get_train_test(filepath)
    for name, classifier in zip(names, classifiers):
        print("Working on: " + name)
        classifier.fit(x_train, y_train)
        score = classifier.score(x_test, y_test)
        print(score)
        print("\n")
        filename = name + ".pkl"
        joblib.dump(clf, os.path.join(filepath, filename)) 


def get_train_test(filepath):
    x = np.load(os.path.join(filepath, "x_image_arrays.npy"))
    y = np.load(os.path.join(filepath, "y_image_labels.npy"))
    x = x.reshape((len(x), -1))
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
    print(set(y_test))
    return x_train, x_test, y_train, y_test


def svm_classifier(x_train, x_test, y_train, y_test):
    classifier = SVC(verbose=True)
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    print("Accuracy:", accuracy_score(y_test,prediction))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    args = parser.parse_args()
    if args.dataset:
        run_classifiers(args.dataset)
