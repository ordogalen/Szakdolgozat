import random
from sklearn import svm, metrics
from sklearn.utils import shuffle
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
import numpy as np


def load_aibo_data(filepath):
    """
    Make training dev and test Bunch
    :param filepath: The filepath for the .npy files
    :return: a shuffled Bunch with the labeled .npy files
    """
    dev_data = Bunch()
    dev_data["target_names"] = ["A", "E", "N", "P", "R"]
    dev_data["target"] = []

    b = 0
    c = True
    count = 0

    for dir in os.listdir(filepath):
        for files in os.listdir(filepath + "/" + dir):
            data = np.array(np.load(filepath + "/" + dir + "/" + files))
            count += 1
            if c:
                b = data
                c = False
            else:
                b = np.concatenate((b, data))

            if dir == "A":
                dev_data["target"] += [0]
            if dir == "E":
                dev_data["target"] += [1]
            if dir == "N":
                dev_data["target"] += [2]
            if dir == "P":
                dev_data["target"] += [3]
            if dir == "R":
                dev_data["target"] += [4]
        print(count)

    dev_data["data"] = b
    print("Process done")

    return dev_data


def load_aibo_data2(filepath):
    """
    Make training dev and test Bunch
    :param filepath: The filepath for the .npy files
    :return: a shuffled Bunch with the labeled .npy files
    """
    dev_data = Bunch()
    dev_data["target_names"] = ["A", "E", "N", "P", "R"]
    dev_data["target"] = []

    b = 0
    c = True
    count = 0

    for file in os.listdir(filepath + "/"):
        data = np.array(np.load(filepath + "/" + file))
        count += 1
        if c:
            b = data
            c = False
        else:
            b = np.concatenate((b, data))

        labelName = file.split("#")[1].split(".")[0]
        if labelName == "A":
            dev_data["target"] += [0]
        if labelName == "E":
            dev_data["target"] += [1]
        if labelName == "N":
            dev_data["target"] += [2]
        if labelName == "P":
            dev_data["target"] += [3]
        if labelName == "R":
            dev_data["target"] += [4]

    dev_data["data"] = b
    print("Process done")

    return dev_data


dev = load_aibo_data2('./meta/aibo/dev2')
train = load_aibo_data2('./meta/aibo/train_downsample2')
test = load_aibo_data2('./meta/aibo/test2')

X_dev = dev.data
y_dev = dev.target

X_train = train.data
y_train = train.target

X_test = test.data
y_test = test.target

print("Size of the dev data: ", len(X_dev))
print("Size of the dev label: ", len(y_dev))

print("Size of the train data: ", len(X_train))
print("Size of the train label: ", len(y_train))

print("Size of the test data: ", len(X_test))
print("Size of the test label: ", len(y_test))


# TODO megkell csinálni hogy a legjobb deven értékeljem ki a tesztet aztán annyi for loop


def train():
    """
    Get the best svm in the training process
    :return: Returns the best_svm, means that the svm scored the highest in accuracy
    """

    rbf_svc = svm.SVC(kernel='rbf', max_iter=-1, random_state=1)
    clf = make_pipeline(StandardScaler(), rbf_svc)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)
    acc = metrics.accuracy_score(y_dev, y_pred)
    print("Dev accuracy: " + str(acc))

    return clf


best = train()

y_pred = best.predict(X_test)

print("Test Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred, average='weighted'))

# Model Recall: what percentage of positive tuples are labelled as such
print("Recall:", metrics.recall_score(y_test, y_pred, average='weighted'))

print("Matrix:", metrics.confusion_matrix(y_test, y_pred))

print("Matrix:", metrics.confusion_matrix(y_test, y_pred).ravel())
