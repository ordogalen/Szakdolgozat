"""
@author: ordogalen
"""

from sklearn import svm, metrics
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

a = set()
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

    for file in os.listdir(filepath + "/"):
        data = np.array(np.load(filepath + "/" + file))
        count += 1
        if c:
            b = data
            c = False
        else:
            b = np.concatenate((b, data))

        labelName = file.split("#")[1].split(".")[0]
        a.add(labelName)
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


dev = load_aibo_data('./meta/aibo/dev')
train = load_aibo_data('./meta/aibo/train_downsample')
test = load_aibo_data('./meta/aibo/test')
print(a)
X_dev = dev.data
y_dev = dev.target

X_train = train.data
y_train = train.target

X_test = test.data
y_test = test.target

print("Size of the dev data: ", len(X_dev))

print("Size of the train data: ", len(X_train))

print("Size of the test data: ", len(X_test))


def train(complexity, kernel, gamma):
    """
    Get the best svm in the training process
    :return: Returns the best_svm, means that the svm scored the highest in accuracy
    """

    svc = svm.SVC(kernel=kernel, max_iter=-1, C=complexity, decision_function_shape='ovo', cache_size=4000, gamma=gamma)
    clf = make_pipeline(StandardScaler(), svc)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)
    acc = metrics.accuracy_score(y_dev, y_pred)
    print("Dev accuracy: " + str(acc * 100))
    print("DEV PARAMETERS: \n"+metrics.classification_report(y_dev, y_pred))
    print("----")
    return clf


def predict(complexity, kernel, gamma):
    """
    For these parameters check this site: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    :param complexity: SVM complexity parameter
    :param kernel: SVM kernel parameter
    :param gamma: SVM gamma parameter
    """
    best = train(complexity, kernel, gamma)

    y_pred = best.predict(X_test)

    print("Test Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100)
    # Model Recall: what percentage of positive tuples are labelled as such
    print("Precision:", metrics.precision_score(y_test, y_pred, average='weighted') * 100)

    print("F1 recall:", metrics.f1_score(y_test, y_pred, average='micro') * 100)

    print("Matrix:", metrics.confusion_matrix(y_test, y_pred))

    print("TEST PARAMETERS: \n"+metrics.classification_report(y_test, y_pred))

    class_names = ["A","E","N","P","R"]

    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            best,
            X_test,
            y_test,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()

# for i in [0.001,0.01,0.1,1.0]:

print("LINEAR kernel")
predict(0.01, "linear", 'scale')
print("---------------------------")

# for i in [0.1,1.0,10.0]:
#     for j in [0.0001,0.001,0.01,0.1,1.0]:
# print("c = " + str(i)+" g = "+str(j))

print("RBF kernel")
predict(1.0,"rbf",0.001)
print("--------------------")


