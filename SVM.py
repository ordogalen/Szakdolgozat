
from sklearn import svm, metrics
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
import numpy as np
from sklearn.metrics import accuracy_score



def load_aibo_data(filepath):
    dev_data = Bunch()
    dev_data["target_names"] = ["A", "E", "N", "P", "R"]
    dev_data["target"] = []
    b = 0
    c = True
    for dir in os.listdir(filepath):
        for files in os.listdir(filepath + "/" + dir):
            data = np.array(np.load(filepath + "/" + dir + "/" + files))
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

    dev_data["data"] = b
    return dev_data




dev = load_aibo_data('./meta/aibo/dev')
train = load_aibo_data('./meta/aibo/train_downsample')
test = load_aibo_data('./meta/aibo/test')


X_dev = dev.data
y_dev = dev.target
X_train = train.data
y_train = train.target
# TODO testet is lehúzni done
X_test = test.data
y_test = test.target


# TODO megkell csinálni hogy a legjobb deven értékeljem ki a tesztet aztán annyi for loop

def train():
    best_svm = 0
    best_acc = 0
    for i in range(500):
        rbf_svc = svm.SVC(kernel='rbf', max_iter=i)
        clf = make_pipeline(StandardScaler(), rbf_svc)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_dev)
        acc = metrics.accuracy_score(y_dev, y_pred)
        print(str(i) + ". iteration")
        print("accuracy: " + str(acc))
        if acc > best_acc:
            best_svm = clf
            best_acc = acc

    return best_svm


asd = train()

y_pred = asd.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred, average='weighted'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred, average='weighted'))

print("Matrix:", metrics.confusion_matrix(y_test, y_pred))

print("Matrix:", metrics.confusion_matrix(y_test, y_pred).ravel())
