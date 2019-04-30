from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# return flattened image
def feature_extractor(images):
    return images.reshape(images.shape[0], -1)


digits = datasets.load_digits()
data = digits.images
labels = digits.target

X, y = fetch_openml('CIFAR_10', version=1, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,np.int8(y),test_size=0.2,shuffle=True)
model = LinearSVC(max_iter=10000)
# model = RandomForestClassifier(n_estimators=100)
# model = LogisticRegression()

print('Starting model training')
model.fit(feature_extractor(X_train/255), y_train)
print('Training set performance {}'.format(model.score(feature_extractor(X_train/255), y_train)))
print('Testing set performance {}'.format(model.score(feature_extractor(X_test/255), y_test)))
