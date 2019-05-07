import numpy as np


# delete classes that are not in keep_classes, simplify cifar-10
def filter_classes(X, y, keep_classes):
    logic_result = y == keep_classes[0]
    for keep_class in keep_classes[1:]:
        logic_result = np.logical_or(logic_result, y == keep_class)

    X = X[logic_result]
    y = y[logic_result]

    return X, y
