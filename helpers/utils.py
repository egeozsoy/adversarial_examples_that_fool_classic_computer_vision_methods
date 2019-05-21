from typing import List,Tuple

import numpy as np
from sklearn.utils import shuffle


# delete classes that are not in keep_classes, simplify cifar-10
def filter_classes(X:np.ndarray, y:np.ndarray, keep_classes:List[int]) -> Tuple[np.ndarray,np.ndarray]:
    logic_result = y == keep_classes[0]
    for keep_class in keep_classes[1:]:
        logic_result = np.logical_or(logic_result, y == keep_class)

    X = X[logic_result]
    y = y[logic_result]

    return X, y

# guarentee that a batch has almost equal class distribution
def get_balanced_batch(x:np.ndarray, y:np.ndarray, batch_size:int, classes:List[int]) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    shape = list(x.shape)
    shape[0] = batch_size
    batch_train_x = np.zeros(shape,dtype=x.dtype)
    batch_train_y = np.zeros(shape[0],dtype=y.dtype)
    shape[0] = len(classes) * 10
    batch_cv_x = np.zeros(shape,dtype=x.dtype)
    batch_cv_y = np.zeros(shape[0],dtype=y.dtype)

    class_size = batch_size // (len(classes))
    for c in classes:
        indices_mask = y == c
        x_c = x[indices_mask]
        y_c = y[indices_mask]

        random_indices = np.random.choice(x_c.shape[0] - 10, class_size, replace=False) # the first 10 elements are reserverd for cv
        batch_train_x[c * class_size:c * class_size + class_size] = x_c[random_indices + 10, :]
        batch_train_y[c * class_size:c * class_size + class_size] = y_c[random_indices + 10]

        for i in range(10):
            batch_cv_x[c*10+i] = x_c[i]
            batch_cv_y[c*10+i] = y_c[i]


    batch_train_x , batch_train_y = shuffle(batch_train_x,batch_train_y)
    batch_cv_x, batch_cv_y = shuffle(batch_cv_x, batch_cv_y)

    return batch_train_x,batch_train_y,batch_cv_x, batch_cv_y
