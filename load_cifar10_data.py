from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder


file = "cifar-10-batches-py/data_batch_1"
test_file = 'cifar-10-batches-py/test_batch'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def get_data():
    data = unpickle(file)
    data[b'data'] = np.transpose(data[b'data'].reshape(10000, 3, 32, 32), (0,2,3,1))
    enc = OneHotEncoder(handle_unknown='ignore')
    X = [[item] for item in data[b'labels']]
    enc.fit(X)
    X = enc.transform(X).toarray()
    data[b'labels'] = X
    return data

    # image = data[b'data'][2]
    # image = np.transpose(image, (1,2,0))
    # plt.imshow(image)
    # plt.show()

def get_test_data():
    data = unpickle(test_file)
    data[b'data'] = np.transpose(data[b'data'].reshape(10000, 3, 32, 32), (0,2,3,1))
    enc = OneHotEncoder(handle_unknown='ignore')
    X = [[item] for item in data[b'labels']]
    enc.fit(X)
    X = enc.transform(X).toarray()
    data[b'labels'] = X
    return data

    # image = data[b'data'][2]
    # image = np.transpose(image, (1,2,0))
    # plt.imshow(image)
    # plt.show()