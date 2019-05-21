import os
from typing import Tuple

import cv2
import numpy as np
from sklearn.datasets import fetch_openml

datasets_root = 'datasets'


def unpickle(file:str):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(get_test:bool=False) -> Tuple[np.ndarray,np.ndarray]:
    file1:str = os.path.join(datasets_root, "cifar-10-batches-py/data_batch_1")
    file2:str = os.path.join(datasets_root, "cifar-10-batches-py/data_batch_2")
    file3:str = os.path.join(datasets_root, "cifar-10-batches-py/data_batch_3")
    file4:str = os.path.join(datasets_root, "cifar-10-batches-py/data_batch_4")
    file5:str = os.path.join(datasets_root, "cifar-10-batches-py/data_batch_5")
    test_file:str = os.path.join(datasets_root, 'cifar-10-batches-py/test_batch')

    if not get_test:
        data1 = unpickle(file1)
        data2 = unpickle(file2)
        data3 = unpickle(file3)
        data4 = unpickle(file4)
        data5 = unpickle(file5)
        data = {b'data': np.concatenate((data1[b'data'], data2[b'data'], data3[b'data'], data4[b'data'], data5[b'data'])),
                b'labels': np.concatenate((data1[b'labels'], data2[b'labels'], data3[b'labels'], data4[b'labels'], data5[b'labels']))}
        data[b'data'] = np.transpose(data[b'data'].reshape(-1, 3, 32, 32), (0, 2, 3, 1))
        return np.uint8(data[b'data']), np.float32(data[b'labels'])

    else:
        data = unpickle(test_file)
        data[b'data'] = np.transpose(data[b'data'].reshape(-1, 3, 32, 32), (0, 2, 3, 1))
        return np.uint8(data[b'data']), np.float32(data[b'labels'])


def load_mnist_data() -> Tuple[np.ndarray,np.ndarray]:
    mnist_folder:str = os.path.join(datasets_root, 'mnist')
    if not os.path.exists(mnist_folder):
        # data not found
        os.mkdir(mnist_folder)
        X, y = fetch_openml('mnist_784', return_X_y=True)
        X = X.reshape(-1, 28, 28)
        X = np.uint8(X)
        y = np.uint8(y)
        np.save(os.path.join(mnist_folder, 'X.npy'), X)
        np.save(os.path.join(mnist_folder, 'y.npy'), y)

    else:
        # data found
        X = np.load(os.path.join(mnist_folder, 'X.npy'))
        y = np.load(os.path.join(mnist_folder, 'y.npy'))

    return X, y


def load_fmnist_data() -> Tuple[np.ndarray,np.ndarray]:
    fmnist_folder = os.path.join(datasets_root, 'fmnist')
    if not os.path.exists(fmnist_folder):
        # data not found
        os.mkdir(fmnist_folder)
        X, y = fetch_openml('Fashion-MNIST', return_X_y=True)
        X = X.reshape(-1, 28, 28)
        X = np.uint8(X)
        y = np.uint8(y)
        np.save(os.path.join(fmnist_folder, 'X.npy'), X)
        np.save(os.path.join(fmnist_folder, 'y.npy'), y)

    else:
        # data found
        X = np.load(os.path.join(fmnist_folder, 'X.npy'))
        y = np.load(os.path.join(fmnist_folder, 'y.npy'))

    return X, y


def load_imagenette_data(image_size:int, get_test:bool=False) -> Tuple[np.ndarray,np.ndarray]:
    imagenette_folder = os.path.join(datasets_root, 'imagenette-160')
    folder_to_load = 'train' if get_test == False else 'val'
    X_name = 'X_{}.npy'.format(folder_to_load)
    y_name = 'y_{}.npy'.format(folder_to_load)

    if os.path.exists(os.path.join(imagenette_folder, X_name)):
        print('Loading preprocessed data')
        X = np.load(os.path.join(imagenette_folder, X_name))
        y = np.load(os.path.join(imagenette_folder, y_name))

    else:
        folder_to_load_path = os.path.join(imagenette_folder, folder_to_load)

        X = []
        y = []

        class_index = 0

        for subfolder in sorted(os.listdir(folder_to_load_path)):
            sub_folder_path = os.path.join(folder_to_load_path, subfolder)
            if not os.path.isdir(sub_folder_path):
                continue
            for img in os.listdir(sub_folder_path):
                # make sure actually picture
                img_path = os.path.join(sub_folder_path, img)
                if '.JPEG' in img_path:
                    current_image = cv2.resize(cv2.imread(img_path), (image_size, image_size))
                    X.append(current_image)
                    y.append(class_index)

            class_index += 1

        X = np.array(X, dtype=np.uint8)
        y = np.array(y, dtype=np.uint8)
        np.save(os.path.join(imagenette_folder, X_name), X)
        np.save(os.path.join(imagenette_folder, y_name), y)

    return X, y


def load_inria_dataset(image_size:int, get_test:bool=False) -> Tuple[np.ndarray,np.ndarray]:
    inria_folder = os.path.join(datasets_root, 'INRIA')
    folder_to_load = 'Train' if get_test == False else 'Test'
    X_name = 'X_{}.npy'.format(folder_to_load)
    y_name = 'y_{}.npy'.format(folder_to_load)

    if os.path.exists(os.path.join(inria_folder, X_name)):
        print('Loading preprocessed data')
        X = np.load(os.path.join(inria_folder, X_name))
        y = np.load(os.path.join(inria_folder, y_name))

    else:
        folder_to_load_path = os.path.join(inria_folder, folder_to_load)

        positive_load_path = os.path.join(folder_to_load_path, 'pos')
        negative_load_path = os.path.join(folder_to_load_path, 'neg')

        X = []
        y = []

        for img in os.listdir(positive_load_path):
            if '.jpg' in img or '.png' in img:
                img_path = os.path.join(positive_load_path, img)
                current_image = cv2.resize(cv2.imread(img_path), (image_size, image_size))
                X.append(current_image)
                y.append(1)

        # to balance the dataset, duplicate the X
        X.extend(X)
        y.extend(y)

        for img in os.listdir(negative_load_path):
            if '.jpg' in img or '.png' in img:
                img_path = os.path.join(negative_load_path, img)
                current_image = cv2.resize(cv2.imread(img_path), (image_size, image_size))
                X.append(current_image)
                y.append(0)

        X = np.array(X, dtype=np.uint8)
        y = np.array(y, dtype=np.uint8)
        np.save(os.path.join(inria_folder, X_name), X)
        np.save(os.path.join(inria_folder, y_name), y)

    return X, y


def load_data(image_size:int, get_test:bool=False, dataset_name:str='imagenette') -> Tuple[np.ndarray,np.ndarray]:
    if dataset_name == 'mnist':
        return load_mnist_data()
    elif dataset_name == 'fmnist':
        return load_fmnist_data()
    elif dataset_name == 'imagenette':
        return load_imagenette_data(image_size, get_test=get_test)
    elif dataset_name == 'cifar-10':
        return load_cifar_10_data(get_test=get_test)
    elif dataset_name == 'inria':
        return load_inria_dataset(image_size, get_test=get_test)

    raise Exception('Not a valid dataset')
