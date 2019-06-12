from copy import deepcopy

import numpy as np

from .utils import gpu_available

if not gpu_available():
    import os
    # we can use plaidml to accelerate local training(support for amd gpus)
    print('Switching to PlaidML, because no GPU was found')
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from configurations import input_dropout

X_shape = None
class_count = None

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

def make_resnet_model():
    # model  = keras.applications.inception_v3.InceptionV3(include_top=True, weights=None, input_shape=X_shape[1:], classes=class_count)
    # model  = keras.applications.resnet50.ResNet50(include_top=True, weights=None, input_shape=X_shape[1:], classes=class_count)
    model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights=None, input_shape=X_shape[1:], classes=class_count) # this works best
    opt = keras.optimizers.Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def get_keras_scikitlearn_model(X_train_shape, n_classes):
    global X_shape, class_count
    X_shape = X_train_shape
    class_count = n_classes
    model = KerasClassifier(make_resnet_model, batch_size=64, epochs=50)
    return model


def get_keras_features_labels(X_train,X_cv, X_test, y_train, y_cv, y_test, n_classes):
    X_train = X_train.astype('float32')
    X_cv = X_cv.astype('float32')
    X_test = X_test.astype('float32')
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_cv = keras.utils.to_categorical(y_cv, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    # zero mean unit variance
    X_train_extracted = (X_train / 255) - 0.5
    X_cv_extracted = (X_cv / 255) - 0.5
    X_test_extracted = (X_test / 255) - 0.5

    return X_train_extracted, X_cv_extracted, X_test_extracted, y_train, y_cv, y_test


# make random pixels black to avoid overreliance on a certain pixel(used as a defence algorithm)
def dropout_images(original_images):
    if not input_dropout:
        return original_images

    images = deepcopy(original_images)
    D1 = np.random.rand(images.shape[0], images.shape[1], images.shape[2])
    D1 = D1 < 0.5  # (using keep_prob as the threshold)
    images[:, :, :, 0] = np.multiply(images[:, :, :, 0], D1)
    images[:, :, :, 1] = np.multiply(images[:, :, :, 1], D1)
    images[:, :, :, 2] = np.multiply(images[:, :, :, 2], D1)

    return images
