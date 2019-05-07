import keras
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.wrappers.scikit_learn import KerasClassifier

X_shape = None
class_count = None


def make_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=X_shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.7))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.7))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(class_count))
    model.add(Activation('softmax'))

    # initiate optimizer
    opt = keras.optimizers.Adam(lr=0.0001)

    # Let's train the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def get_keras_scikitlearn_model(X_train_shape, n_classes):
    global X_shape, class_count
    X_shape = X_train_shape
    class_count = n_classes
    model = KerasClassifier(make_model, batch_size=64, epochs=40)
    return model


def get_keras_features_labels(X_train, X_test, y_train, y_test, n_classes):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
    X_train_extracted = X_train / 255
    X_test_extracted = X_test / 255

    return X_train_extracted, X_test_extracted, y_train, y_test
