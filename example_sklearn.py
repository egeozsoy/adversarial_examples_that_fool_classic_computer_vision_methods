import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from skimage.feature import hog
from skimage import exposure
import foolbox
from foolbox.models import Model
from loader import load_data
from utils import plot_result
from copy import deepcopy


class FoolboxSklearnWrapper(Model):

    def __init__(self, bounds, channel_axis, preprocessing=(0, 1), feature_extractor=None, predictor=None, num_classes=10):
        super().__init__(bounds, channel_axis, preprocessing)
        self.predictor = predictor
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes

    def num_classes(self):
        return self.num_classes

    def batch_predictions(self, images):
        features = self.feature_extractor(images)
        predictions = int(self.predictor.predict(features))
        # convert the prediction to one hot
        one_hot_pred = np.zeros(len(use_classes))
        one_hot_pred[predictions] = 1

        return np.array([one_hot_pred])


def show_cifar_10_image(image, label='image'):
    cv2.imshow(label, cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def show_imagenette_image(image, label='image'):
    cv2.imshow(label, image)
    cv2.waitKey(0)


def show_mnist_image(image):
    cv2.imshow('image', np.uint8(image))
    cv2.waitKey(0)


def visualize_sift_points(image):
    kp, desc = extract_sift_features(image)
    gray = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2GRAY)
    gray = np.uint8(image)
    img = cv2.drawKeypoints(gray, kp, None)
    cv2.imshow('sift_keypoints.jpg', img)
    cv2.waitKey(0)


# just return flattened image
def dummy_feature_extractor(images):
    return images.reshape(images.shape[0], -1) / 255


# output sift descriptors
def extract_sift_features(image: np.uint8):
    # some images are already black and white, only convert if not black and white
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    kp, desc = sift.detectAndCompute(grayscale, None)
    return kp, desc


def bovw_extractor_helper(image: np.uint8):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    siftkp = sift.detect(grayscale)
    bov = bow_extract.compute(grayscale, siftkp)
    if bov is None:
        # set bov to zero if no sift features were found
        bov = np.zeros((1, vocab_size))

    return bov


def bovw_extractor(images):
    bovws = [bovw_extractor_helper(image) for image in images]
    bovws = np.array(bovws)
    bovws = bovws.reshape(bovws.shape[0], bovws.shape[2])

    return bovws


# delete classes that are not in keep_classes, simplify cifar-10
def filter_classes(X, y, keep_classes):
    logic_result = y == keep_classes[0]
    for keep_class in keep_classes[1:]:
        logic_result = np.logical_or(logic_result, y == keep_class)

    X = X[logic_result]
    y = y[logic_result]

    return X, y


def hog_visualizer(image):
    _, hog_img = hog(image, visualize=True, pixels_per_cell=(8, 8))
    hog_img = exposure.rescale_intensity(hog_img, in_range=(0, 10)) * 255
    cv2.imshow('hog_features', np.uint8(hog_img))
    cv2.waitKey(0)


def hog_extractor(images):
    hogs = [hog(image) for image in images]
    hogs = np.array(hogs)
    return hogs


if __name__ == '__main__':
    # Hyperparams
    vocab_size = 2500
    data_size = 20000
    feature_extractor = hog_extractor
    visualize_sift = False
    visualize_hog = False
    image_size = 32 * 5
    use_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    n_features = 0  # means sift chooses
    dataset_name = 'inria'
    vocs_folder = 'vocs'
    models_folder = 'models'
    features_folder = 'features'
    model_name = 'svc'

    if not os.path.exists(vocs_folder):
        os.mkdir(vocs_folder)

    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    if not os.path.exists(features_folder):
        os.mkdir(features_folder)

    iter_name = 'iter_dtn_{}_vs{}_ds{}_is{}_cc{}_nf{}_fe_{}'.format(dataset_name, vocab_size, data_size, image_size, len(use_classes), n_features,
                                                                    feature_extractor.__name__)

    print('Running iteration {}'.format(iter_name))

    # initilize features to use
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features)

    print('Loading Data')
    X, y = load_data(image_size, dataset_name=dataset_name)

    X, y = filter_classes(X, y, keep_classes=use_classes)

    # we might want to resize images
    if image_size != X.shape[1]:
        print('Resizing images')
        X_resized = []
        for img in X:
            X_resized.append(cv2.resize(img, (image_size, image_size)))

        X = np.array(X_resized)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

    X_train = X_train[:data_size]
    X_test = X_test[:data_size]
    y_train = y_train[:data_size]
    y_test = y_test[:data_size]

    print('Dataset size {}'.format(X_train.shape[0]))

    if visualize_sift:
        print('Visualising images with sift points')
        for a in range(0, 100):
            print(y_train[a:a + 1])
            show_imagenette_image(X_train[a])
            visualize_sift_points(X_train[a])

    if visualize_hog:
        for a in range(0, 100):
            print(y_train[a:a + 1])
            show_imagenette_image(X_train[a])
            hog_visualizer(X_train[a])

    if feature_extractor == bovw_extractor and model_name != 'cnn':

        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
        ## 1. setup BOW
        bow_extract = cv2.BOWImgDescriptorExtractor(sift, matcher)

        if not os.path.exists(os.path.join(vocs_folder, 'voc_{}.npy'.format(iter_name))):
            bow_train = cv2.BOWKMeansTrainer(vocab_size)
            # Fill bow
            print('Calculating Vocabulary for training images')
            images_with_problems = 0
            for idx, train_image in enumerate(X_train):

                kp, desc = extract_sift_features(train_image)
                if desc is None:
                    images_with_problems += 1
                else:
                    bow_train.add(desc)

            print('No sift features were found for {} images'.format(images_with_problems))
            print('Generating vocabulary by clustering')

            voc = bow_train.cluster()
            np.save(os.path.join(vocs_folder, 'voc_{}.npy'.format(iter_name)), voc)

        else:
            print('Loaded Vocabulary')
            voc = np.load(os.path.join(vocs_folder, 'voc_{}.npy'.format(iter_name)))

        bow_extract.setVocabulary(voc)

    full_model_path = os.path.join(models_folder, '{}_{}'.format(model_name, iter_name))
    full_features_path = os.path.join(features_folder, '{}_{}'.format(model_name, iter_name))

    if model_name == 'cnn':
        from keras_train_helpers import get_keras_scikitlearn_model, get_keras_features_labels

        model = get_keras_scikitlearn_model(X_train.shape, len(use_classes))

        if os.path.exists(full_features_path):
            print('Loading Features Labels')
            features_labels = joblib.load(full_features_path)
        else:
            print('Generating and saving Features Labels')
            features_labels = get_keras_features_labels(X_train, X_test, y_train, y_test, len(use_classes))
            joblib.dump(features_labels,full_features_path)

        X_train_extracted, X_test_extracted, y_train, y_test = features_labels

        if os.path.exists(full_model_path):
            print('Loading Model')
            model = joblib.load(full_model_path)

        else:
            print('Starting Model training {} and saving model'.format(model_name))
            model.fit(X_train_extracted, y_train, validation_data=(X_test_extracted, y_test))
            joblib.dump(model, full_model_path)

    else:
        if model_name == 'svc':
            model = LinearSVC()
        elif model_name == 'bagged_svc':
            base_model = LinearSVC()
            # this train many svms like random forest does with decision trees
            model = BaggingClassifier(base_model, n_jobs=-1,
                                      n_estimators=100)  # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
        elif model_name == 'forest':
            model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        elif model_name == 'logreg':
            model = LogisticRegression()
        else:
            model = LinearSVC()

        if os.path.exists(full_features_path):
            print('Loading Features Labels')
            features_labels = joblib.load(full_features_path)

        else:
            print('Generating and saving Features Labels')
            features_labels = (feature_extractor(X_train), feature_extractor(X_test))
            joblib.dump(features_labels, full_features_path)


        X_train_extracted, X_test_extracted = features_labels

        if os.path.exists(full_model_path):
            model = joblib.load(full_model_path)

        else:

            print('Starting Model training {} and saving model'.format(model_name))
            model.fit(X_train_extracted, y_train)
            joblib.dump(model, full_model_path)

    print('Training set performance {}'.format(model.score(X_train_extracted, y_train)))
    print('Testing set performance {}'.format(model.score(X_test_extracted, y_test)))

    print('Sample predictions from training {}'.format(model.predict(X_train_extracted[:20])))
    print('Ground truth for        training {}'.format(y_train[:20]))
    print('Sample predictions from testing {}'.format(model.predict(X_test_extracted[:20])))
    print('Ground truth for        testing {}'.format(y_test[:20]))

    # visualize some predictions
    for i in range(0):
        label = str(model.predict(X_test_extracted[i:i + 1])[0])
        show_imagenette_image(X_test[i], '{}-{}'.format(y_test[i], label))


    #starting adversarial
    test_image = np.float32(X_test[1])
    label = y_test[1]

    adversarial = None
    # stop if unsuccessful after #timeout trials
    timeout = 5
    while adversarial is None and timeout >= 0:
        fmodel = FoolboxSklearnWrapper(bounds=(0, 255), channel_axis=2, feature_extractor= feature_extractor, predictor=model)
        attack = foolbox.attacks.BoundaryAttack(model=fmodel)
        # multiply the image with 255, to reverse the normalization before the activations
        adversarial = attack(deepcopy(test_image), label, verbose=True, iterations=100)
        timeout -= 1

    print('Original image predicted as {}'.format(label))
    adv_label = model.predict(feature_extractor(np.array([adversarial])))[0]
    print('Adverserial image predicted as {}'.format(adv_label))
    if adversarial is not None:
        show_imagenette_image(np.uint8(test_image),str(label))
        show_imagenette_image(np.uint8(adversarial),str(adv_label))
