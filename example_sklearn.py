import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import foolbox
from foolbox.models import Model
from loader import load_data
from utils import plot_result


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
        return self.predictor.predict(features)


def show_cifar_10_image(image):
    cv2.imshow('image', cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def show_imagenette_image(image):
    cv2.imshow('image', image)
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


from skimage.feature import hog
from skimage import exposure


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
    vocab_size = 1000
    data_size = 10000
    feature_extractor = hog_extractor
    visualize_sift = False
    visualize_hog = False
    image_size = 32 * 5
    use_classes = [0, 1]
    n_features = 0  # means sift chooses
    dataset_name = 'inria'
    vocs_folder = 'vocs'
    model_name = 'svc'

    if not os.path.exists(vocs_folder):
        os.mkdir(vocs_folder)

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
            bow_train = cv2.BOWKMeansTrainer(vocab_size)  # toy world, you want more.
            # 2. Fill bow
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

    if model_name == 'cnn':
        from keras_train_helpers import get_keras_scikitlearn_model, get_keras_features_labels

        model = get_keras_scikitlearn_model(X_train.shape, len(use_classes))

        X_train_extracted, X_test_extracted, y_train, y_test = get_keras_features_labels(X_train, X_test, y_train, y_test, len(use_classes))
        print('Starting training {}'.format(model_name))
        model.fit(X_train_extracted, y_train, validation_data=(X_test_extracted, y_test))

    else:
        print('Starting feature extraction')
        if model_name == 'svc':
            model = LinearSVC()
        elif model_name == 'forest':
            model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        elif model_name == 'logreg':
            model = LogisticRegression()
        else:
            model = LinearSVC()

        X_train_extracted = feature_extractor(X_train)
        X_test_extracted = feature_extractor(X_test)

        print('Starting training {}'.format(model_name))
        model.fit(X_train_extracted, y_train)

    print('Training set performance {}'.format(model.score(X_train_extracted, y_train)))
    print('Testing set performance {}'.format(model.score(X_test_extracted, y_test)))

    print('Sample predictions from training {}'.format(model.predict(X_train_extracted[:10])))
    print('Ground truth for        training {}'.format(y_train[:10]))
    print('Sample predictions from testing {}'.format(model.predict(X_test_extracted[:10])))
    print('Ground truth for        testing {}'.format(y_test[:10]))

    # adversarial = None
    # # stop if unsuccessful after #timeout trials
    # timeout = 5
    # while adversarial is None and timeout >= 0:
    #     fmodel = FoolboxSklearnWrapper(bounds=(0, 255), channel_axis=2, feature_extractor=feature_extractor, predictor=model)
    #     attack = foolbox.attacks.BoundaryAttack(model=fmodel)
    #     # multiply the image with 255, to reverse the normalization before the activations
    #     adversarial = attack(test_image, 9, verbose=True, iterations=10)
    #     timeout -= 1
    #
    # print('Original image predicted as {}'.format(np.argmax(model.predict(test_image.reshape(-1, 32 * 32 * 3)))))
    # print('Adversarial image predicted as {}'.format(np.argmax(model.predict(adversarial.reshape(-1, 32 * 32 * 3)))))
    # if adversarial is not None:
    #     plot_result([test_image / 255], [adversarial / 255])
