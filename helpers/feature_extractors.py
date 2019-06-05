# just return flattened image
import os

import cv2
from cv2.cv2 import BOWKMeansTrainer
import numpy as np
from numpy.core._multiarray_umath import ndarray
from skimage.feature import hog
from skimage import exposure
from sklearn.externals import joblib
from fishervector import FisherVectorGMM

from configurations import n_features, vocab_size, gaussion_components, batch_size

# initilize features to use
sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features)
matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
bow_extract = cv2.BOWImgDescriptorExtractor(sift, matcher)
fishervector_gmm = None


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
    bovws = [bovw_extractor_helper(image) for image in np.uint8(images)]
    bovws = np.array(bovws)
    bovws = bovws.reshape(bovws.shape[0], bovws.shape[2])

    return bovws


def prepare_bovw_vocabulary(X_train: np.array, vocs_folder: str, iter_name: str):
    if not os.path.exists(os.path.join(vocs_folder, 'voc_{}.npy'.format(iter_name))):
        bow_train: BOWKMeansTrainer = cv2.BOWKMeansTrainer(vocab_size)
        # Fill bow with sift calculations
        print('Calculating Vocabulary for training images')
        images_with_problems: int = 0
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


def visualize_sift_points(image):
    kp, desc = extract_sift_features(image)
    gray = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2GRAY)
    gray = np.uint8(image)
    img = cv2.drawKeypoints(gray, kp, None)
    cv2.imshow('sift_keypoints.jpg', img)
    cv2.waitKey(0)


def hog_visualizer(image):
    _, hog_img = hog(image, visualize=True, pixels_per_cell=(8, 8))
    hog_img = exposure.rescale_intensity(hog_img, in_range=(0, 10)) * 255
    cv2.imshow('hog_features', np.uint8(hog_img))
    cv2.waitKey(0)


def hog_extractor(images):
    hogs = [hog(image) for image in images]
    hogs = np.array(hogs)
    return hogs


def initilize_fishervector_gmm(fv_gmm):
    global fishervector_gmm
    fishervector_gmm = fv_gmm


def chunks(l, n: int):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# http://www.vlfeat.org/api/fisher-fundamentals.html
def fishervector_extractor(images: ndarray) -> ndarray:
    training_features: ndarray = np.zeros((images.shape[0], n_features, 128), dtype=np.float32)  # 128 because of sift

    for idx, image in enumerate(images):
        _, desc = extract_sift_features(np.uint8(image))

        if desc is not None and desc.shape[0] >= n_features:  # else leave descriptors as np.zeros
            desc = desc[:n_features]
            training_features[idx] = desc

    fishervectors: ndarray = np.empty((images.shape[0], 2 * gaussion_components, 128))  # (n_images, 2*n_kernels, n_feature_dim)
    current_idx: int = 0
    for batch in chunks(training_features, batch_size):  # doing this in batches as doing it in one shot creates big memory problems
        current_size = batch.shape[0]
        fishervectors[current_idx:current_idx + current_size] = fishervector_gmm.predict(batch)  # type: ignore
        current_idx += current_size

    # flatten
    fishervectors = fishervectors.reshape((fishervectors.shape[0], -1))  # (n_images, 2*n_kernels * n_feature_dim)

    return fishervectors


def prepare_fishervector_gmm(X_train, features_folder, iter_name):
    if not os.path.exists(os.path.join(features_folder, 'fisherkernel_{}'.format(iter_name))):
        # Fill bow with sift calculations
        print('Calculating Sift Features for training images')
        images_with_problems = 0
        training_features = None
        for idx, train_image in enumerate(X_train):

            kp, desc = extract_sift_features(train_image)

            # sometimes, sift extracts less than it is suppose to
            if desc is None or desc.shape[0] < n_features:
                images_with_problems += 1
            else:
                desc = desc[:n_features]
                desc = np.expand_dims(desc, axis=0).astype(np.float32)  # increase this if more accuracy is required
                if training_features is None:
                    training_features = desc
                else:
                    training_features = np.concatenate([training_features, desc], axis=0)

        # Generally errors occur if less than the requested amount of sift features were found per image,
        # in training time, we ignore these images, in test time, we use zero vector to represent these images.
        # Obviously makes it impossible to predict that image, so we want to choose a n_features count for sift that minimizes this error, while still giving good results.
        print('Errors occurred for {} images'.format(images_with_problems))

        print('Calculating Fisher Kernel for training images')
        fishervector_gmm: FisherVectorGMM = FisherVectorGMM(n_kernels=gaussion_components).fit(training_features)

        joblib.dump(fishervector_gmm, os.path.join(features_folder, 'fisherkernel_{}'.format(iter_name)))

    else:
        print('Loaded Fisher Kernel')
        fishervector_gmm = joblib.load(os.path.join(features_folder, 'fisherkernel_{}'.format(iter_name)))
    initilize_fishervector_gmm(fishervector_gmm)


def get_feature_extractor(extractor_name: str):
    if extractor_name == 'bovw_extractor':
        return bovw_extractor
    elif extractor_name == 'hog_extractor':
        return hog_extractor
    elif extractor_name == 'fishervector_extractor':
        return fishervector_extractor
    else:
        print('WARNING RETURNING DUMMY EXTRACTOR')
        return dummy_feature_extractor
