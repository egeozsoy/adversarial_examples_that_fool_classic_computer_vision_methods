# just return flattened image
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure

from configurations import n_features, vocab_size

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


# http://www.vlfeat.org/api/fisher-fundamentals.html
def fishervector_extractor(images):
    training_features = np.zeros((images.shape[0],n_features,128)) # 128 because of sift

    for idx,image in enumerate(images):
        _, desc = extract_sift_features(np.uint8(image))

        if desc is not None and desc.shape[0] >= n_features: #else leave descriptors as np.zeros
            desc = desc[:n_features]
            training_features[idx] = desc

    fishervectors = fishervector_gmm.predict(training_features)  # (n_images, 2*n_kernels, n_feature_dim)
    # TODO double check if flattening makes sense
    # flatten
    fishervectors = fishervectors.reshape((fishervectors.shape[0], -1))  # (n_images, 2*n_kernels * n_feature_dim)

    return fishervectors


def get_feature_extractor(extractor_name):
    if extractor_name == 'bovw_extractor':
        return bovw_extractor
    elif extractor_name == 'hog_extractor':
        return hog_extractor
    elif extractor_name == 'fishervector_extractor':
        return fishervector_extractor
    else:
        print('WARNING RETURNING DUMMY EXTRACTOR')
        return dummy_feature_extractor
