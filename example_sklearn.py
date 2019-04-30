from foolbox.models import Model
import foolbox
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from load_cifar10_data import get_data
import numpy as np
from utils import plot_result
import cv2

#TODO visualize sift function
#TODO mnist and fashion mnist
# TODO easier cifar 10 with only 2 classes

def one_hot_to_class(one_hot):
    return np.argmax(one_hot,axis=1)

def show_cifar_10_image(image):
    cv2.imshow('image',cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

# just return flattened image
def dummy_feature_extractor(images):
    return images.reshape(-1, 32 * 32 * 3)

# output sift descriptors
def extract_sift_features(image):
    grayscale = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2GRAY)
    kp, desc = sift.detectAndCompute(grayscale,None)
    return kp, desc

def bovw_extractor(images):
    bovws = None
    for image in images:
        grayscale = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2GRAY)
        siftkp = sift.detect(grayscale)
        bov = bow_extract.compute(grayscale,siftkp)
        if bov is None:
            # set bov to zero if no sift features were found
            bov = np.zeros((1,vocab_size))
        if bovws is None:
            bovws = bov
        else:
            bovws = np.concatenate([bovws,bov])

    return bovws

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


if __name__ == '__main__':
    #Hyperparams
    vocab_size = 2000
    data_size = 60000
    feature_extractor = dummy_feature_extractor


    # initilize features to use
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.02,edgeThreshold = 25)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
    ## 1. setup BOW
    bow_train = cv2.BOWKMeansTrainer(vocab_size)  # toy world, you want more.
    bow_extract = cv2.BOWImgDescriptorExtractor(sift, matcher)

    data = np.float32(get_data()[b'data'])[0:data_size]
    labels = np.float32(get_data()[b'labels'])[0:data_size]

    X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=0.2,shuffle=True)

    # 2. Fill bow
    images_with_problems = 0
    for idx,train_image in enumerate(X_train):

        kp,desc = extract_sift_features(train_image)
        if desc is None:
            images_with_problems += 1
        else:
            bow_train.add(desc)

    print('No sift features were found for {} images'.format(images_with_problems))
    voc = bow_train.cluster()
    bow_extract.setVocabulary(voc)

    model = RandomForestClassifier(n_estimators=100)
    model = LinearSVC(max_iter=20)

    y_train = one_hot_to_class(y_train)
    y_test = one_hot_to_class(y_test)

    print('Starting model training')
    model.fit(feature_extractor(X_train/255), y_train)
    print('Training set performance {}'.format(model.score(feature_extractor(X_train/255), y_train)))
    print('Testing set performance {}'.format(model.score(feature_extractor(X_test/255), y_test)))

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
