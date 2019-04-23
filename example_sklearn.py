from foolbox.models import Model
import foolbox
from sklearn.ensemble import RandomForestClassifier
from load_cifar10_data import get_data
import numpy as np
from utils import plot_result


# just return flattened image
def dummy_feature_extractor(images):
    return images.reshape(-1, 32 * 32 * 3)


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
    model = RandomForestClassifier(n_estimators=10, max_depth=20)
    data = np.float32(get_data()[b'data'])
    train_images = data[0:1000]
    train_labels = np.float32(get_data()[b'labels'][0:1000])
    test_images = data[1000:2000]
    test_labels = np.float32(get_data()[b'labels'][1000:2000])
    test_image = data[1]
    model.fit(train_images.reshape(-1, 32 * 32 * 3), train_labels)
    print('Training set performance {}'.format(model.score(train_images.reshape(-1, 32 * 32 * 3), train_labels)))
    print('Testing set performance {}'.format(model.score(test_images.reshape(-1, 32 * 32 * 3), test_labels)))

    adversarial = None
    # stop if unsuccessful after #timeout trials
    timeout = 5
    while adversarial is None and timeout >= 0:
        fmodel = FoolboxSklearnWrapper(bounds=(0, 255), channel_axis=2, feature_extractor=dummy_feature_extractor, predictor=model)
        attack = foolbox.attacks.BoundaryAttack(model=fmodel)
        # multiply the image with 255, to reverse the normalization before the activations
        adversarial = attack(test_image, 9, verbose=True, iterations=10)
        timeout -= 1

    print('Original image predicted as {}'.format(np.argmax(model.predict(test_image.reshape(-1, 32 * 32 * 3)))))
    print('Adversarial image predicted as {}'.format(np.argmax(model.predict(adversarial.reshape(-1, 32 * 32 * 3)))))
    plot_result([test_image / 255], [adversarial / 255])
