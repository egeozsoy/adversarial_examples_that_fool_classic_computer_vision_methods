import numpy as np
from foolbox.models import Model

from configurations import use_classes


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
