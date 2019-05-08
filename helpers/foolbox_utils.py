import numpy as np

from foolbox.models import Model

from configurations import use_classes, model_name


class FoolboxSklearnWrapper(Model):

    def __init__(self, bounds, channel_axis, preprocessing=(0, 1), feature_extractor=None, predictor=None, num_classes=10):
        super().__init__(bounds, channel_axis, preprocessing)
        self.predictor = predictor
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes

    def num_classes(self):
        return self.num_classes

    def batch_predictions(self, images: np.ndarray):
        if model_name != 'cnn':
            features = self.feature_extractor(images)
            predictions = self.predictor.predict(features)
        else:
            from helpers.keras_train import dropout_images
            predictions = self.predictor.predict(dropout_images(images))
        # convert the prediction to one hot
        one_hot_pred = np.zeros((images.shape[0], len(use_classes)))
        for idx, prediction in enumerate(predictions):
            prediction = int(prediction)
            one_hot_pred[idx][prediction] = 1

        return one_hot_pred
