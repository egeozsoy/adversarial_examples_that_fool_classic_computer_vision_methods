from typing import Optional

import numpy as np

from foolbox.models import Model
from numpy.core._multiarray_umath import ndarray

from configurations import use_classes, model_name
from helpers.image_utils import show_image


class FoolboxSklearnWrapper(Model):

    def __init__(self, bounds, channel_axis, preprocessing=(0, 1), feature_extractor=None, predictor=None, num_classes=10):
        super().__init__(bounds, channel_axis, preprocessing)
        self.predictor = predictor
        self.feature_extractor = feature_extractor
        self.number_classes = num_classes
        self.queries = 0

    def num_classes(self):
        return self.number_classes

    def batch_predictions(self, images: np.ndarray):
        self.queries += images.shape[0] # we want to count amount of queries
        if model_name != 'cnn':
            features = self.feature_extractor(images)
            predictions = self.predictor.predict(features)
        else:
            from helpers.keras_train import dropout_images
            predictions = self.predictor.predict(dropout_images((images / 255)-0.5))
        # convert the prediction to one hot
        one_hot_pred: ndarray = np.zeros((images.shape[0], len(use_classes)))
        for idx, prediction in enumerate(predictions):
            prediction = int(prediction)
            one_hot_pred[idx][prediction] = 1

        return one_hot_pred

def find_closest_reference_image(attacked_img: ndarray, reference_images: ndarray, reference_labels: ndarray, reference_predictions: ndarray, original_label: int,
                                 target_label: int = None):
    # finds the closes reference img to images, that still belongs to other class

    # Make sure we leave only images which belong to our target class and are getting correctly predicted as such
    if target_label is not None:
        #targeted setting

        mask_wanted_class: Optional[bool] = reference_predictions == target_label
        mask_correctly_predicted = reference_predictions == reference_labels
        mask = np.logical_and(mask_wanted_class,mask_correctly_predicted)

    else:
        #untargeted setting
        mask_wanted_class = reference_predictions != original_label
        mask_correctly_predicted = reference_predictions == reference_labels
        mask = np.logical_and(mask_wanted_class, mask_correctly_predicted)

    ref_images_masked = reference_images[mask]
    difference_metric = np.linalg.norm(np.reshape(ref_images_masked - attacked_img, (ref_images_masked.shape[0], -1)), axis=1)
    most_similar_image:np.ndarray = ref_images_masked[np.argmin(difference_metric)]

    # most_different_image = ref_images_masked[np.argmax(difference_metric)]
    # show_image(goal_img,'goal_img')
    # show_image(most_different_image,'most_different')
    # show_image(most_similar_image,'most_similar')

    return most_similar_image
