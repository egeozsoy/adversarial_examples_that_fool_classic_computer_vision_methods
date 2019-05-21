from typing import List

# should correspond to a configuration in custom_configurations
configuration_name = 'svm_inria_bovw' #TODO continue with svm_inria_fisher

# Hyperparams, they(most of them) will be overwritten by a corresponding configuration, these values are just a guide line, or example
vocab_size: int = 0
data_size: int = 20000
batch_size: int = 11000
dataset_name: str = 'imagenette'
input_dropout: bool = False
feature_extractor_name: str = ''
attack_name: str = 'BoundaryPlusPlus'
visualize_sift: bool = False
visualize_hog: bool = False
image_size: int = 32 * 5

use_classes: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

n_features: int = 0  # has no impact in this case

gaussion_components: int = 0  # has no impcact in this case

model_name: str = 'cnn'
force_model_reload: bool = False
exec('from custom_configurations.{} import *'.format(configuration_name))
