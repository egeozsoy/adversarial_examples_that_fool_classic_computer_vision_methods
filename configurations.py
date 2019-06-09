from typing import List
import os

# Sample configuration names can be found in trained_configurations.txt
# should correspond to a configuration in custom_configurations folder
with open('active_configuration.txt','r') as f:
    configuration_name = f.read().strip()

# Hyperparams, they(most of them) will be overwritten by a corresponding configuration, these values are just a guide line, or example
vocab_size: int = 0
data_size: int = 20000
batch_size: int = 11000
dataset_name: str = 'imagenette'
input_dropout: bool = False
feature_extractor_name: str = ''
visualize_sift: bool = False
visualize_hog: bool = False
image_size: int = 32 * 5
use_classes: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
n_features: int = 0  # has no impact in this case
gaussion_components: int = 0  # has no impcact in this case
model_name: str = 'cnn'
force_model_reload: bool = False

save_correct_predictions:bool = False
matplotlib_backend = 'Agg'

configuration_name,target_mode = configuration_name.rsplit('-') # seperate targeted or untargeted from configuration_name
targeted_attack:bool = target_mode == 'targeted' # set targeted_attack parameter based on the information from the model name

# This overwrites the above defined hyperparameters with hyperparameters from custom_configurations
exec('from custom_configurations.{} import *'.format(configuration_name))

if dataset_name == 'inria':
    targeted_attack = False # it makes no sense to use targeted in binary classification

no_feature_reload:bool = True
just_train:bool = False
adversarial_test_size:int = 100

correct_predictions_folder: str = 'correct_predictions'
correct_predictions_file: str = os.path.join(correct_predictions_folder, '{}_correct_predictions.npy'.format(dataset_name))

# This overwrites the above defined hyperparameters with hyperparameters from custom_configurations
exec('from custom_configurations.{} import *'.format(configuration_name))
