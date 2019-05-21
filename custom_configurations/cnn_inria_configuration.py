from typing import List

# Hyperparams
vocab_size: int = 0  # has no impact
data_size: int = 20000
batch_size: int = 11000
dataset_name: str = 'inria'
input_dropout: bool = False
feature_extractor_name: str = ''
attack_name: str = 'BoundaryPlusPlus'
visualize_sift: bool = False
visualize_hog: bool = False
image_size: int = 32 * 5

use_classes: List[int] = [0, 1]

n_features: int = 0  # has no impact in this case

gaussion_components: int = 0  # has no impcact in this case

model_name: str = 'cnn'
force_model_reload: bool = False
