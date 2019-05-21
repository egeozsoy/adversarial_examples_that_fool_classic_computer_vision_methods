# Hyperparams
vocab_size = 2500
data_size = 20000
batch_size = None # has no impact
input_dropout = False # has no impact
dataset_name = 'inria'
feature_extractor_name = 'hog_extractor'
attack_name = 'BoundaryPlusPlus'
visualize_sift = False
visualize_hog = False
image_size = 32 * 5

use_classes = [0,1]
n_features = 0

gaussion_components = 256  # has no impact

model_name = 'svc'
force_model_reload = False