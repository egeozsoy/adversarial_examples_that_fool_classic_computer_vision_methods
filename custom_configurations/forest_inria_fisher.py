# Hyperparams
vocab_size = 2500 # has no impact
data_size = 20000
batch_size = 1000
input_dropout = False # has no impact
dataset_name = 'inria'
feature_extractor_name = 'fishervector_extractor'
visualize_sift = False
visualize_hog = False
image_size = 32 * 5

use_classes = [0,1]
n_features = 75 # for fishervector hardcode it to a certain value, for bovw, we can set it to 0 and let sift decide

gaussion_components = 256

model_name = 'forest'
force_model_reload = False