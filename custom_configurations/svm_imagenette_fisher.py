# Hyperparams
vocab_size = 2500
data_size = 20000
batch_size = 8000
input_dropout = False # has no impact
dataset_name = 'imagenette'
feature_extractor_name = 'fishervector_extractor'
attack_name = 'BoundaryPlusPlus'
visualize_sift = False
visualize_hog = False
image_size = 32 * 5

use_classes = [0,1,2,3,4,5,6,7,8,9]
n_features = 25 # for fishervector hardcode it to a certain value, for bovw, we can set it to 0 and let sift decide

gaussion_components = 256

model_name = 'sgd_svc'
force_model_reload = False