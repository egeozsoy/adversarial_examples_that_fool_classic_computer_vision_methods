# Hyperparams
vocab_size = 2500
data_size = 20000
dataset_name = 'imagenette'
feature_extractor_name = 'fishervector_extractor'
attack_name = 'BoundaryPlusPlus'
visualize_sift = False
visualize_hog = False
image_size = 32 * 5
use_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
n_features = 50  # for fishervector hardcode it to a certain value, for bovw, we can set it to 0 and let sift decide
gaussion_components = 64  # the paper uses 256 kernel, and values like 128 seem to be realistic. Fisher Kernels on Visual Vocabularies for Image Categorization
model_name = 'svc'
