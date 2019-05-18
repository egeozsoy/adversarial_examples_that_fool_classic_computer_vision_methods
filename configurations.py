# Hyperparams
vocab_size = 2500
data_size = 20000
batch_size = None
dataset_name = 'imagenette'
feature_extractor_name = 'fishervector_extractor'
attack_name = 'BoundaryPlusPlus'
visualize_sift = False
visualize_hog = False
image_size = 32 * 5

if dataset_name == 'inria':
    use_classes = [0,1]
else:
    use_classes = [0,1,2,3,4,5,6,7,8,9]

# for fishervector hardcode it to a certain value, for bovw, we can set it to 0 and let sift decide
if feature_extractor_name == 'fishervector_extractor':
    n_features = 25
else:
    n_features = 0

gaussion_components = 256  # the paper uses 256 kernel, and values like 128 seem to be realistic. Fisher Kernels on Visual Vocabularies for Image Categorization

model_name = 'sgd_svc'

if model_name == 'sgd_svc':
    batch_size = 8000