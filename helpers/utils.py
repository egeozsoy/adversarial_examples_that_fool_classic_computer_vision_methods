from typing import List, Tuple, Dict
import random
import os

import numpy as np
from sklearn.utils import shuffle

from configurations import save_correct_predictions, correct_predictions_file, use_classes, adversarial_test_size, max_queries

# Not used any more
deprecated_colors = {'imagenette_forest_fishervector_extractor_targeted': '#200000',
              'imagenette_forest_hog_extractor_targeted': '#600000',
              'imagenette_logreg_bovw_extractor_targeted': '#800000',
              'imagenette_logreg_hog_extractor_untargeted': '#a00000',
              'imagenette_logreg_fishervector_extractor_untargeted': '#c00000',
              'inria_logreg_bovw_extractor_untargeted': '#e00000',
              'inria_svc_hog_extractor_untargeted': '#201400',
              'inria_cnn__untargeted': '#402900',
              'inria_logreg_fishervector_extractor_untargeted': '#e03d00',
              'imagenette_svc_hog_extractor_targeted': '#805100',
              'imagenette_forest_bovw_extractor_untargeted': '#a06600',
              'imagenette_svc_fishervector_extractor_untargeted': '#c07a00',
              'inria_forest_hog_extractor_untargeted': '#e08f01',
              'imagenette_svc_fishervector_extractor_targeted': '#206700',
              'imagenette_svc_bovw_extractor_targeted': '#309c00',
              'imagenette_forest_fishervector_extractor_untargeted': '#3dc601',
              'inria_forest_fishervector_extractor_untargeted': '#004032',
              'imagenette_logreg_hog_extractor_targeted': '#00604b',
              'inria_logreg_hog_extractor_untargeted': '#008265',
              'imagenette_logreg_fishervector_extractor_targeted': '#00a17d',
              'inria_svc_fishervector_extractor_untargeted': '#00cea0',
              'imagenette_logreg_bovw_extractor_untargeted': '#00e0ae',
              'imagenette_cnn__untargeted': '#002036',
              'imagenette_cnn__targeted': '#00406d',
              'imagenette_forest_bovw_extractor_targeted': '#0060a3',
              'imagenette_svc_bovw_extractor_untargeted': '#0080d9',
              'inria_svc_bovw_extractor_untargeted': '#0c0046',
              'inria_forest_bovw_extractor_untargeted': '#160083',
              'imagenette_svc_hog_extractor_untargeted': '#1e00b7',
              'imagenette_forest_hog_extractor_untargeted': '#d600b0'}

def gpu_available():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    for x in local_device_protos:
        if x.device_type == 'GPU':
            return True
    return False


# delete classes that are not in keep_classes, simplify cifar-10
def filter_classes(X: np.ndarray, y: np.ndarray, keep_classes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    logic_result = y == keep_classes[0]
    for keep_class in keep_classes[1:]:
        logic_result = np.logical_or(logic_result, y == keep_class)

    X = X[logic_result]
    y = y[logic_result]

    return X, y


# guarentee that a batch has almost equal class distribution
def get_balanced_batch(x: np.ndarray, y: np.ndarray, batch_size: int, classes: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = list(x.shape)
    shape[0] = batch_size
    batch_train_x = np.zeros(shape, dtype=x.dtype)
    batch_train_y = np.zeros(shape[0], dtype=y.dtype)
    shape[0] = len(classes) * 10
    batch_cv_x = np.zeros(shape, dtype=x.dtype)
    batch_cv_y = np.zeros(shape[0], dtype=y.dtype)

    class_size = batch_size // (len(classes))
    for c in classes:
        indices_mask = y == c
        x_c = x[indices_mask]
        y_c = y[indices_mask]

        random_indices = np.random.choice(x_c.shape[0] - 10, class_size, replace=False)  # the first 10 elements are reserverd for cv
        batch_train_x[c * class_size:c * class_size + class_size] = x_c[random_indices + 10, :]
        batch_train_y[c * class_size:c * class_size + class_size] = y_c[random_indices + 10]

        for i in range(10):
            batch_cv_x[c * 10 + i] = x_c[i]
            batch_cv_y[c * 10 + i] = y_c[i]

    batch_train_x, batch_train_y = shuffle(batch_train_x, batch_train_y)
    batch_cv_x, batch_cv_y = shuffle(batch_cv_x, batch_cv_y)

    return batch_train_x, batch_train_y, batch_cv_x, batch_cv_y


def equally_distributed_indices(correct_indices, y: np.ndarray):
    needed_sample_per_class = adversarial_test_size // len(use_classes)  # make sure it is divideble

    group_by_class: Dict[int, List[int]] = {}

    final_indices: List[int] = []

    for correct_idx in correct_indices:
        element = y[correct_idx]
        if element in group_by_class:
            group_by_class[element].append(correct_idx)
        else:
            group_by_class[element] = [correct_idx]

    for cls in group_by_class:
        cls_elemenets = list(group_by_class[cls])
        if len(cls_elemenets) <= needed_sample_per_class:
            final_indices += cls_elemenets
        else:
            final_indices += random.choices(cls_elemenets, k=needed_sample_per_class)

    return np.array(final_indices)


def get_adversarial_test_set(predictions_from_testing, y_test, all_predicted_correctly: bool = False):
    if all_predicted_correctly:
        # TODO this requires fixing if we want to use
        if save_correct_predictions:
            # Create a list of correct predictions, so we can make sure every model predicts our end test set correctly
            correct_predictions: np.bool = predictions_from_testing == y_test
            if os.path.exists(correct_predictions_file):
                all_correct_predictions = np.load(correct_predictions_file)
                all_correct_predictions = np.concatenate([all_correct_predictions, correct_predictions[None, :]])

            else:
                all_correct_predictions = correct_predictions[None, :]
            np.save(correct_predictions_file, all_correct_predictions)

        all_correct_predictions = np.load(correct_predictions_file)
        shared_correct_prediction_idx = np.all(all_correct_predictions, axis=0)

        return shared_correct_prediction_idx

    else:
        # just return random correct predictions equally distrubited from testset
        correct_predictions = predictions_from_testing == y_test
        correct_prediction_idx = np.where(correct_predictions)[0]

        return equally_distributed_indices(correct_prediction_idx, y_test)


def generate_graph_data(folder_name: str, max_queries: int):
    from matplotlib import pyplot as plt

    graphs = None
    for idx, f in enumerate(os.listdir(folder_name)):
        if '.csv' not in f:
            continue
        points = np.loadtxt(os.path.join(folder_name, f), delimiter=',')
        xp = points[:, 1]
        fp = points[:, 2]
        x = np.arange(1, max_queries + 1)
        graph = np.interp(x, xp, fp)
        graph = np.expand_dims(graph, axis=0)
        if graphs is None:
            graphs = graph
        else:
            graphs = np.concatenate([graphs, graph])

    if graphs is None:
        return
    median_graph = np.median(graphs, axis=0)
    mean_graph = np.mean(graphs,axis=0)
    np.save(os.path.join(folder_name, 'mean_graph.npy'), mean_graph)
    with open(os.path.join(folder_name,'mean_median.txt'),'w') as f:
        f.write('Median: {} ; Mean: {}'.format(median_graph[-1],mean_graph[-1]))
    plt.plot(median_graph)
    plt.ylabel("mean of thresholds")
    plt.xlabel("queries")
    plt.savefig(os.path.join(folder_name, 'graph'))
    plt.close()

    return median_graph


def create_folders(folder_names: List[str]):
    for folder_name in folder_names:
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

def filter_graphs(folder_names):
    filtered = []

    for folder_name in folder_names:
        if dataset_name not in folder_name:
            # Skip these
            continue
        if target_mode not in folder_name:
            # Skip these
            continue
        # if all, don't filter out based on model_type
        if 'ALL' not in model_types:
            model_type_found = False
            for model_type in model_types:
                if model_type in folder_name:
                    model_type_found = True
                    break

            # Skip if folder doesn't include model type
            if not model_type_found:
                continue

        # if all, don't filter out based on model_type
        if 'ALL' not in feature_extractor_types:
            feature_extractor_found = False
            for feature_extractor_type in feature_extractor_types:
                if feature_extractor_type in folder_name:
                    feature_extractor_found = True
                    break

            # Skip if folder doesn't include model type
            if not feature_extractor_found:
                continue

        # Add those which make it this far to the filtered list
        filtered.append(folder_name)

    return filtered

if __name__ == '__main__':
    #TODO CREATE TABLE WITH HOW MUCH BETTER IT IS COMPARED TO CNN
    from matplotlib import pyplot as plt
    import matplotlib.ticker as plticker

    graphs_to_export = [
        ('inria','_untargeted',['ALL'],['ALL']),
        ('inria', '_untargeted', ['ALL'], ['bovw','cnn']),
        ('inria', '_untargeted', ['ALL'], ['hog', 'cnn']),
        ('inria', '_untargeted', ['ALL'], ['fishervector', 'cnn']),
        ('inria', '_untargeted', ['logreg','cnn'], ['ALL']),
        ('inria', '_untargeted', ['forest', 'cnn'], ['ALL']),
        ('inria', '_untargeted', ['svc', 'cnn'], ['ALL']),
        ('imagenette', '_untargeted', ['ALL'], ['ALL']),
        ('imagenette', '_untargeted', ['ALL'], ['bovw', 'cnn']),
        ('imagenette', '_untargeted', ['ALL'], ['hog', 'cnn']),
        ('imagenette', '_untargeted', ['ALL'], ['fishervector', 'cnn']),
        ('imagenette', '_untargeted', ['logreg', 'cnn'], ['ALL']),
        ('imagenette', '_untargeted', ['forest', 'cnn'], ['ALL']),
        ('imagenette', '_untargeted', ['svc', 'cnn'], ['ALL']),
        ('imagenette', '_targeted', ['ALL'], ['ALL']),
        ('imagenette', '_targeted', ['ALL'], ['bovw', 'cnn']),
        ('imagenette', '_targeted', ['ALL'], ['hog', 'cnn']),
        ('imagenette', '_targeted', ['ALL'], ['fishervector', 'cnn']),
        ('imagenette', '_targeted', ['logreg', 'cnn'], ['ALL']),
        ('imagenette', '_targeted', ['forest', 'cnn'], ['ALL']),
        ('imagenette', '_targeted', ['svc', 'cnn'], ['ALL']),

    ]

    for graph_to_export in graphs_to_export:
        dataset_name,target_mode,model_types,feature_extractor_types = graph_to_export


        legends = []
        mean_graphs = []
        filtered_folders = filter_graphs(os.listdir('evaluations'))

        for folder in filtered_folders:
            folder_path = os.path.join('evaluations', folder)
            if not os.path.isdir(folder_path):
                continue

            legends.append(folder)
            mean_graphs.append(generate_graph_data(folder_path, max_queries))

        fig, ax = plt.subplots()
        for idx, graph in enumerate(mean_graphs):
            # Make sure cnn always has the same color
            if 'cnn' in legends[idx]:
                ax.plot(graph, 'r',linewidth=1.0)
            else:
                ax.plot(graph, linewidth=1.0)

        loc_major = plticker.MultipleLocator(base=0.005)  # locator puts ticks at regular intervals
        ax.yaxis.set_major_locator(loc_major)
        loc_minor = plticker.MultipleLocator(base=0.0025)
        ax.yaxis.set_minor_locator(loc_minor)

        plt.legend(legends,fontsize = 'x-small')
        ax.grid(which='both', alpha=0.3)
        plt.savefig('all_graphs_{}_{}_models_{}_extractors_{}'.format(dataset_name,target_mode,model_types[0],feature_extractor_types[0]),dpi=300)
