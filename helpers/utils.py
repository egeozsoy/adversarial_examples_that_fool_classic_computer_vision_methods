from typing import List, Tuple,Dict
import random
import os

import numpy as np
from sklearn.utils import shuffle

from configurations import save_correct_predictions, correct_predictions_file, use_classes, adversarial_test_size,max_queries


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

    group_by_class:Dict[int,List[int]] = {}

    final_indices:List[int] = []

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
    mean_graph = graphs.mean(axis=0)
    np.save(os.path.join(folder_name, 'mean_graph.npy'), mean_graph)

    plt.plot(mean_graph)
    plt.ylabel("mean of thresholds")
    plt.xlabel("queries")
    plt.savefig(os.path.join(folder_name, 'graph'))
    plt.close()

    return mean_graph

def create_folders(folder_names:List[str]):
    for folder_name in folder_names:
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import matplotlib.ticker as plticker

    filtering_keywords = ['inria','_untargeted']

    legends = []
    mean_graphs = []
    for folder in os.listdir('evaluations'):
        # skip folder based on filtering keywords
        skip_folder = False
        for keyword in filtering_keywords:
            if keyword not in folder:
                skip_folder = True
                break

        if skip_folder:
            continue
        folder_path = os.path.join('evaluations', folder)
        if not os.path.isdir(folder_path):
            continue

        legends.append(folder)
        mean_graphs.append(generate_graph_data(folder_path, max_queries))

    fig, ax = plt.subplots()
    for mean_graph in mean_graphs:
        ax.plot(mean_graph, linewidth=1.0)

    loc_major = plticker.MultipleLocator(base=0.005)  # locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc_major)
    loc_minor = plticker.MultipleLocator(base=0.0025)
    ax.yaxis.set_minor_locator(loc_minor)

    plt.legend(legends)
    ax.grid(which='both', alpha=0.3)
    plt.savefig('all_graphs_{}'.format(', '.join(filtering_keywords)))
