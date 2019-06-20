import os
from copy import deepcopy
import random
from typing import Optional, Any, Union, List
from multiprocessing import Pool,cpu_count

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import foolbox

# these will not be found automatically because of the way they are important, they can be ignored
from configurations import vocab_size, data_size, image_size, batch_size, use_classes, n_features, gaussion_components, \
    feature_extractor_name, visualize_hog, \
    visualize_sift, model_name, force_model_reload, dataset_name, targeted_attack, no_feature_reload, \
    matplotlib_backend,just_train,max_queries

import matplotlib

matplotlib.use(matplotlib_backend)

from helpers.utils import filter_classes, get_balanced_batch, get_adversarial_test_set, create_folders
from helpers.image_utils import show_image, plot_result
from helpers.feature_extractors import visualize_sift_points, hog_visualizer, get_feature_extractor, prepare_bovw_vocabulary, prepare_fishervector_gmm
from helpers.foolbox_utils import FoolboxSklearnWrapper, find_closest_reference_image
from helpers.dataset_loader import load_data


def attack_image(idx, test_idx):
    if idx < skip_n_images:
        return
    try:
        test_image = np.float32(X_test[test_idx])
        label = int(y_test[test_idx])

        # 3. The target label will be randomly picked from the remaining classes, this value doesn't play a role if the attack is untargeted
        possible_target_classes: List[int] = [i for i in range(len(use_classes)) if i != label]
        target_label: int = random.choice(possible_target_classes)

        # 4. Get a reference image, this should fullfill the following criterias: The image belongs to the target class, and is also classified as such
        # Among the many images that fullfill this criteria, we pick the image that has the least distance to the image we want to attack
        if targeted_attack:
            reference_image: np.ndarray = np.float32(find_closest_reference_image(attacked_img=test_image, reference_images=X_test, reference_labels=y_test,
                                                                                  reference_predictions=predictions_from_testing, original_label=label,
                                                                                  target_label=target_label))
        else:
            reference_image = np.float32(find_closest_reference_image(attacked_img=test_image, reference_images=X_test, reference_labels=y_test,
                                                                      reference_predictions=predictions_from_testing, original_label=label,
                                                                      target_label=None))

        # 5. Initilize attack type, our model and criteria.
        if targeted_attack:
            criterion = foolbox.criteria.TargetClass(target_label)
        else:
            criterion = foolbox.criteria.Misclassification()

        # Mainly for debugging purposes
        print('Image mean:{},std:{}'.format(test_image.mean(), test_image.std()))

        fmodel = FoolboxSklearnWrapper(bounds=(0, 255), channel_axis=2, feature_extractor=feature_extractor, predictor=model)
        iter: int = 100000  # max_queries will stop us before the iterations
        attack = foolbox.attacks.BoundaryAttackPlusPlus(model=fmodel, criterion=criterion)

        # currently using batchsize 1 for more realistic testing, but some models might profit from bigger batches
        # threshold 0.003 is a good limit
        # maybe play around with the query limit

        # 6. Run, results will be saved in a attack_{}.csv file for every image, with the corresponding config_str name
        adversarial: Optional[Any] = attack(deepcopy(test_image), label, verbose=True, iterations=iter, starting_point=reference_image, max_queries=max_queries,
                                            log_name='attack_{}_{}.csv'.format(evaluation_config_str, idx), batch_size=1)

        # 7. Save the results.
        print('Original image predicted as {}'.format(label))
        if model_name != 'cnn':
            adv_label = model.predict(feature_extractor(np.array([adversarial])))[0]
        else:
            adv_label = int(model.predict(np.array([adversarial]))[0])

        print('Adverserial image predicted as {}'.format(adv_label))

        save_name = os.path.join(adversarial_images_config_folder, '{}_{}'.format(evaluation_config_str, idx))
        if adversarial is not None:
            plot_result(np.float32(cv2.cvtColor(np.uint8(test_image), cv2.COLOR_RGB2BGR)),
                        np.float32(cv2.cvtColor(np.uint8(adversarial), cv2.COLOR_RGB2BGR)), save_name)

        os.rename('attack_{}_{}.csv'.format(evaluation_config_str, idx), os.path.join(evaluation_config_folder, '{}_{}.csv'.format(idx, test_idx)))

    except Exception as e:
        # if any errors occur, maybe because of the chosen target, skip that iteration
        print(e)


if __name__ == '__main__':

    # Avoid opencv multiprocessing bug that occurs when attacking with multiple processes. We can still use more threads if we are only training
    # https://github.com/opencv/opencv/issues/5150
    if not just_train:
        cv2.setNumThreads(0)


    vocs_folder: str = 'vocs'
    models_folder: str = 'models'
    features_folder: str = 'features'
    evaluation_folder: str = 'evaluations'
    adversarial_images_folder: str = 'adversarial_images'
    targeted_str: str = 'targeted' if targeted_attack else 'untargeted'
    evaluation_config_str: str = '{}_{}_{}_{}'.format(dataset_name, model_name, feature_extractor_name, targeted_str)
    evaluation_config_folder: str = os.path.join(evaluation_folder, evaluation_config_str)
    adversarial_images_config_folder = os.path.join(adversarial_images_folder, evaluation_config_str)

    feature_extractor = get_feature_extractor(feature_extractor_name)

    create_folders(
        [vocs_folder, models_folder, features_folder, evaluation_folder, evaluation_config_folder, adversarial_images_folder,
         adversarial_images_config_folder])

    iter_name: str = 'iter_dtn_{}_vs{}_ds{}_is{}_cc{}_nf{}_fe_{}'.format(dataset_name, vocab_size, data_size, image_size, len(use_classes), n_features,
                                                                         feature_extractor.__name__)

    print('Running iteration {}'.format(iter_name))

    print('Loading Data')
    X, y = load_data(image_size, dataset_name=dataset_name)
    X, y = filter_classes(X, y, keep_classes=use_classes)

    # we might want to resize images
    if image_size != X.shape[1]:
        X = np.array([cv2.resize(img, (image_size, image_size)) for img in X])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

    X_train = X_train[:data_size]
    X_test = X_test[:data_size]
    y_train = y_train[:data_size]
    y_test = y_test[:data_size]

    print('Dataset size {}'.format(X_train.shape[0]))

    if visualize_sift:
        print('Visualising images with sift points')
        for a in range(0, 10):
            print(y_train[a:a + 1])
            show_image(X_train[a])
            visualize_sift_points(X_train[a])

    if visualize_hog:
        for a in range(0, 10):
            print(y_train[a:a + 1])
            show_image(X_train[a])
            hog_visualizer(X_train[a])

    # bovw training
    if feature_extractor_name == 'bovw_extractor':
        prepare_bovw_vocabulary(X_train, vocs_folder, iter_name)

    iter_name = '{}_gc_{}'.format(iter_name, gaussion_components)
    # fisher kernel training
    if feature_extractor_name == 'fishervector_extractor':
        prepare_fishervector_gmm(X_train, features_folder, iter_name)

    # Define where we should save
    full_model_path: Union[bytes, str] = os.path.join(models_folder, '{}_{}'.format(model_name, iter_name))
    full_features_path: Union[bytes, str] = os.path.join(features_folder, '{}_{}'.format(model_name, iter_name))

    # model selection
    if model_name == 'cnn':
        from helpers.keras_train import get_keras_scikitlearn_model, get_keras_features_labels, dropout_images
        from keras.callbacks import EarlyStopping

        model = get_keras_scikitlearn_model(X_train.shape, len(use_classes))
    elif model_name == 'svc':
        model = LinearSVC()
    elif model_name == 'forest':
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    elif model_name == 'logreg':
        model = LogisticRegression()
    else:
        raise Exception('model_name not known')

    # model training
    model_training_needed: bool = True
    if os.path.exists(full_model_path) and not force_model_reload:
        print('Loading Model, not going to train')
        model = joblib.load(full_model_path)
        model_training_needed = False

    if model_name == 'cnn':
        # we only do one big balanced batch as keras wrapper for scikitlearn doesn't support partial fit.
        if not model_training_needed:
            batch_size = 1000  # if not training use smaller batchsize because we just want to see some results.
        batch_train_x, batch_train_y, batch_cv_x, batch_cv_y = get_balanced_batch(X_train, y_train, batch_size, use_classes)
        features_labels = get_keras_features_labels(batch_train_x, batch_cv_x, X_test, batch_train_y, batch_cv_y, y_test, len(use_classes))
        X_train_extracted, X_cv_extracted, batch_x_test_extract, batch_train_y, batch_cv_y, batch_test_y = features_labels

        if model_training_needed:
            print('Starting Model training {} and saving model'.format(model_name))
            early_stopper: EarlyStopping = EarlyStopping(patience=40, verbose=1, restore_best_weights=True)

            model.fit(dropout_images(X_train_extracted), batch_train_y, validation_data=(dropout_images(X_cv_extracted), batch_cv_y), callbacks=[early_stopper])
            joblib.dump(model, full_model_path)

        print('Training set performance {}'.format(model.score(dropout_images(X_train_extracted), batch_train_y)))
        print('CV set performance {}'.format(model.score(dropout_images(X_cv_extracted), batch_cv_y)))
        print('Testing set performance {}'.format(model.score(dropout_images(batch_x_test_extract), batch_test_y)))
        predictions_from_testing = model.predict(dropout_images(batch_x_test_extract))

    else:
        if os.path.exists(full_features_path) and not no_feature_reload:
            print('Loading Features Labels')
            features_labels = joblib.load(full_features_path)
        else:
            print('Generating and saving Features Labels')
            features_labels = (feature_extractor(X_train), feature_extractor(X_test))
            joblib.dump(features_labels, full_features_path)

        X_train_extracted, X_test_extracted = features_labels

        if model_training_needed:
            print('Starting Model training {} and saving model'.format(model_name))
            model.fit(X_train_extracted, y_train)
            joblib.dump(model, full_model_path)

        print('Training set performance {}'.format(model.score(X_train_extracted, y_train)))
        print('Testing set performance {}'.format(model.score(X_test_extracted, y_test)))

        print('Sample predictions from training {}'.format(model.predict(X_train_extracted[:20])))
        print('Ground truth for        training {}'.format(y_train[:20]))
        predictions_from_testing = model.predict(X_test_extracted)
        print('Sample predictions from testing {}'.format(predictions_from_testing[:20]))
        print('Ground truth for        testing {}'.format(y_test[:20]))

    # visualize some predictions, change number to something other than 0 to visualize
    for i in range(0):
        if model_name == 'cnn':
            str_label: str = str(model.predict((X_test[i:i + 1] / 255) - 0.5)[0])
        else:
            str_label = str(model.predict(X_test_extracted[i:i + 1])[0])
        show_image(X_test[i], '{}-{}'.format(y_test[i], str_label))

    # -----------------------------------START OF ADVERSARIAL IMAGE GENERATION-----------------------------------
    if not just_train:
        # 1. From the testing set, pick adversarial_test_size of images which are classified correctly
        if targeted_attack:
            print('TARGETED ATTACK')
        else:
            print('UNTARGETED ATTACK')

        adversarial_prediction_indices = get_adversarial_test_set(predictions_from_testing, y_test)

        # 2. Iterate over this data, and apply an attack
        skip_n_images = 0  # can be used if attack was interrupted eg. after 5 images. Set this value to 5 to skip the first 5 images in the next run

        numbers = [i for i in range(len(adversarial_prediction_indices))]
        # Cnn already uses all available resources, don't paralellise it
        if model_name != 'cnn':
            # Run attack in parallel
            p = Pool(cpu_count())
            # Starmap to feed more than one value
            p.starmap(attack_image, zip(numbers, adversarial_prediction_indices))

        else:
            # Run attack sequentially
            for number,adversarial_prediction_idx in zip(numbers,adversarial_prediction_indices):
                attack_image(number,adversarial_prediction_idx)
