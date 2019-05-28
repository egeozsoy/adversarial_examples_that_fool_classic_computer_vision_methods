import os
from copy import deepcopy
import pickle
import random
from typing import Optional, Any, Union, List

import cv2
import numpy as np
from cv2.cv2 import BOWKMeansTrainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import foolbox
from fishervector import FisherVectorGMM

# these will not be found automatically because of the way they are important, they can be ignored
from configurations import vocab_size, data_size, image_size, batch_size, use_classes, n_features, gaussion_components, \
    feature_extractor_name, visualize_hog, \
    visualize_sift, model_name, force_model_reload, dataset_name, attack_name, save_correct_predictions, targeted_attack, no_feature_reload, \
    correct_predictions_folder, correct_predictions_file,matplotlib_backend

import matplotlib
matplotlib.use(matplotlib_backend)
from helpers.utils import filter_classes, get_balanced_batch, get_adversarial_test_set
from helpers.image_utils import show_image, plot_result
from helpers.feature_extractors import visualize_sift_points, hog_visualizer, get_feature_extractor, extract_sift_features, bow_extract, \
    initilize_fishervector_gmm
from helpers.foolbox_utils import FoolboxSklearnWrapper, find_closest_reference_image
from helpers.dataset_loader import load_data

if __name__ == '__main__':

    vocs_folder: str = 'vocs'
    models_folder: str = 'models'
    features_folder: str = 'features'
    evaluation_folder: str = 'evaluations'
    targeted_str: str = 'targeted' if targeted_attack else 'untargeted'
    evaluation_config_str:str = '{}_{}_{}_{}'.format(dataset_name, model_name, feature_extractor_name, targeted_str)
    evaluation_config_folder: str = os.path.join(evaluation_folder, evaluation_config_str)

    feature_extractor = get_feature_extractor(feature_extractor_name)

    if not os.path.exists(vocs_folder):
        os.mkdir(vocs_folder)

    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    if not os.path.exists(features_folder):
        os.mkdir(features_folder)

    if not os.path.exists(correct_predictions_folder):
        os.mkdir(correct_predictions_folder)

    if not os.path.exists(evaluation_folder):
        os.mkdir(evaluation_folder)

    if not os.path.exists(evaluation_config_folder):
        os.mkdir(evaluation_config_folder)

    iter_name: str = 'iter_dtn_{}_vs{}_ds{}_is{}_cc{}_nf{}_fe_{}'.format(dataset_name, vocab_size, data_size, image_size, len(use_classes), n_features,
                                                                         feature_extractor.__name__)

    print('Running iteration {}'.format(iter_name))

    print('Loading Data')
    X, y = load_data(image_size, dataset_name=dataset_name)

    X, y = filter_classes(X, y, keep_classes=use_classes)

    # we might want to resize images
    if image_size != X.shape[1]:
        print('Resizing images')
        X_resized = []
        for img in X:
            X_resized.append(cv2.resize(img, (image_size, image_size)))

        X = np.array(X_resized)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True, random_state=0)

    X_train = X_train[:data_size]
    X_test = X_test[:data_size]
    y_train = y_train[:data_size]
    y_test = y_test[:data_size]

    print('Dataset size {}'.format(X_train.shape[0]))

    if visualize_sift:
        print('Visualising images with sift points')
        for a in range(0, 100):
            print(y_train[a:a + 1])
            show_image(X_train[a])
            visualize_sift_points(X_train[a])

    if visualize_hog:
        for a in range(0, 100):
            print(y_train[a:a + 1])
            show_image(X_train[a])
            hog_visualizer(X_train[a])

    # bovw training
    if feature_extractor_name == 'bovw_extractor':

        if not os.path.exists(os.path.join(vocs_folder, 'voc_{}.npy'.format(iter_name))):
            bow_train: BOWKMeansTrainer = cv2.BOWKMeansTrainer(vocab_size)
            # Fill bow with sift calculations
            print('Calculating Vocabulary for training images')
            images_with_problems: int = 0
            for idx, train_image in enumerate(X_train):

                kp, desc = extract_sift_features(train_image)
                if desc is None:
                    images_with_problems += 1
                else:
                    bow_train.add(desc)

            print('No sift features were found for {} images'.format(images_with_problems))
            print('Generating vocabulary by clustering')

            voc = bow_train.cluster()
            np.save(os.path.join(vocs_folder, 'voc_{}.npy'.format(iter_name)), voc)

        else:
            print('Loaded Vocabulary')
            voc = np.load(os.path.join(vocs_folder, 'voc_{}.npy'.format(iter_name)))

        bow_extract.setVocabulary(voc)

    # fisher kernel training
    if feature_extractor_name == 'fishervector_extractor':
        if not os.path.exists(os.path.join(features_folder, 'fisherkernel_{}'.format(iter_name))):
            # Fill bow with sift calculations
            print('Calculating Sift Features for training images')
            images_with_problems = 0
            training_features = None
            for idx, train_image in enumerate(X_train):

                kp, desc = extract_sift_features(train_image)

                # sometimes, sift extracts less than it is suppose to
                if desc is None or desc.shape[0] < n_features:
                    images_with_problems += 1
                else:
                    desc = desc[:n_features]
                    desc = np.expand_dims(desc, axis=0).astype(np.float32)  # increase this if more accuracy is required
                    if training_features is None:
                        training_features = desc
                    else:
                        training_features = np.concatenate([training_features, desc], axis=0)

            # Generally errors occur if less than the requested amount of sift features were found per image,
            # in training time, we ignore these images, in test time, we use zero vector to represent these images.
            # Obviously makes it impossible to predict that image, so we want to choose a n_features count for sift that minimizes this error, while still giving good results.
            print('Errors occurred for {} images'.format(images_with_problems))

            print('Calculating Fisher Kernel for training images')
            fishervector_gmm: FisherVectorGMM = FisherVectorGMM(n_kernels=gaussion_components).fit(training_features)

            joblib.dump(fishervector_gmm, os.path.join(features_folder, 'fisherkernel_{}'.format(iter_name)))

        else:
            print('Loaded Fisher Kernel')
            fishervector_gmm = joblib.load(os.path.join(features_folder, 'fisherkernel_{}'.format(iter_name)))

        initilize_fishervector_gmm(fishervector_gmm)

    iter_name = '{}_gc_{}'.format(iter_name, gaussion_components)
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
            early_stopper: EarlyStopping = EarlyStopping(patience=20, verbose=1, restore_best_weights=True)

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

    # visualize some predictions
    for i in range(0):
        if model_name == 'cnn':
            str_label: str = str(model.predict((X_test[i:i + 1] / 255) - 0.5)[0])
        else:
            str_label = str(model.predict(X_test_extracted[i:i + 1])[0])
        show_image(X_test[i], '{}-{}'.format(y_test[i], str_label))

    # ----------START OF ADVERSARIAL IMAGE GENERATION---------------
    # 1. From the testing set, pick N amount of images which are also classified correctly
    # TODO maybe it is a better idea to pick a random set for every model(one for each class or something like that),
    #  instead of relying on classes which are classified by all as correct
    # TODO we can use get_balanced_batch for this

    # TODO we can delete these okey
    # # We won't need this if we decide to pick target randomly as well
    # if os.path.exists('target_indices.pt'):
    #     target_labels = pickle.load(open('target_indices.pt', 'rb'))
    #     target_calculation_needed: bool = False
    #
    # else:
    #     target_labels = {}
    #     target_calculation_needed = True

    if targeted_attack:
        print('TARGETED ATTACK')
    else:
        print('UNTARGETED ATTACK')

    adversarial_prediction_idx = get_adversarial_test_set(predictions_from_testing, y_test)
    # 2. Iterate over this data, and apply a targeted and untargeted attacks in case of imagenette, untargeted in case of inria(already binary)
    # TODO implemment for inria
    for idx, test_idx in enumerate(adversarial_prediction_idx):
        test_image = np.float32(X_test[test_idx])
        label = int(y_test[test_idx])

        # 3. The target label will be randomly picked from the remaining 9 classes ( currently we load if we so every model uses the same target.
        # If we use different data sets, this will become irrelevant)

        # if test_idx not in target_labels:
        possible_target_classes: List[int] = [i for i in range(len(use_classes)) if i != label]
        target_label: int = random.choice(possible_target_classes)

        #     TODO we can delete these probably
        #     target_labels[test_idx] = target_label
        # else:
        #     target_label = target_labels[test_idx]

        # 4. Get a reference image, this should fullfill the following criterias: The image belongs to the target class, and is also classified as such
        # Among the many images that fullfill this criteria, we pick the image that has the least distance to the image we want to attack
        reference_image: np.ndarray = np.float32(find_closest_reference_image(test_image, X_test, y_test, predictions_from_testing, label, target_label))

        # 5. Initilize attack type, also our model and criteria.
        if targeted_attack:
            criterion = foolbox.criteria.TargetClass(target_label)
        else:
            criterion = foolbox.criteria.Misclassification()

        print('Image mean:{},std:{}'.format(test_image.mean(), test_image.std()))

        fmodel = FoolboxSklearnWrapper(bounds=(0, 255), channel_axis=2, feature_extractor=feature_extractor, predictor=model)
        if attack_name == 'BoundaryPlusPlus':
            iter: int = 1000  # because max_queries will stop us
            attack = foolbox.attacks.BoundaryAttackPlusPlus(model=fmodel, criterion=criterion)
        elif attack_name == 'Boundary':
            iter = 2000
            attack = foolbox.attacks.BoundaryAttack(model=fmodel, criterion=criterion)
        else:
            raise Exception('ATTACK NOT KNOWN')

        # 6. Run, results will be saved in a attack_log.csv file for every image
        try:
            adversarial: Optional[Any] = attack(deepcopy(test_image), label, verbose=True, iterations=iter, starting_point=reference_image, max_queries=1000,log_name='attack_{}.csv'.format(evaluation_config_str))

        except Exception as e:
            print(e)
            continue

        # 7. Save the results appropiataly.
        print('Original image predicted as {}'.format(label))
        if model_name != 'cnn':
            adv_label = model.predict(feature_extractor(np.array([adversarial])))[0]
        else:
            adv_label = int(model.predict(np.array([adversarial]))[0])

        print('Adverserial image predicted as {}'.format(adv_label))
        if adversarial is not None:
            plot_result(np.float32(cv2.cvtColor(np.uint8(test_image), cv2.COLOR_RGB2BGR)), np.float32(cv2.cvtColor(np.uint8(adversarial), cv2.COLOR_RGB2BGR)))

        os.rename('attack_{}.csv'.format(evaluation_config_str), os.path.join(evaluation_config_folder, '{}_{}.csv'.format(idx, test_idx)))

    # TODO if we are going to use this,fix TypeError: file must have a 'write' attribute
    # if target_calculation_needed:
    #     pickle.dump(target_labels, 'target_indices.pt')
