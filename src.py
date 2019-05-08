import os
from copy import deepcopy

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import foolbox

from configurations import vocab_size, data_size, image_size, use_classes, n_features, feature_extractor_name, visualize_hog, visualize_sift, model_name,dataset_name,attack_name
from helpers.utils import filter_classes
from helpers.image_utils import show_image, plot_result
from helpers.feature_extractors import visualize_sift_points, hog_visualizer, get_feature_extractor, sift, extract_sift_features, bow_extract
from helpers.foolbox_utils import FoolboxSklearnWrapper,find_closest_reference_image
from helpers.dataset_loader import load_data

if __name__ == '__main__':

    vocs_folder = 'vocs'
    models_folder = 'models'
    features_folder = 'features'

    feature_extractor = get_feature_extractor(feature_extractor_name)

    if not os.path.exists(vocs_folder):
        os.mkdir(vocs_folder)

    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    if not os.path.exists(features_folder):
        os.mkdir(features_folder)

    iter_name = 'iter_dtn_{}_vs{}_ds{}_is{}_cc{}_nf{}_fe_{}'.format(dataset_name, vocab_size, data_size, image_size, len(use_classes), n_features,
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

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

    if feature_extractor_name == 'bovw_extractor' and model_name != 'cnn':

        if not os.path.exists(os.path.join(vocs_folder, 'voc_{}.npy'.format(iter_name))):
            bow_train = cv2.BOWKMeansTrainer(vocab_size)
            # Fill bow with sift calculations
            print('Calculating Vocabulary for training images')
            images_with_problems = 0
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

    # Define where we should save
    full_model_path = os.path.join(models_folder, '{}_{}'.format(model_name, iter_name))
    full_features_path = os.path.join(features_folder, '{}_{}'.format(model_name, iter_name))

    if model_name == 'cnn':
        from helpers.keras_train import get_keras_scikitlearn_model, get_keras_features_labels, dropout_images

        model = get_keras_scikitlearn_model(X_train.shape, len(use_classes))

        if os.path.exists(full_features_path):
            print('Loading Features Labels')
            features_labels = joblib.load(full_features_path)
        else:
            print('Generating and saving Features Labels')
            features_labels = get_keras_features_labels(X_train, X_test, y_train, y_test, len(use_classes))
            joblib.dump(features_labels, full_features_path)

        X_train_extracted, X_test_extracted, y_train, y_test = features_labels

        if os.path.exists(full_model_path):
            print('Loading Model')
            model = joblib.load(full_model_path)

        else:
            print('Starting Model training {} and saving model'.format(model_name))
            model.fit(dropout_images(X_train_extracted), y_train, validation_data=(dropout_images(X_test_extracted), y_test))
            joblib.dump(model, full_model_path)

    else:
        if model_name == 'svc':
            model = LinearSVC()
        elif model_name == 'bagged_svc':
            base_model = LinearSVC()
            # this train many svms like random forest does with decision trees
            model = BaggingClassifier(base_model, n_jobs=-1,
                                      n_estimators=100)  # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
        elif model_name == 'forest':
            model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        elif model_name == 'logreg':
            model = LogisticRegression()
        else:
            model = LinearSVC()

        if os.path.exists(full_features_path):
            print('Loading Features Labels')
            features_labels = joblib.load(full_features_path)

        else:
            print('Generating and saving Features Labels')
            features_labels = (feature_extractor(X_train), feature_extractor(X_test))
            joblib.dump(features_labels, full_features_path)

        X_train_extracted, X_test_extracted = features_labels

        if os.path.exists(full_model_path):
            model = joblib.load(full_model_path)

        else:

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
        label = str(model.predict(X_test_extracted[i:i + 1])[0])
        show_image(X_test[i], '{}-{}'.format(y_test[i], label))

    test_idx = 1
    reference_idx = 0

    # starting adversarial
    test_image = np.float32(X_test[test_idx])

    if model_name != 'cnn':
        label = y_test[test_idx]
    else:
        label = int(np.argmax(y_test[test_idx]))

    reference_image:np.ndarray = np.float32(find_closest_reference_image(test_image,X_test,predictions_from_testing,label))

    adversarial = None
    # stop if unsuccessful after #timeout trials
    timeout = 5

    while adversarial is None and timeout >= 0:
        fmodel = FoolboxSklearnWrapper(bounds=(0, 255), channel_axis=2, feature_extractor=feature_extractor, predictor=model)

        if attack_name == 'BoundaryPlusPlus':
            iter = 20
            attack = foolbox.attacks.BoundaryAttackPlusPlus(model=fmodel)
        elif attack_name == 'Boundary':
            iter = 2000
            attack = foolbox.attacks.BoundaryAttack(model=fmodel)
        else:
            raise Exception('ATTACK NOT KNOWN')

        # multiply the image with 255, to reverse the normalization before the activations
        adversarial = attack(deepcopy(test_image), label, verbose=True, iterations=iter, starting_point=reference_image)
        timeout -= 1

    print('Original image predicted as {}'.format(label))
    if model_name != 'cnn':
        adv_label = model.predict(feature_extractor(np.array([adversarial])))[0]
    else:
        adv_label = int(np.argmax(model.predict(np.array([adversarial]))[0]))

    print('Adverserial image predicted as {}'.format(adv_label))
    if adversarial is not None:
        plot_result(np.float32(cv2.cvtColor(np.uint8(test_image), cv2.COLOR_RGB2BGR)), np.float32(cv2.cvtColor(np.uint8(adversarial), cv2.COLOR_RGB2BGR)))
