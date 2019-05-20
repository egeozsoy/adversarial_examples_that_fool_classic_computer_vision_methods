import os
from copy import deepcopy

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import foolbox
from fishervector import FisherVectorGMM

from configurations import vocab_size, data_size, image_size, batch_size, use_classes, n_features, gaussion_components, \
    feature_extractor_name, visualize_hog, \
    visualize_sift, model_name, force_model_reload, dataset_name, attack_name
from helpers.utils import filter_classes,get_balanced_batch
from helpers.image_utils import show_image, plot_result
from helpers.feature_extractors import visualize_sift_points, hog_visualizer, get_feature_extractor, extract_sift_features, bow_extract, \
    initilize_fishervector_gmm
from helpers.foolbox_utils import FoolboxSklearnWrapper, find_closest_reference_image
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

    # fisher kernel training
    if feature_extractor_name == 'fishervector_extractor' and model_name != 'cnn':
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
                    desc = np.expand_dims(desc, axis=0)
                    if training_features is None:
                        training_features = desc
                    else:
                        training_features = np.concatenate([training_features, desc], axis=0)

            print('Errors occurred for {} images'.format(images_with_problems))

            print('Calculating Fisher Kernel for training images')
            fishervector_gmm = FisherVectorGMM(n_kernels=gaussion_components).fit(training_features)

            joblib.dump(fishervector_gmm, os.path.join(features_folder, 'fisherkernel_{}'.format(iter_name)))

        else:
            print('Loaded Fisher Kernel')
            fishervector_gmm = joblib.load(os.path.join(features_folder, 'fisherkernel_{}'.format(iter_name)))

        initilize_fishervector_gmm(fishervector_gmm)

    iter_name = '{}_gc_{}'.format(iter_name, gaussion_components)
    # Define where we should save
    full_model_path = os.path.join(models_folder, '{}_{}'.format(model_name, iter_name))
    full_features_path = os.path.join(features_folder, '{}_{}'.format(model_name, iter_name))

    # model selection
    if model_name == 'cnn':
        from helpers.keras_train import get_keras_scikitlearn_model, get_keras_features_labels, dropout_images
        from keras.callbacks import EarlyStopping

        model = get_keras_scikitlearn_model(X_train.shape, len(use_classes))

    else:
        # if no batchsize is used, we can work with the typical sklearn models, else we have to use models that support partial fit
        if model_name == 'svc':
            model = LinearSVC()
        elif model_name == 'forest':
            model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        elif model_name == 'logreg':
            model = LogisticRegression()
        elif model_name == 'sgd_svc':
            model = SGDClassifier(max_iter=1000,tol=1e-3,warm_start=True,n_jobs=-1)
        else:
            raise Exception('model_name not known')

    # model training
    model_training_needed = True
    if os.path.exists(full_model_path) and not force_model_reload:
        print('Loading Model, not going to train')
        model = joblib.load(full_model_path)
        model_training_needed = False

    if model_name == 'cnn':
        #we only do one big balanced batch as keras wrapper for scikitlearn doesn't support partial fit.
        if not model_training_needed:
            batch_size = 6000 # smaller batchsize because we just want to see some results
        batch_train_x, batch_train_y, batch_cv_x, batch_cv_y = get_balanced_batch(X_train, y_train, batch_size, use_classes)
        features_labels = get_keras_features_labels(batch_train_x, batch_cv_x,X_test, batch_train_y,batch_cv_y, y_test, len(use_classes))
        X_train_extracted, X_cv_extracted, batch_x_test_extract, batch_train_y, batch_cv_y, batch_test_y = features_labels

        if model_training_needed:
            print('Starting Model training {} and saving model'.format(model_name))
            early_stopper = EarlyStopping(patience=20, verbose=1, restore_best_weights=True)

            model.fit(dropout_images(X_train_extracted), batch_train_y, validation_data=(dropout_images(X_cv_extracted), batch_cv_y),callbacks=[early_stopper])
            joblib.dump(model, full_model_path)

        print('Training set performance {}'.format(model.score(dropout_images(X_train_extracted), batch_train_y)))
        print('CV set performance {}'.format(model.score(dropout_images(X_cv_extracted), batch_cv_y)))
        print('Testing set performance {}'.format(model.score(dropout_images(batch_x_test_extract), batch_test_y)))
        predictions_from_testing = model.predict(dropout_images(batch_x_test_extract))

    elif model_name == 'sgd_svc':
        #TODO make sure these are also normalized

        fitted = False
        best_cross_val_score = -1

        for epoch in range(20):
            batch_train_x,batch_train_y,batch_cv_x, batch_cv_y = get_balanced_batch(X_train,y_train,batch_size,use_classes)

            samples_per_class = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}

            for elem in batch_train_y:
                samples_per_class[elem] += 1

            samples_per_class = sorted(samples_per_class.items(), key=lambda kv: kv[1])

            print('Generating and saving Features Labels')
            features_labels = (feature_extractor(batch_train_x),feature_extractor(batch_cv_x),feature_extractor(X_test))

            X_train_extracted, X_cv_extracted, X_test_extracted = features_labels

            if model_training_needed:
                print('Starting Model training {} and saving model'.format(model_name))
                if not fitted:
                    print('Fitting initial model')
                    model.fit(X_train_extracted, batch_train_y)
                    fitted = True
                else:
                    for i in range(10): # as partialfit is not as effective(max_iter=1), we can call it more than once for the same data
                        model.partial_fit(X_train_extracted, batch_train_y,classes=use_classes)

            print('Class distribution for batch is : {}'.format(samples_per_class))

            print('Training set performance {}'.format(model.score(X_train_extracted, batch_train_y)))
            cross_val_performance = model.score(X_cv_extracted, batch_cv_y)
            # We can implement early stopping if we want, we currently don't
            print('CrossVal set performance {}'.format(cross_val_performance))
            print('Testing set performance {}'.format(model.score(X_test_extracted, y_test)))

            predictions_from_testing = model.predict(X_test_extracted)

            samples_per_class = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

            for elem in predictions_from_testing:
                samples_per_class[elem] += 1

            samples_per_class = sorted(samples_per_class.items(), key=lambda kv: kv[1])

            print('Class distribution for testset prediction is : {}'.format(samples_per_class))

            print('Sample predictions from testing {}'.format(predictions_from_testing[:20]))
            print('Ground truth for        testing {}'.format(y_test[:20]))

            if model_training_needed and cross_val_performance > best_cross_val_score:
                print('Saving model with score {}'.format(cross_val_performance))
                joblib.dump(model, full_model_path)
                best_cross_val_score = cross_val_performance


    else:
        if os.path.exists(full_features_path):
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

    if model_name != 'sgd_svc' and model_name != 'cnn': # because if we used batchsize, this values dont really make sense
        print('Training set performance {}'.format(model.score(X_train_extracted, y_train)))
        print('Testing set performance {}'.format(model.score(X_test_extracted, y_test)))

        print('Sample predictions from training {}'.format(model.predict(X_train_extracted[:20])))
        print('Ground truth for        training {}'.format(y_train[:20]))
        predictions_from_testing = model.predict(X_test_extracted)
        print('Sample predictions from testing {}'.format(predictions_from_testing[:20]))
        print('Ground truth for        testing {}'.format(y_test[:20]))

    # visualize some predictions
    for i in range(5):
        if model_name == 'cnn':
            label = str(model.predict(X_test[i:i + 1])[0])
        else:
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

    reference_image: np.ndarray = np.float32(find_closest_reference_image(test_image, X_test, predictions_from_testing, label))

    adversarial = None
    # stop if unsuccessful after #timeout trials
    timeout = 5


    #TODO CHECK image values(where are there between 0 and 1, where 0 and 255 etc.)
    while adversarial is None and timeout >= 0:
        fmodel = FoolboxSklearnWrapper(bounds=(0, 255), channel_axis=2, feature_extractor=feature_extractor, predictor=model)

        if attack_name == 'BoundaryPlusPlus':
            iter = 70
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
