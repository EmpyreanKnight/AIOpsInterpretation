import os
import timeit
import argparse
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from utilities import obtain_period_data, obtain_metrics, downsampling, obtain_tuned_model

from ensemble_model import AWEModel, SEAModel
from concept_drift_detection import StaticModel, SlidingWindowRetrain, HistoryRetrain

GOOGLE_OUTPUT_FILE = r'update_google_'
BACKBLAZE_OUTPUT_FILE = r'update_disk_'
MODEL_FOLDER = r'./saved_models/'
SAVE_MODEL = True
N_ROUNDS = 100
MODEL_NAME = ''
RANDOM_CONST = 114514


def experiment_driver(feature_list, label_list, out_file, n_round):
    out_columns = ['Scenario', 'Model', 'Period', 'Test P', 'Test R', 'Test A', 'Test F', 'Test AUC', 'Test MCC', 'Test B'] + ['Importance_'+str(i) for i in range(feature_list[0].shape[1])]
    out_ls = []
    
    num_periods = len(feature_list)
    print('Total number of periods:', num_periods)

    ensemble_models = [
        SEAModel(MODEL_NAME, num_periods//2),
        AWEModel(MODEL_NAME, num_periods//2)
    ]

    retrain_models = [
        #StaticModel(MODEL_NAME, num_periods//2),
        SlidingWindowRetrain(MODEL_NAME, num_periods//2),
        HistoryRetrain(MODEL_NAME, num_periods//2)
    ]

    # list of features and labels for the windowed approaches
    training_feature_list = []
    training_label_list = []

    # iterate through every time period
    for i in range(num_periods):
        print('Fitting models on period', i + 1)

        # Proprocessing: scaler on the training data -> downsample the training data -> apply the scaler on the testing data
        training_features = feature_list[i]
        training_labels = label_list[i]

        scaler = StandardScaler()
        scaler.fit(training_features)
        np.random.seed(RANDOM_CONST+n_round+i)  # control the randomness of downsampling
        training_features, training_labels = downsampling(training_features, training_labels)
        training_feature_list.append(training_features)
        training_label_list.append(training_labels)

        # for retrain models, no individual scaler needed as periods are grouped into window then scale
        for retrain_model in retrain_models:
            retrain_model.fit(training_feature_list, training_label_list)

        # use the same trained model on each ensemble to reduce randomness
        fitted_model = obtain_tuned_model(MODEL_NAME, scaler.transform(training_features), training_labels)
        for ensemble_model in ensemble_models:
            ensemble_model.fit(feature_list[i], label_list[i], fitted_model, scaler)

        # we don't predict on the first half of data (but still need to fit ensembles on each period)
        if i < num_periods//2 - 1:
            continue
        
        for model in ensemble_models + retrain_models:
            # store model to the MODEL_FOLDER, naming as {output_file_name}_R_{n_round}_P_{n_period}.joblib
            if SAVE_MODEL:
                pickle_file = MODEL_FOLDER + out_file[:-4] + '_R' + str(n_round) + '_P' + str(i+1) + '.joblib'
                dump(model, pickle_file)

            feature_importance = permutation_importance(model, training_features, training_labels, scoring='roc_auc')

            # save output (execution info, feature importance, and model evaluation for all except the last period (no testing data for it))
            if i == num_periods - 1:
                out_ls.append([model.get_name(), MODEL_NAME.upper(), i + 1] + [0.0]*7 + feature_importance.importances_mean.tolist())
            else:
                testing_features = feature_list[i + 1]
                testing_labels = label_list[i + 1]
                # models have scalers inside
                out_ls.append([model.get_name(), MODEL_NAME.upper(), i + 1] + obtain_metrics(testing_labels, model.predict_proba(testing_features)[:, 1]) + feature_importance.importances_mean.tolist())

        # append the output CSV for each iteration
        out_df = pd.DataFrame(out_ls[-(len(ensemble_models)+len(retrain_models)):], columns=out_columns)
        out_df.to_csv(out_file, mode='a', index=False, header=(not os.path.isfile(out_file)))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment on updating approaches')
    parser.add_argument("-m", help="specify the model, random forest by default.", required=True, choices=['lda', 'qda', 'lr', 'cart', 'gbdt', 'nn', 'rf', 'rgf'])
    parser.add_argument("-d", help="specify the dataset, d for Googole and b for Backblaze.", required=True, choices=['g', 'b'])
    parser.add_argument("-n", help="specify the testing rounds, 100 by default.", default=100)
    parser.add_argument("-s", help="starting from this iteration.")
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(save=True)
    args = parser.parse_args()

    N_ROUNDS = int(args.n)
    MODEL_NAME = args.m.lower()
    SAVE_MODEL = args.save
    feature_list, label_list = obtain_period_data(args.d)

    print('Number of iterations:', N_ROUNDS)
    print('Save model?', SAVE_MODEL)
    print(f'Choose {MODEL_NAME.upper()} as model')

    if args.d == 'g':
        print('Choose Google as dataset')
        OUTPUT_FILE = GOOGLE_OUTPUT_FILE + args.m + '.csv'
        DATASET = 'Google'
    elif args.d == 'b':
        print('Choose Backblaze as dataset')
        OUTPUT_FILE = BACKBLAZE_OUTPUT_FILE + args.m + '.csv'
        DATASET = 'Backblaze'
    else:
        exit(-1)

    # don't remove the old files to be able to have multiple parallel models
    start_round = 0
    if args.s != None:
        start_round = int(args.s)

    random_constant = 114514
    for i in range(start_round, start_round+N_ROUNDS):
        print('Round', i+1)
        experiment_driver(feature_list, label_list, OUTPUT_FILE, i)
        
    print('Experiment completed!')
