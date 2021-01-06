import os
import argparse
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from utilities import obtain_period_data, obtain_metrics, downsampling, obtain_tuned_model

GOOGLE_OUTPUT_PREFIX = r'importance_google_'
BACKBLAZE_OUTPUT_PREFIX = r'importance_disk_'
MODEL_FOLDER = r'./saved_models/'
SAVE_MODEL = True
N_ROUNDS = 100
MODEL_NAME = ''


def experiment_driver(feature_list, label_list, out_file, n_round):
    out_columns = ['Model', 'Period', 'Test P', 'Test R', 'Test A', 'Test F', 'Test AUC', 'Test MCC', 'Test B'] + ['Importance_'+str(i) for i in range(feature_list[0].shape[1])]
    out_ls = []

    num_periods = len(feature_list)
    print('Total number of periods:', num_periods)

    # iterate through every time period
    for i in range(num_periods):
        print('Fitting models on period', i + 1)
        training_features = feature_list[i]
        training_labels = label_list[i]

        # Proprocessing: scaler on the training data -> downsample the training data -> apply the scaler on the testing data
        scaler = StandardScaler()
        training_features = scaler.fit_transform(training_features)
        training_features, training_labels = downsampling(training_features, training_labels)

        # tune and fit model -> calculate permutation importance
        model = obtain_tuned_model(MODEL_NAME, training_features, training_labels)
        #model.fit(training_features, training_labels)  # no need to fit model as the above line already fit it
        feature_importance = permutation_importance(model, training_features, training_labels, scoring='roc_auc')

        # store model to the MODEL_FOLDER, naming as {output_file_name}_R_{n_round}_P_{n_period}.joblib
        if SAVE_MODEL:
            pickle_file = MODEL_FOLDER + out_file[:-4] + '_R' + str(n_round) + '_P' + str(i+1) + '.joblib'
            dump(model, pickle_file)

        # save output (execution info, feature importance, and model evaluation for all except the last period (no testing data for it))
        if i == num_periods - 1:
            out_ls.append([MODEL_NAME.upper(), i + 1] + [0.0]*7 + feature_importance.importances_mean.tolist())
        else:
            testing_features = feature_list[i + 1]
            testing_labels = label_list[i + 1]
            testing_features = scaler.transform(testing_features)
            out_ls.append([MODEL_NAME.upper(), i + 1] + obtain_metrics(testing_labels, model.predict_proba(testing_features)[:, 1]) + feature_importance.importances_mean.tolist())

    # append the output CSV for each iteration
    out_df = pd.DataFrame(out_ls, columns=out_columns)
    out_df.to_csv(out_file, mode='a', index=False, header=(not os.path.isfile(out_file)))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment on feature importance on each period')
    parser.add_argument("-m", help="specify the model, random forest by default.", required=True, choices=['lda', 'qda', 'lr', 'cart', 'gbdt', 'nn', 'rf', 'rgf'])
    parser.add_argument("-d", help="specify the dataset, d for Googole and b for Backblaze.", required=True, choices=['g', 'b'])
    parser.add_argument("-n", help="specify the testing rounds, 100 by default.", default=100)
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
        OUTPUT_FILE = GOOGLE_OUTPUT_PREFIX + args.m + '.csv'
    elif args.d == 'b':
        print('Choose Backblaze as dataset')
        OUTPUT_FILE = BACKBLAZE_OUTPUT_PREFIX + args.m + '.csv'
    else:
        exit(-1)
    print()

    # remove the output file if exists
    if os.path.isfile(OUTPUT_FILE): 
        os.remove(OUTPUT_FILE)
    print('Output path:', OUTPUT_FILE)

    for i in range(N_ROUNDS):
        print('Round', i)
        experiment_driver(feature_list, label_list, OUTPUT_FILE, i)
        
    print('Experiment completed!')
