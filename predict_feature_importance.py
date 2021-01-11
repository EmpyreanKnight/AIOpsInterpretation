# Comment the following two lines if running on windows. Windows does not support "fork"
from multiprocessing import set_start_method
set_start_method("fork", force=True)

import os
import argparse
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from utilities import obtain_period_data, obtain_metrics, downsampling, obtain_tuned_model, bootstrapping, obtain_untuned_model
from multiprocessing import Pool
import logging
from multiprocessing_logging import install_mp_handler

LOG_FOLDER = r'./logs/'
MODEL_FOLDER = r'./saved_models/'
OUTPUT_FOLDER = r'./outputs/'
SAVE_MODEL = True
N_ROUNDS = 10
MODEL_NAME = ['lda', 'qda', 'lr', 'cart', 'gbdt', 'nn', 'rf', 'rgf']

def run_experiment_on_period(period_id, num_periods, debug_periods, OUTPUT_PREFIX, BOOTSTRAP, MODEL_TUNE, training_features, training_labels, testing_features, testing_labels):
    # skip the period if not in the specified list
    if (debug_periods is not None) and (period_id not in debug_periods):
        logging.info("...Skipping period %s.", str(period_id) + '/' + str(num_periods))
        return

    try:
        logging.info("...Process for period %s is created.", str(period_id) + '/' + str(num_periods))

        OUTPUT_FILE = OUTPUT_FOLDER + OUTPUT_PREFIX + '_P' + str(period_id) + '.csv'
        # remove the output file if exists
        if os.path.isfile(OUTPUT_FILE): 
            os.remove(OUTPUT_FILE)
        logging.info('...Output file: %s', OUTPUT_FILE)

        for i in range(N_ROUNDS):
            logging.info("...Starting iteration %d.", i+1)
            out_columns = ['Model', 'Iteration', 'Test P', 'Test R', 'Test A', 'Test F', 'Test AUC', 'Test MCC', 'Test B', 'Test AP', 'Test AUPRC', 'Test AUPRC baseline'] + ['Importance_'+str(i) for i in range(training_features.shape[1])]
            out_ls = []

            if BOOTSTRAP:
                training_features, training_labels = bootstrapping(training_features, training_labels)
                logging.info("......Sample bootstrapped for iteration %d.", i+1)

            for learner in MODEL_NAME:
                # fit model
                logging.info("......Training model. Learner: %s.", learner)
                if MODEL_TUNE:
                    # tune and fit model
                    model = obtain_tuned_model(learner, training_features, training_labels)
                else:
                    # fit untuned model
                    model = obtain_untuned_model(learner)
                    model.fit(training_features, training_labels)
                logging.info("......Model trained. Learner: %s.", learner)
                
                # calculate permutation feature importance
                logging.info("......Calculating feature importance. Learner: %s.", learner)
                feature_importance = permutation_importance(model, training_features, training_labels, scoring='roc_auc')
                logging.info("......Feature importance calculated. Learner: %s.", learner)

                # store model to the MODEL_FOLDER
                if SAVE_MODEL:
                    logging.info("......Saving model. Learner: %s.", learner)
                    pickle_file = MODEL_FOLDER + OUTPUT_PREFIX + '_M' + learner + '_P' + str(period_id) + '_I' + str(i+1) + '.joblib'
                    dump(model, pickle_file)
                    logging.info("......Model saved to: %s", pickle_file)
                    
                # save output (execution info, feature importance, and model evaluation for all except the last period (no testing data for it))
                if period_id == num_periods:
                    out_ls.append([learner.upper(), i + 1] + [0.0]*10 + feature_importance.importances_mean.tolist())
                else:
                    out_ls.append([learner.upper(), i + 1] + obtain_metrics(testing_labels, model.predict_proba(testing_features)[:, 1]) + feature_importance.importances_mean.tolist())
                logging.info("......Result saved for learner: %s.", learner)

            # append results to the output CSV for each iteration
            out_df = pd.DataFrame(out_ls, columns=out_columns)
            out_df.to_csv(OUTPUT_FILE, mode='a', index=False, header=(not os.path.isfile(OUTPUT_FILE)))

            logging.info("...Results written. Iteration %d complete.", i+1)

        logging.info("Process for period %s finished.", str(period_id) + '/' + str(num_periods))

    except Exception as e:
        logging.error("Unknown error on period %d.", period_id, exc_info=True)        

def experiment_driver(dataset, tuned, bootstrapped, debug_periods):
    MODEL_TUNE = tuned
    BOOTSTRAP = bootstrapped
    DATASET_NAME = 'GOOGLE' if dataset == 'g' else 'BLACKBLAZE'
    OUTPUT_PREFIX = DATASET_NAME + ('_TUNED_' if MODEL_TUNE else '_UNTUNED_') + ('BOOTSTRAP' if BOOTSTRAP else 'STATIC')

    LOG_FILE = LOG_FOLDER + OUTPUT_PREFIX + '.log'
    print("...Start running experiment on %(dataset)s dataset. Log file: %(log)s" % {'dataset': DATASET_NAME, 'log': LOG_FILE})

    # initialize multiprocess logger
    logging.basicConfig(filename=LOG_FILE, filemode='a', format='[pid%(process)d-tid%(thread)d] %(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    install_mp_handler()

    # read dataset
    logging.info("Reading %s dataset...", DATASET_NAME)
    feature_list, label_list = obtain_period_data(dataset)
    global num_periods
    num_periods = len(feature_list)
    logging.info("...Dataset reading complete. Number of periods: %d.", num_periods)

    # scaler + downsample
    logging.info("Scaling and downsampling each time period...")
    downsampled_periods = []
    for i in range(num_periods):
        training_features = feature_list[i]
        training_labels = label_list[i]
        testing_features = feature_list[i + 1] if i < num_periods - 1 else None
        testing_labels = label_list[i + 1] if i < num_periods - 1 else None

        # Preprocessing: scaler on the training data -> downsample the training data -> apply the scaler on the testing data
        scaler = StandardScaler()
        training_features = scaler.fit_transform(training_features)
        if i < num_periods - 1:
            testing_features = scaler.transform(testing_features)
        training_features, training_labels = downsampling(training_features, training_labels)

        downsampled_periods.append((i+1, num_periods, debug_periods, OUTPUT_PREFIX, BOOTSTRAP, MODEL_TUNE, training_features, training_labels, testing_features, testing_labels))
        logging.info("...Period %d complete...", i+1)
    logging.info("...Scaling and downsampling complete.")
    
    logging.info("Creating processes for each time period...")
    period_pool = Pool(8)
    results = period_pool.starmap(run_experiment_on_period, downsampled_periods)
    logging.info("All processes finished. Experiment for dataset %s complete.", DATASET_NAME)

    print("...Finished running experiment for %(dataset)s dataset. Log file: %(log)s" % {'dataset': DATASET_NAME, 'log': LOG_FILE})

if __name__ == "__main__":
    DATASET = ['g', 'b']
    PERIODS = None

    parser = argparse.ArgumentParser(description='Experiment for RQ1')
    parser.add_argument("-t", help="tune hyperparameters", action='store_true')
    parser.add_argument("-b", help="bootstrap after downsampling", action='store_true')
    args = parser.parse_args()
    tuned = args.t
    bootstrapped = args.b

    # For debugging or limiting experiment scope
    #tuned = True
    #DATASET = ['g']
    #PERIODS = [4, 5]

    print("Experiment setup: Hyperparameter tuning: " + ('ON' if tuned else 'OFF') + " BOOTSTRAP: " + ('ON' if bootstrapped else 'OFF'))
    print("Start processing each dataset...")

    for data in DATASET:
        experiment_driver(data, tuned, bootstrapped, PERIODS)

    print('Experiment completed!')
