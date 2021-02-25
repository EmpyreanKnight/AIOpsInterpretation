# models used (LDA, QDA, LR, CART, GBDT, NN, RF, RGF)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from rgf.sklearn import RGFClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics, preprocessing
from sklearn.utils.fixes import loguniform
import scipy.stats as stats
from sklearn.utils import resample
import pandas as pd
import numpy as np
import traceback
import mkl
import os

import warnings

mkl.set_num_threads(1)  # control the number of thread used for NN model
N_WORKERS = 1  # control the number of workers used for the RF and GBDT model

INPUT_FOLDER = r'../../data/'
GOOGLE_INPUT_FILE = r'google_job_failure.csv'
BACKBLAZE_INPUT_FILE = r'disk_failure_v2.csv'


def obtain_tuned_model(model_name, features, labels):
    '''
    Return the model tuned through random search
    The model is already fit on the whole training data so no more training needed

    Args:
        model_name: the name of model, need to be all lowercase letter (lda, qda, lr, cart, gbdt, nn, rf, rgf)
    '''
    model = obtain_untuned_model(model_name)

    N_ITER = 100
    # control the iteration of random search for time-consuming models
    if model_name in ['rf', 'nn', 'rgf', 'lr', 'gbdt']:
        N_ITER = 10

    random_search = RandomizedSearchCV(model, param_distributions=obtain_param_dist(model_name), n_iter=N_ITER, scoring='roc_auc', cv=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        random_search.fit(features, labels)

    print("Best parameters (AUC {0}): {1}".format(random_search.best_score_, random_search.best_params_))

    return random_search.best_estimator_


def obtain_param_dist(model_name):
    '''
    INTERNAL USE
    return the parameter list for random searching model configurations

    Args:
        model_name: the name of model, need to be all lowercase letter (lda, qda, lr, cart, gbdt, nn, rf, rgf)
    '''
    param_dist = None

    if model_name == 'lda':
        param_dist = [
            {'solver': ['lsqr', 'eigen'], 'shrinkage': ['auto', 0.1, 0.3, 0.5, 0.7, 0.9]},
            {'solver': ['svd'], 'tol': loguniform(1e-5, 1)}
        ]
    elif model_name == 'qda':
        param_dist = {
            'reg_param': loguniform(1e-5, 1),
            'tol': loguniform(1e-5, 1)
        }
    elif model_name == 'lr':
        param_dist = [
            {'solver': ['newton-cg', 'lbfgs', 'sag'], 'C': loguniform(1e-5, 1e2), 'penalty': ['l2', 'none']},
            {'solver': ['saga'], 'C': loguniform(1e-5, 1e2), 'penalty': ['l1', 'l2', 'none']},
            {'solver': ['liblinear'], 'C': loguniform(1e-5, 1e2), 'penalty': ['l1']}
        ]
    elif model_name == 'cart':
        param_dist = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 4, 8],
            'class_weight':['balanced', None]
        }
    elif model_name == 'gbdt':
        param_dist = {
            'n_estimators': stats.randint(1e1, 1e2),
              'learning_rate': stats.uniform(1e-2, 1),
              'max_depth': stats.randint(2, 10),
              'subsample': loguniform(5e-1, 1),
              'booster': ['gbtree', 'gblinear', 'dart']
             }
    elif model_name == 'nn':
        param_dist = {
            'hidden_layer_sizes': [(8,), (16,), (32,), 
                                   (8, 8), (16, 16),],
            'activation': ['tanh', 'relu', 'logistic'],
            'solver': ['sgd', 'adam', 'lbfgs'],
            'alpha': loguniform(1e-4, 1e-2),
            'learning_rate': ['constant', 'adaptive', 'invscaling'],
            'max_iter': stats.randint(1e1, 2e2)
        }
    elif model_name == 'rf':
        param_dist = {
            'n_estimators': stats.randint(1e1, 1e2),
            'criterion': ['gini', 'entropy'],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [int(x) for x in np.linspace(10, 110, num=6)] + [None],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 4, 8],
            'class_weight':['balanced', None],
            'bootstrap': [True, False]
        }
    elif model_name == 'rgf':
        param_dist = {
            'max_leaf': stats.randint(1e2, 5e3),
            'loss': ['LS', 'Log', 'Expo', 'Abs'],
            'l2': [1, 0.1, 0.01, 1e-10],
            'min_samples_leaf': [8, 10, 16],
            'learning_rate': loguniform(1e-4, 1)
        }
    return param_dist


def obtain_untuned_model(model_name):
    '''
    Return model with default configurations

    Args:
        model_name: the name of model, need to be all lowercase letter (lda, qda, lr, cart, gbdt, nn, rf, rgf)
    '''
    model = None
    if model_name == 'lda':
        model = LinearDiscriminantAnalysis()
    elif model_name == 'qda':
        model = QuadraticDiscriminantAnalysis()
    elif model_name == 'lr':
        model = LogisticRegression()
    elif model_name == 'cart':
        model = DecisionTreeClassifier()
    elif model_name == 'gbdt':
        model = XGBClassifier(n_jobs=N_WORKERS)
    elif model_name == 'nn':
        model = MLPClassifier()
    elif model_name == 'rf':
        model = RandomForestClassifier(n_jobs=N_WORKERS)
    elif model_name == 'rgf':
        model = SafeRGF()
    return model


def obtain_data(dataset, interval='m'):
    '''
    Read CSV file and return processed features and labels
    
    Args:
        dataset (chr): Assuming only Backblaze (b) and Google (g) datasets exists
        interval (chr): For Backblaze only. The interval of timestamp, by default the numbering (1~36) of month (m). See get_disk_data() for more details.

    Returns:
        features (np.array): feature array, the first column is the timestamp, shape of (n_samples, 1+n_features)
        labels (np.array): binary label array, shape of (n_samples,)
    '''
    if dataset == 'g':
        return get_google_data()
    elif dataset == 'b':
        return get_disk_data(interval)


def get_google_data():
    '''
    INTERNAL USE
    Read the Google dataset from csv file
    The input folder and filename are specified in constant variables above
    Return features and labels after proper preprocessing (categorical variable encoding, label conversion)
    Note that it only returns the features after redundancy and correlation tests

    Returns:
        features (np.array): feature array, the first column is the timestamp, shape of (n_samples, 1+n_features)
        labels (np.array): binary label array, shape of (n_samples,)
    '''
    path = os.path.join(INPUT_FOLDER, GOOGLE_INPUT_FILE)
    print('Loading data from', path)
    df = pd.read_csv(path)

    #columns = ['Start Time', 'User ID', 'Job Name', 'Scheduling Class',
    #           'Num Tasks', 'Priority', 'Diff Machine', 'CPU Requested', 'Mem Requested', 'Disk Requested',
    #           'Avg CPU', 'Avg Mem', 'Avg Disk', 'Std CPU', 'Std Mem', 'Std Disk']

    columns = ['Start Time', 'Scheduling Class',
               'Num Tasks', 'Priority', 'Diff Machine', 'CPU Requested', 'Disk Requested',
               'Avg CPU', 'Avg Mem', 'Avg Disk', 'Std Mem']
    print('Load complete')

    print('Preprocessing features')

    features = df[columns].to_numpy()
    labels = (df['Status']==3).to_numpy()

    print('Preprocessing complete\n')

    return features, labels


def get_disk_data(interval='d'):
    '''
    INTERNAL USE
    Read the Backblaze disk dataset from csv file
    The input folder and filename are specified in constant variables above
    Return features and labels after proper preprocessing (timestamp conversion)
    Note that it only returns the features after redundancy and correlation tests
    
    Args:
        interval (chr): the interval of timestamp, by default day of year (d)
        Possible choices also include month in a year (m)

    Returns:
        features (np.array): feature array, the first column is the timestamp, shape of (n_samples, 1+n_features)
        labels (np.array): binary label array, shape of (n_samples,)
    '''
    path = os.path.join(INPUT_FOLDER, BACKBLAZE_INPUT_FILE)
    print('Loading data from', path)
    df = pd.read_csv(path)
    print('Load complete')
    
    print('Preprocessing features')
    #df = df[['date',
    #    'smart_1_raw', 'smart_4_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_12_raw', 'smart_187_raw', 'smart_193_raw', 'smart_194_raw', 'smart_197_raw', 'smart_199_raw',
    #    'smart_4_raw_diff', 'smart_5_raw_diff', 'smart_9_raw_diff', 'smart_12_raw_diff', 'smart_187_raw_diff', 'smart_193_raw_diff', 'smart_197_raw_diff', 'smart_199_raw_diff',
    #    'label']]

    df = df[['date',
        'smart_1_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_12_raw', 'smart_187_raw', 'smart_193_raw', 'smart_194_raw', 'smart_197_raw', 'smart_199_raw',
        'smart_4_raw_diff', 'smart_5_raw_diff', 'smart_9_raw_diff', 'smart_187_raw_diff', 'smart_193_raw_diff', 'smart_197_raw_diff', 'smart_199_raw_diff',
        'label']]
    # change the date into days of a year as all data are in 2015
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    if interval == 'd':
        df['date'] = pd.Series(pd.DatetimeIndex(df['date']).dayofyear)
    elif interval == 'm':
        df['date'] = pd.Series((pd.DatetimeIndex(df['date']).year - 2015) * 12 + pd.DatetimeIndex(df['date']).month)
    else: 
        print('Invalid time interval argument for reading disk failure data. Possible options are (d, m).')
        exit(-1)
    
    features = df[df.columns[:-1]].to_numpy()
    labels = df[df.columns[-1]].to_numpy()

    return features, labels


def obtain_metrics(labels, probas):
    '''
    Calculate performance on various metrics (precision, recall, accuracy, F1, AUC, MCC, and brier score)

    Args: 
        labels (np.array): labels of samples, should be True/False
        probas (np.array): predicted probabilities of samples, should be in [0, 1]
            and should be generated with predict_proba()[:, 1]
    Returns:
        (list): [ Precision, Recall, Accuracy, F-Measure, AUC, MCC, Brier Score ]
    '''
    preds = probas > 0.5
    ret = []
    ret.append(metrics.precision_score(labels, preds))
    ret.append(metrics.recall_score(labels, preds))
    ret.append(metrics.accuracy_score(labels, preds))
    ret.append(metrics.f1_score(labels, preds))
    ret.append(metrics.roc_auc_score(labels, probas))
    ret.append(metrics.matthews_corrcoef(labels, preds))

    p, r, _ = metrics.precision_recall_curve(labels, probas)
    prc = metrics.auc(r, p)
    ret.append(prc)
    #ret.append(metrics.brier_score_loss(labels, probas))

    return ret


def downsampling(training_features, training_labels, ratio=10):
    '''
    Random downsampling of the training features and labels, by default it downsample to true/false ratio of 1:10

    Args:
        features (np.array): feature array, should be in shape (n_samples, n_features)
        labels (np.array): label array, should be in shape (n_samples,)
        ratio (int): target downsampling ratio, default as 10 (true/false ratio of 1:10)
    '''
    #return training_features, training_labels

    idx_true = np.where(training_labels == True)[0]
    idx_false = np.where(training_labels == False)[0]
    #print('Before dowmsampling:', len(idx_true), len(idx_false))
    idx_false_resampled = resample(idx_false, n_samples=len(idx_true)*ratio, replace=False)
    idx_resampled = np.concatenate([idx_false_resampled, idx_true])
    idx_resampled.sort()
    resampled_features = training_features[idx_resampled]
    resampled_labels = training_labels[idx_resampled]
    #print('After dowmsampling:', len(idx_true), len(idx_false_resampled))
    return resampled_features, resampled_labels

    
def obtain_intervals(dataset):
    '''
    INTERNAl USE
    Generate interval terminals, so that samples in each interval have:
        interval_i = (timestamp >= terminal_i) and (timestamp < terminal_{i+1})

    Args:
        dataset (chr): Assuming only Backblaze (b) and Google (g) datasets exists
    '''
    if dataset == 'g':
        # time unit in Google: millisecond, tracing time: 28 days
        start_time = 604046279
        unit_period = 24 * 60 * 60 * 1000 * 1000  # unit period: one day
        end_time = start_time + 28*unit_period
    elif dataset == 'b':
        # time unit in Backblaze: month, tracing time: 3 years (36 months)
        start_time = 1
        unit_period = 1  # unit period: one month
        end_time = start_time + 36*unit_period

    # add one unit for the open-end of range function
    terminals = [i for i in range(start_time, end_time+unit_period, unit_period)]

    return terminals


def obtain_period_data(dataset):
    '''
    This function partition the data into its natural time periods.
    For the Google data, return a list of features and a list of labels in 28 one-day periods;
    For the Backblaze data, return a list of features and a list of labels in 36 one-month periods.

    Args:
        dataset (chr): Assuming only Backblaze (b) and Google (g) datasets exists
    '''
    features, labels = obtain_data(dataset, 'm')
    terminals = obtain_intervals(dataset)
    feature_list = []
    label_list = []

    for i in range(len(terminals) - 1):
        idx = np.logical_and(features[:, 0] >= terminals[i], features[:, 0] < terminals[i + 1])
        feature_list.append(features[idx][:, 1:])
        label_list.append(labels[idx])
    return feature_list, label_list


class SafeRGF(RGFClassifier):
    '''
    Since the RGF model randomly fail to fit and throw exceptions, 
    we wrapped it up with try-catch block and report all-zero predictions once it failed
    '''
    def __init__(self,
                 max_leaf=1000,
                 test_interval=100,
                 algorithm="RGF_Sib",
                 loss="Log",
                 reg_depth=1.0,
                 l2=0.1,
                 sl2=None,
                 normalize=False,
                 min_samples_leaf=10,
                 n_iter=None,
                 n_tree_search=1,
                 opt_interval=100,
                 learning_rate=0.5,
                 calc_prob="sigmoid",
                 n_jobs=1,
                 memory_policy="generous",
                 verbose=0,
                 init_model=None):
        super(SafeRGF, self).__init__()
        self.max_leaf = max_leaf
        self.test_interval = test_interval
        self.algorithm = algorithm
        self.loss = loss
        self.reg_depth = reg_depth
        self.l2 = l2
        self.sl2 = sl2
        self.normalize = normalize
        self.min_samples_leaf = min_samples_leaf
        self.n_iter = n_iter
        self.n_tree_search = n_tree_search
        self.opt_interval = opt_interval
        self.learning_rate = learning_rate
        self.calc_prob = calc_prob
        self.n_jobs = n_jobs
        self.memory_policy = memory_policy
        self.verbose = verbose
        self.init_model = init_model
        self.is_foul = False

    def fit(self, X, y):
        try:
            super(SafeRGF, self).fit(X, y)
        except Exception:
            self.is_foul = True
            print('Error fitting the model.')
        else:
            self.is_foul = False
        
    def predict_proba(self, X):
        if self.is_foul:
            return np.hstack((np.ones((X.shape[0], 1)), np.zeros((X.shape[0], 1))))
        else:
            return super(SafeRGF, self).predict_proba(X)
    
    def predict(self, X):
        if self.is_foul:
            return np.zeros(X.shape[0]).astype(bool)
        else:
            return super(SafeRGF, self).predict(X)

