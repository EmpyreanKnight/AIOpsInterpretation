import numpy as np
from sklearn.metrics import mean_squared_error
from utilities import obtain_tuned_model, downsampling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.proportion import proportions_ztest
from sklearn.model_selection import KFold


class ConceptDetection(object):
    def __init__(self, model_name, window_size, dataset):
        self.name = "Base detector"
        self.feature_list = []
        self.label_list = []
        self.window_size = window_size
        self.model_name = model_name
        self.dataset = dataset
        self.scaler = StandardScaler()
        self.model = None
        self.retrain = False
    
    def initial_fit(self, X_list, y_list):
        self.feature_list = X_list.copy()
        self.label_list = y_list.copy()
        #print([X.shape[0] for X in self.feature_list])
        X = np.vstack(self.feature_list[-self.window_size:])
        y = np.hstack(self.label_list[-self.window_size:])
        X = self.scaler.fit_transform(X)
        X, y = downsampling(X, y)
        self.model = obtain_tuned_model(self.model_name, self.dataset, len(self.feature_list)+1, 'w')
        self.model.fit(X, y)

    def fit(self, X, y):
        self.feature_list.append(X)
        self.label_list.append(y)
        #print([X.shape[0] for X in self.feature_list])
        if self.detect_concept():
            print(self.name + ': concept drift detected, retrain the model.')
            X = np.vstack(self.feature_list[-self.window_size:])
            y = np.hstack(self.label_list[-self.window_size:])
            X = self.scaler.fit_transform(X)
            X, y = downsampling(X, y)
            self.model = obtain_tuned_model(self.model_name, self.dataset, len(self.feature_list)+1, 'w')
            self.model.fit(X, y)
            self.retrain = True
        else:
            print(self.name + ': no concept drift detected, keep the model.')
            self.retrain = False

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def get_name(self):
        return self.name

    def detect_concept(self):
        pass

    def update_setting(self, X, y):
        pass

    def is_retrained(self):
        return self.retrain


class SlidingWindow(ConceptDetection):
    def __init__(self, model_name, window_size, dataset):
        super().__init__(model_name, window_size, dataset)
        self.name = "Sliding Window"

    def detect_concept(self):
        return True


class StaticModel(ConceptDetection):
    def __init__(self, model_name, window_size, dataset):
        super().__init__(model_name, window_size, dataset)
        self.name = "Static Model"

    def detect_concept(self):
        return False


class HistoryRetrain(ConceptDetection):
    def __init__(self, model_name, window_size, dataset):
        super().__init__(model_name, window_size, dataset)
        self.name = "Drift History"

    def detect_concept(self):
        retrun True
        
    def fit(self, X, y):
        self.feature_list.append(X)
        self.label_list.append(y)

        print(self.name + ': concept drift detected, retrain the model.')
        X = np.vstack(self.feature_list)
        y = np.hstack(self.label_list)
        X = self.scaler.fit_transform(X)
        X, y = downsampling(X, y)        
        self.model = obtain_tuned_model(self.model_name, self.dataset, len(self.feature_list)+1, 'w')
        self.model.fit(X, y)
        self.retrain = True
        

class GamaDetector(ConceptDetection):
    def __init__(self, model_name, window_size, dataset):
        super().__init__(model_name, window_size, dataset)
        self.name = "Gama"
        self.p_min = 1.0
        self.s_min = 1.0
        self.start_point = window_size

    def detect_concept(self):
        testing_features = self.scaler.transform(np.vstack(self.feature_list[self.start_point:]))
        p = np.count_nonzero(np.hstack(self.label_list[self.start_point:]) != self.model.predict(testing_features)) / testing_features.shape[0]
        s = np.sqrt((p * (1-p)) / testing_features.shape[0])
        #print(p, s, self.p_min, self.s_min)
        if p + s < self.p_min + self.s_min:
            self.p_min = p
            self.s_min = s
            return False
        elif p + s >= self.p_min + 3*self.s_min:
            self.p_min = 1.0
            self.s_min = 1.0
            self.start_point = len(self.feature_list)
            return True
        return False


class HarelDetector(ConceptDetection):
    def __init__(self, model_name, window_size, dataset):
        super().__init__(model_name, window_size, dataset)
        self.name = "Harel"
        self.P = 10
        self.Delta = 0.0
        self.delta = 1 / self.P

    def loss_function(self, y_pred, y_true):
        return mean_squared_error(y_true.astype(float), y_pred)

    def detect_concept(self):
        # obtain risk for ordered permutation
        training_features = self.feature_list[-2]
        training_labels = self.label_list[-2]
        testing_features = self.feature_list[-1]
        testing_labels = self.label_list[-1]
        scaler = StandardScaler()
        training_features = scaler.fit_transform(training_features)
        training_features, training_labels = downsampling(training_features, training_labels)
        model = obtain_tuned_model(self.model_name, self.dataset, len(self.feature_list)-1, 'p')
        model.fit(training_features, training_labels)
        R_ord = self.loss_function(model.predict_proba(scaler.transform(testing_features))[:, 1], testing_labels)

        # obtain risks for random permutations
        R_rnds = []
        for _ in range(self.P):
            features = np.vstack(self.feature_list[-2:])
            labels = np.hstack(self.label_list[-2:])

            training_features, testing_features, training_labels, testing_labels = train_test_split(features, labels, test_size = 0.5)
            scaler = StandardScaler()
            training_features = scaler.fit_transform(training_features)
            training_features, training_labels = downsampling(training_features, training_labels)
            model = obtain_tuned_model(self.model_name, self.dataset, len(self.feature_list)-1, 'p')
            model.fit(training_features, training_labels)
            R_rnds.append(self.loss_function(model.predict_proba(scaler.transform(testing_features))[:, 1], testing_labels))
        
        # test concept drift
        value = (1 + np.sum(R_ord - np.array(R_rnds) <= self.Delta)) / (self.P + 1)
        if value <= self.delta:
            return True
        else:
            return False


class ZTestDetector(ConceptDetection):
    def __init__(self, model_name, window_size, dataset):
        super().__init__(model_name, window_size, dataset)
        self.name = "Z-test"

    def detect_concept(self):
        training_features = np.vstack(self.feature_list[-1-self.window_size:-1])
        training_labels = np.hstack(self.label_list[-1-self.window_size:-1])
        testing_features = self.feature_list[-1]

        model = obtain_tuned_model(self.model_name, self.dataset, len(self.feature_list)-1, 'p')
        training_err, training_size = ZTestDetector.cross_validation(model, training_features, training_labels)

        scaler = StandardScaler()
        training_features = scaler.fit_transform(training_features)
        training_features, training_labels = downsampling(training_features, training_labels)
        model.fit(training_features, training_labels)

        testing_err = np.count_nonzero(self.label_list[-1] != self.model.predict(scaler.transform(testing_features)))
        _, pval = proportions_ztest([training_err, testing_err], [training_size, testing_features.shape[0]])
        
        return pval < 0.05  # null hypothesis: prop_0 == prop_1

    @staticmethod
    def cross_validation(model, features, labels):
        kf = KFold(n_splits=10, shuffle=False)
        error_num = 0
        total_num = 0
        for training_index, testing_index in kf.split(features):
            training_features, training_labels = features[training_index], labels[training_index]
            testing_features, testing_labels = features[testing_index], labels[testing_index]

            scaler = StandardScaler()
            training_features = scaler.fit_transform(training_features)
            training_features, training_labels = downsampling(training_features, training_labels)
            model.fit(training_features, training_labels)

            testing_preds = model.predict(scaler.transform(testing_features))
            error_num += np.count_nonzero(testing_preds != testing_labels)
            total_num += len(testing_labels)

        return error_num, total_num
