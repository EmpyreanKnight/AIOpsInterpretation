import numpy as np
from sklearn.metrics import mean_squared_error
from utilities import obtain_tuned_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.proportion import proportions_ztest
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator


class ConceptDetection(BaseEstimator):
    def __init__(self, model_name, window_size):
        self.name = "Base detector"
        self.window_size = window_size
        self.model_name = model_name
        self.scaler = StandardScaler()
        self.model = None
        self.retrain = False

    def fit(self, X_list, y_list):
        if len(X_list) < self.window_size:
            return
        
        if len(X_list) == self.window_size:
            # initial fit
            X = np.vstack(X_list)
            y = np.hstack(y_list)
            X = self.scaler.fit_transform(X)
            self.model = obtain_tuned_model(self.model_name, X, y)
            return

        if self.detect_concept():
            print(self.name + ': concept drift detected, retrain the model.')
            X = np.vstack(X_list[-self.window_size:])
            y = np.hstack(y_list[-self.window_size:])
            X = self.scaler.fit_transform(X)
            self.model = obtain_tuned_model(self.model_name, X, y)
            self.retrain = True
        else:
            print(self.name + ': no concept drift detected, keep the model.')
            self.retrain = False

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def detect_concept(self):
        pass

    def get_name(self):
        return self.name

    def is_retrained(self):
        return self.retrain


class SlidingWindowRetrain(ConceptDetection):
    def __init__(self, model_name, window_size):
        super().__init__(model_name, window_size)
        self.name = "Sliding Window"

    def detect_concept(self):
        return True


class StaticModel(ConceptDetection):
    def __init__(self, model_name, window_size):
        super().__init__(model_name, window_size)
        self.name = "Static Model"

    def detect_concept(self):
        return False


class HistoryRetrain(ConceptDetection):
    def __init__(self, model_name, window_size):
        super().__init__(model_name, window_size)
        self.name = "History Window"

    def detect_concept(self):
        return True
        
    def fit(self, X_list, y_list):
        if len(X_list) < self.window_size:
            return

        if len(X_list) == self.window_size or self.detect_concept():
            X = np.vstack(X_list)
            y = np.hstack(y_list)
            X = self.scaler.fit_transform(X)
            self.model = obtain_tuned_model(self.model_name, X, y)
            if len(X_list) > self.window_size:
                print(self.name + ': concept drift detected, retrain the model.')
                self.retrain = True
        else:
            print(self.name + ': no concept drift detected, keep the model.')
            self.retrain = False
