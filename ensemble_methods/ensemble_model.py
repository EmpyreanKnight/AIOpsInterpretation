import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from utilities import obtain_tuned_model, downsampling
from sklearn.preprocessing import StandardScaler

MSE_R = 0.25
EPS = 1e-5


class EnsembleModel(object):
    def __init__(self, model_name, ensemble_size, dataset):
        self.name = "Base ensemble"
        self.models = []
        self.weights = []
        self.scalers = []
        self.add = False
        self.dataset = dataset
        self.model_name = model_name
        self.ensemble_size = ensemble_size

    def fit(self, X, y, period):
        pass
                
    def predict_proba(self, X):
        probas = np.array([self.models[i].predict_proba(self.scalers[i].transform(X))[:, 1] for i in range(len(self.models))])
        return np.clip(np.average(probas, axis=0, weights=self.weights), 0.0, 1.0)

    def get_name(self):
        return self.name
    
    def is_added(self):
        return self.add


class SEAModel(EnsembleModel):
    def __init__(self, model_name, ensemble_size, dataset):
        super().__init__(model_name, ensemble_size, dataset)
        self.name = "SEA"
        self.last_model = None
        self.last_scaler = None
    
    def fit(self, X, y, period):
        model = obtain_tuned_model(self.model_name, self.dataset, period, 'p')
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X, y = downsampling(X, y)
        model.fit(X, y)

        if self.last_model == None: # first model
            self.last_model = model
            self.last_scaler = scaler
            return

        if len(self.models) == 0:
            self.models.append(model)
            self.scalers.append(scaler)
            self.weights.append(0.0)
            self.add = True
            return

        classifier_probas = np.array([self.models[i].predict_proba(self.scalers[i].transform(X))[:, 1] for i in range(len(self.models))])
        classifier_preds = classifier_probas > 0.5
        ensemble_preds = np.average(classifier_probas, axis=0) > 0.5
        occurrence = np.vstack([np.count_nonzero(classifier_preds, axis=0), np.count_nonzero(np.logical_not(classifier_preds), axis=0)])
        P_1 = np.max(occurrence, axis=0) / len(self.models) # top voted class
        P_2 = np.min(occurrence, axis=0) / len(self.models) # 2nd voted class
        P_cor = np.count_nonzero(classifier_preds == ensemble_preds, axis=0) / len(self.models) # correct class

        # evaluate C_{i-1}
        model_preds = self.last_model.predict(self.last_scaler.transform(X))
        P_new = np.count_nonzero(classifier_preds == model_preds, axis=0) / len(self.models) # new classifier class
        quality = 0.0
        quality += np.sum(np.logical_and(ensemble_preds == y, model_preds == y).astype(int) * (1 - np.abs(P_1 - P_2)))
        quality += np.sum(np.logical_and(ensemble_preds != y, model_preds == y).astype(int) * (1 - np.abs(P_1 - P_cor)))
        quality -= np.sum((model_preds != y).astype(int) * (1 - np.abs(P_cor - P_new)))

        # evaluate ensemble members
        for idx in range(len(self.models)):
            model_preds = self.models[idx].predict(self.scalers[idx].transform(X))
            P_new = np.count_nonzero(classifier_preds == model_preds, axis=0) / len(self.models)
            quality = self.weights[idx]
            quality += np.sum(np.logical_and(ensemble_preds == y, model_preds == y).astype(int) * (1 - np.abs(P_1 - P_2)))
            quality += np.sum(np.logical_and(ensemble_preds != y, model_preds == y).astype(int) * (1 - np.abs(P_1 - P_cor)))
            quality -= np.sum((model_preds != y).astype(int) * (1 - np.abs(P_cor - P_new)))
            self.weights[idx] = quality

        if len(self.models) < self.ensemble_size:
            self.models.append(model)
            self.scalers.append(scaler)
            self.weights.append(quality)
            self.add = True
        else:
            if min(self.weights) < quality:
                idx = self.weights.index(min(self.weights))
                self.models[idx] = model
                self.scalers[idx] = scaler
                self.weights[idx] = quality
                self.add = True
            else:
                self.add = False

    def predict_proba(self, X):
        if len(self.models) == 0:
            return self.last_model.predict_proba(self.last_scaler.transform(X))[:, 1]
        probas = np.array([self.models[i].predict_proba(self.scalers[i].transform(X))[:, 1] for i in range(len(self.models))])
        return np.clip(np.average(probas, axis=0), 0.0, 1.0)


class AWEModel(EnsembleModel):
    def __init__(self, model_name, ensemble_size, dataset):
        super().__init__(model_name, ensemble_size, dataset)
        self.name = "AWE"
        
    def fit(self, X, y, period):
        model = obtain_tuned_model(self.model_name, self.dataset, period, 'p')
        scaler = StandardScaler()
        weight = MSE_R - self.MSE_with_CV(X, y, period)

        X = scaler.fit_transform(X)
        X, y = downsampling(X, y)
        model.fit(X, y)
        
        for idx in range(len(self.models)):
            proba = self.models[idx].predict_proba(self.scalers[idx].transform(X))[:, 1]
            mse = mean_squared_error(y, proba)
            self.weights[idx] = max(0, MSE_R - mse)

        if len(self.models) < self.ensemble_size:
            self.models.append(model)
            self.scalers.append(scaler)
            self.weights.append(weight)
            self.add = True
        else:
            if min(self.weights) < weight:
                idx = self.weights.index(min(self.weights))
                self.models[idx] = model
                self.scalers[idx] = scaler
                self.weights[idx] = weight  
                self.add = True
            else:
                self.add = False
   
    def MSE_with_CV(self, X, y, period):
        kf = KFold(n_splits=10, shuffle=False)
        model = obtain_tuned_model(self.model_name, self.dataset, period, 'p')
        mse_list = []
        for training_index, testing_index in kf.split(X):
            training_features, training_labels = X[training_index], y[training_index]
            testing_features, testing_labels = X[testing_index], y[testing_index]

            scaler = StandardScaler()
            training_features = scaler.fit_transform(training_features)
            training_features, training_labels = downsampling(training_features, training_labels)
            model.fit(training_features, training_labels)

            testing_proba = model.predict_proba(scaler.transform(testing_features))[:, 1]
            mse = mean_squared_error(testing_labels, testing_proba)
            mse_list.append(mse)
        return np.mean(mse_list)
