from sklearn import metrics
from comet_ml import Experiment
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.append("../../")
sys.path.append("../../utils")
from HiggsBosonUtils import AMS
from utils.comet_ import create_experiment, log_experiment, plot_cv_scores

def get_signal_background(y_true, y_pred, weights):
    s = weights[(y_pred==1) & (y_true==1)].sum()
    b = weights[(y_pred==1) & (y_true==0)].sum()
    return s, b

def AMSGridSearchCV(model, param_grid, cv, X, y, weights):
    best_score = 0
    skf = StratifiedKFold(n_splits=3, shuffle=False)

    weights_signal_bs = weights[y == 1].sum()
    weights_bg_bs = weights[y == 0].sum()
    
    for g in ParameterGrid(param_grid):
        model.set_params(**g)
        score = []
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            weights_train = weights[train_index]
            weights_val = weights[val_index]
            
            # ajuste de pesos
            weight_signal_as = weights_train[y_train == 1].sum()
            weight_bg_as = weights_train[y_train == 0].sum()
            weights_train[y_train == 1] *= (weights_signal_bs/weight_signal_as)
            weights_train[y_train == 0] *= (weights_bg_bs/weight_bg_as)

            weight_signal_as = weights_val[y_val == 1].sum()
            weight_bg_as = weights_val[y_val == 0].sum()
            weights_val[y_val == 1] *= (weights_signal_bs/weight_signal_as)
            weights_val[y_val == 0] *= (weights_bg_bs/weight_bg_as)

            model.fit(X_train, y_train, sample_weight=weights_train)
            # score con AMS
            y_pred = model.predict(X_val)
            s, b = get_signal_background(y_val, y_pred, weights_val)
            score.append(AMS(s, b))
        print(f'{score}, {g}\n\n')
        mean_score = np.mean(score)
        if mean_score > best_score:
            best_scores = score
            best_params = g
    return best_scores, best_params

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('NO se entreno ningun modelo.')
    else:
        INPUT_DATA_PATH = "../../data/output"
        data_training = pd.read_csv('../../data/input/training.csv')
        X_train = np.loadtxt(f"{INPUT_DATA_PATH}/X_train.txt")
        y_train = np.loadtxt(f"{INPUT_DATA_PATH}/y_train.txt")
        weights = np.array(data_training['Weight'])

        model_to_train = sys.argv[1]

        exp = create_experiment()
        if model_to_train == 'rf':
            model = RandomForestClassifier()
            rf_grid = {'n_estimators': [50],
                            'min_samples_leaf': [1],
                            'max_depth': [10],}
            best_scores, best_params = AMSGridSearchCV(model, param_grid=rf_grid, cv=3, X=X_train, y=y_train, weights=weights)
            log_experiment(exp, params=rf_grid, metrics=np.mean(best_scores))
            log_experiment(exp, best_params=best_params)
            plot_cv_scores(exp, best_scores)
            print('Se entreno un Random Forest.')
            print('best_scores: ', np.mean(best_scores))
            print('best_params: ', best_params)
        elif model_to_train == 'xgb':
            model = XGBClassifier()
            print('Se entrno un XGBoost.')
        else:
            print('NO se entrno ningun modelo. Las opciones de modelos a entranr son: rf, xbg')

