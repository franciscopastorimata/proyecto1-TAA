import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.append("../../utils")
from HiggsBosonUtils import AMS


def get_signal_background(y_true, y_pred, weights):
    s = weights[(y_pred==1) & (y_true==1)].sum()
    b = weights[(y_pred==1) & (y_true==0)].sum()
    return s, b


def AMSGridSearchCV(model , param_grid, cv, X, y, weights):
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
            weights_test = weights[val_index]
            
            # ajuste de pesos
            weight_signal_as = weights_train[y_train == 1].sum()
            weight_bg_as = weights_train[y_train == 0].sum()
            weights_train[y_train == 1] *= (weights_signal_bs/weight_signal_as)
            weights_train[y_train == 0] *= (weights_bg_bs/weight_bg_as)

            weight_signal_as = weights_test[y_val == 1].sum()
            weight_bg_as = weights_test[y_val == 0].sum()
            weights_test[y_val == 1] *= (weights_signal_bs/weight_signal_as)
            weights_test[y_val == 0] *= (weights_bg_bs/weight_bg_as)

            model.fit(X_train, y_train)
            # score con AMS
            y_pred = model.predict(X_val)
            s, b = get_signal_background(y_val, y_pred, weights_test)
            score.append(AMS(s, b))
            print(score)
        mean_score = np.mean(score)
        if mean_score > best_score:
            best_scores = score
            best_params = g
    return best_scores, best_params        