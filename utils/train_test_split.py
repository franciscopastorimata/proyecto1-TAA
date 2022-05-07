from sklearn.model_selection import StratifiedShuffleSplit

def split_train_test(X, y, weights):
    
    weights_signal_bs = weights[y == 1].sum()
    weights_bg_bs = weights[y == 0].sum()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, val_index in sss.split(X, y):
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

    return X_train, X_val, y_train, y_val, weights_train, weights_val