import numpy as np
import pickle
import os
import sys
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from evaluate_models import get_signal_background
sys.path.append('../')
from utils.train_test_split import split_train_test
from utils.HiggsBosonUtils import AMS
from submission.make_submission import create_submission


def custom_loss_function(y_true, y_pred):
    s, b = get_signal_background(y_true, y_pred, weights_val)
    loss = AMS(s, b)
    print(loss)
    return loss

def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
        model.add(keras.layers.Dense(1))
        optimizer = keras.optimizers.SGD(lr=learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
        # model.compile(optimizer='adam', loss=custom_loss_function)
    return model


if __name__ == '__main__':
    OUTPUT_MODELS_PATH = "./trained-models"
    os.makedirs(OUTPUT_MODELS_PATH, exist_ok=True)

    X = np.loadtxt("../../data/output/X_train.txt")
    y = np.loadtxt("../../data/output/y_train.txt")
    w = np.loadtxt("../../data/output/w_train.txt")

    X_train, X_val, y_train, y_val, weights_train, weights_val = split_train_test(X, y, w)

    keras_reg = keras.wrappers.scikit_learn.KerasClassifier(build_model, input_shape=X_train.shape[1])
    # checkpoint_cb = keras.callbacks.ModelCheckpoint("nn.h5", save_best_only=True)
    keras_reg.fit(X_train, y_train, epochs=100,
                    validation_data=(X_val, y_val),
                    callbacks=[keras.callbacks.EarlyStopping(patience=5)])
    
    output_submission_path = "./output"
    INPUT_MODELS_PATH = "/trained-models"
    INPUT_DATA_PATH = "../../data/output"
    INPUT_RAW_DATA_PATH = "../../data/input"
    df_test = pd.read_csv(f'{INPUT_RAW_DATA_PATH}/test.csv')
    X_test = np.loadtxt(f"{INPUT_DATA_PATH}/X_test.txt")
    create_submission("nn_sub", keras_reg.predict(X_test), keras_reg.predict_proba(X_test), df_test['EventId'])
    