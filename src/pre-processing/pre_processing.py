import sys
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

INPUT_DATA_PATH = "../../data/input"
OUTPUT_DATA_PATH = "../../data/output"
os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

def map_s_and_b_values(data):
    data['Label'] = data['Label'].map({'b': 0, 's': 1})
    return data

def get_target(data):
    return data["Label"].to_numpy()  

def change_999values_to_NaN(data):
    for col in data.columns:
        data[col] = np.where(data[col] == -999.0, np.nan, data[col])
    return data    

def drop_unused_columns(data):
    return data.drop(columns=["EventId", "PRI_jet_leading_eta", "PRI_jet_leading_phi", 
    "PRI_jet_subleading_eta", "PRI_jet_subleading_phi", "DER_sum_pt", "PRI_met_sumet", "DER_deltaeta_jet_jet",
    "PRI_met_phi", "PRI_lep_phi", "PRI_lep_eta", "PRI_tau_eta", "PRI_tau_phi", 'DER_lep_eta_centrality', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet'])

def change_Nan_values_by_the_mean(data):
    # this function recives a pandas data frame and returns two numpy arrays
    atrib = data.columns
    pipe = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
    preprocessor = ColumnTransformer([('preprocessing', pipe, atrib)])
    X = preprocessor.fit_transform(data)
    return X


if __name__ == '__main__':
    if len(sys.argv) == 1:
        input_file_name = 'training.csv'
        X_file_name = 'X_train.txt'
        y_file_name = 'y_train.txt'
    else:
        input_file_name = sys.argv[1]
        if input_file_name == 'training.csv':
            X_file_name, y_file_name, w_file_name = 'X_train.txt', 'y_train.txt', 'w_train.txt'
        elif input_file_name == 'test.csv':
            X_file_name, y_file_name = 'X_test.txt', 'y_test.txt'
    data = pd.read_csv(f"{INPUT_DATA_PATH}/{input_file_name}")
    if input_file_name == 'training.csv':
        data = map_s_and_b_values(data)
        y = get_target(data)
        np.savetxt(f"{OUTPUT_DATA_PATH}/{y_file_name}", y, fmt='%d')
        w = np.array(data['Weight'])
        np.savetxt(f"{OUTPUT_DATA_PATH}/{w_file_name}", w)
        data = data.drop(columns=["Weight", "Label"])
    data = change_999values_to_NaN(data)
    data = drop_unused_columns(data)
    X = change_Nan_values_by_the_mean(data)
    np.savetxt(f"{OUTPUT_DATA_PATH}/{X_file_name}", X)
    