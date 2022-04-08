import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

INPUT_DATA_PATH = "../../data/input"
OUTPUT_DATA_PATH = "../../data/output"

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
    return data.drop(columns=["EventId", "Weight", "Label", "DER_mass_MMC", "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi", 
    "PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi"])

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
            X_file_name, y_file_name = 'X_train.txt', 'y_train.txt'
        elif input_file_name == 'test.csv':
            X_file_name, y_file_name = 'X_test.txt', 'y_test.txt'
        elif  input_file_name == 'random_submission.csv':
            X_file_name, y_file_name = 'X_test.txt', 'y_test.txt'
    data = pd.read_csv(f"{INPUT_DATA_PATH}/{input_file_name}")
    data = map_s_and_b_values(data)
    data = change_999values_to_NaN(data)
    y = get_target(data)
    data = drop_unused_columns(data)
    X = change_Nan_values_by_the_mean(data)
    np.savetxt(f"{OUTPUT_DATA_PATH}/{X_file_name}", X)
    np.savetxt(f"{OUTPUT_DATA_PATH}/{y_file_name}", y, fmt='%d')
    