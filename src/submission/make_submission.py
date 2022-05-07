import os
import sys
import numpy as np
import pandas as pd
from tensorflow import keras

sys.path.append("../../utils")
from HiggsBosonUtils import check_submission

output_submission_path = f"{os.path.dirname(__file__)}/output"

def create_submission(file_name, y_pred, y_proba, eventIds):
    n_events = eventIds.size
    df_submission = pd.DataFrame({'EventId': eventIds,'RankOrder':y_proba[:, 1],'Class':y_pred.astype(int)})
    df_submission = df_submission.sort_values(by=['RankOrder'], ignore_index=True)
    df_submission['RankOrder'] = pd.Series(np.arange(1, n_events + 1, dtype=int))
    df_submission['Class'] = df_submission['Class'].map({0: 'b', 1: 's'})
    os.makedirs(output_submission_path, exist_ok=True)
    output_path = f"{output_submission_path}/{file_name}.csv"
    df_submission = df_submission.sort_values(by=['EventId'], ignore_index=True)
    df_submission.to_csv(output_path, index = False)
    if check_submission(output_path, n_events):
        print('The submission has consistent information')
    print(output_path)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('NO se genero ningun archivo de submission.')
    else:
        import pickle
        output_submission_path = "./output"
        INPUT_MODELS_PATH = "../models/trained-models"
        INPUT_DATA_PATH = "../../data/output"
        INPUT_RAW_DATA_PATH = "../../data/input"
        model_name = sys.argv[1]
        submission_name = sys.argv[2]
        df_test = pd.read_csv(f'{INPUT_RAW_DATA_PATH}/test.csv')
        X_test = np.loadtxt(f"{INPUT_DATA_PATH}/X_test.txt")
        if 'nn' in model_name:
            model = keras.models.load_model("../models/nn.h5")
        else:
            model = pickle.load(open(f"{INPUT_MODELS_PATH}/{model_name}.sav", 'rb'))
        create_submission(submission_name, model.predict(X_test), model.predict_proba(X_test), df_test['EventId'])
        print('the submission file has been successfully created')
        