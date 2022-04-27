import os
import sys; sys.path.append("../../utils/HiggsBosonUtils.py")
import numpy as np
import pandas as pd

from HiggsBosonUtils import check_submission

OUTPUT_SUBMISSION_PATH = f"{os.path.dirname(__file__)}/output"

def create_submission(file_name, y_pred, y_proba, eventIds):
    n_events = eventIds.size
    df_submission = pd.DataFrame({'EventId': eventIds,'RankOrder':y_proba[:, 1],'Class':y_pred.astype(int)})
    df_submission = df_submission.sort_values(by=['RankOrder'], ignore_index=True)
    df_submission['RankOrder'] = pd.Series(np.arange(1, n_events + 1, dtype=int))
    df_submission['Class'] = df_submission['Class'].map({0: 'b', 1: 's'})
    os.makedirs(OUTPUT_SUBMISSION_PATH, exist_ok=True)
    output_path = f"{OUTPUT_SUBMISSION_PATH}/{file_name}.csv"
    df_submission = df_submission.sort_values(by=['EventId'], ignore_index=True)
    df_submission.to_csv(output_path, index = False)
    if check_submission(output_path, n_events):
        print('the submission has consistent information')
    print(output_path)
    return df_submission

if __name__ == '__main__':
    pass
