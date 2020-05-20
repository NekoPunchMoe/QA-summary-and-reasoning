import pandas as pd
from utils.textprocessing import transform_original_dataframe

def load_data(transform_data=True):
    # read raw data from csv file
    train_df = pd.read_csv('drive/kaikeba/Abstract/data/AutoMaster_TrainSet.csv', encoding='utf-8')
    test_df = pd.read_csv('drive/kaikeba/Abstract/data/AutoMaster_TestSet.csv', encoding='utf-8')

    # remove rows contain na
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # merge question column and dialog column
    if transform_data:
        trans_train_df = transform_original_dataframe(train_df)
        trans_test_df = transform_original_dataframe(test_df)
    else:
        trans_train_df = train_df
        trans_test_df = test_df

    train_question_list = trans_train_df['Question'].tolist()
    train_dialog_list = trans_train_df['Dialogue'].tolist()
    train_report_list = trans_train_df['Report'].tolist()

    test_question_list = trans_test_df['Question'].tolist()
    test_dialog_list = trans_test_df['Dialogue'].tolist()

    train_X = []
    train_Y = train_report_list
    for i, question in enumerate(train_question_list):
        curr_train = question + ' ' + train_dialog_list[i]
        curr_train = curr_train.replace('<start>', '').replace('<end>', '').strip()
        train_X.append(curr_train)

    test_X = []
    for i, question in enumerate(test_question_list):
        curr_test = question + ' ' + test_dialog_list[i]
        curr_test = curr_test.replace('<start>', '').replace('<end>', '').strip()
        test_X.append(curr_test)

    return train_X, train_Y, test_X


if __name__ == '__main__':
    train_X, train_Y, test_X = load_data()
