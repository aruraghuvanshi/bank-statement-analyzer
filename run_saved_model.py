import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import numpy as np

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)


def load_test_data(model_name, cv_name, le_name, testfile=r'Output/master_output.csv'):

    '''
    Change upper and lower as per csv received from Axis to remove non-table areas.
    testfile: Name of the file to be achieve classification of Expesne category on.
    model_name: name of the model to be used for classification.
    cv_name: Name of the vectorizer to be used.
    le_name: Name of the label encoder to be used to inverse the label encoding to readable format.
    upper: number of cells from top to the header section of csv.
    lower: number of cells from the end of the table in csv to the bottom.

    '''

    pd.options.display.max_rows = 500
    pd.options.display.max_columns = 40

    print(f'\nFile model running on: \033[0;32m{testfile}\033[0m\n')
    dt = pd.read_csv(testfile, na_values=[' '])
    print(f'DataFrame of master_output file loaded into load test data:\n {dt}')

    nn = load_model(model_name)
    cv = pickle.load(open(cv_name, 'rb'))
    le = pickle.load(open(le_name, 'rb'))

    U = dt.PARTICULARS
    Ucv = cv.transform(U).toarray()

    solver = 'adam'
    loss = 'sparse_categorical_crossentropy'
    nn.compile(optimizer=solver, loss=loss, metrics=['accuracy'])

    dt['PRED_CAT'] = nn.predict_classes(Ucv)
    print(f'PRED_CAT in load_test_data before le inverse: {dt.PRED_CAT.head()}')
    dt['PRED_CAT'] = le.inverse_transform(dt['PRED_CAT'])
    print(f'PRED_CAT in load_test_data: {dt.PRED_CAT.head()}')
    dt.DATE = dt.DATE.astype(str)

    dx = dt.copy()
    dt.DR = dt.DR.astype(str)
    dt.CR = dt.CR.astype(str)
    dt = dt.T

    res = []
    for column in dt.columns:
        li = dt[column].tolist()
        res.append(li)

    res.insert(0, ['DATE','PARTICULARS', 'DR', 'CR', 'TYPE', 'PREDICTED CATEGORY'])

    return res, dx
