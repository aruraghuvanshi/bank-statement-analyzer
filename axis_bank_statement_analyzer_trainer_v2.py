import pandas as pd
import numpy as np
import warnings

import re
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dropout, Dense

from pythonml.datafunctions import qualityreport as qr
import pickle
from tensorflow.keras.models import load_model
import tabula





pd.options.display.max_rows = 500
pd.options.display.max_columns = 40
warnings.filterwarnings('ignore')
pd.options.display.max_rows = 500
pd.options.display.max_columns = 40
warnings.filterwarnings('ignore')


# from pdf_to_excel import pdf_to_excel
# c, xl = pdf_to_excel('AruAxis_test_march.pdf', 'AruAxis_test_march')

# print(f'type(c): {type(c)}')
# print(f'type(xl): {type(xl)}')
# print(f'xl: {xl}')
# org = f'{xl}.csv'

def ingest_test_pdf(testfile):

    pdf_file = testfile
    output_csv_name = f'{pdf_file[:-4]}.csv'

    dfs = tabula.read_pdf(pdf_file, pages='all', guess=False)
    tabula.convert_into(pdf_file, output_csv_name, output_format='csv', pages='all')

    df = pd.read_csv(output_csv_name)
    df.drop(['Chq No', 'Balance', 'Init.'], inplace=True, axis=1)
    df = df.dropna(how='all').reset_index(drop=True)
    df.drop(0, inplace=True)
    df.Debit.fillna(0, inplace=True)
    df.Credit.fillna(0, inplace=True)
    print(f'Pdf to Csv dataset shape: {df.shape}')

    print(f'Shape before dropna: {df.shape}')
    print(df.isna().sum())
    df.dropna(inplace=True)
    print()
    print(df.isna().sum())
    print(f'Shape after dropna: {df.shape}')

    df.to_csv(output_csv_name, index=False)

    return df, output_csv_name



def load_training_data(org, qrep=False):
    edt = org+'_edited.csv' 
    with open(org, 'r') as f:
        with open(edt, 'w') as f1:
            data_in = f.readlines()
            f1.writelines(data_in[9:-16])

    df = pd.read_csv(edt, na_values=[' '])
    print(f'Loaded Training file name: {org}')
    print(f'Loaded Training file shape: {df.shape}')
    df.dropna(how='all', inplace=True)
    if qrep:
        print(qr(df))
    
    return df


def pre_process_input_document(df):
    df.rename(columns={'Tran Date': 'DATE', 'Debit': 'DR',
                       'Credit': 'CR', 'Particulars': 'PARTICULARS'}, inplace=True)
    #     df.drop(0, inplace=True)
    #     df.drop(['CHQNO', 'SOL'], axis=1, inplace=True)
    df.DATE = pd.to_datetime(df.DATE)
    df.DR.fillna(0, inplace=True)
    df.CR.fillna(0, inplace=True)

    def remove_digits(org_string):
        pattern = r'[0-9]'
        mod_string = re.sub(pattern, '', str(org_string))
        return mod_string

    def preprocess_bankdata(df):

        df.PARTICULARS = df.PARTICULARS.str.split('/')
        df['TYPE'] = df.PARTICULARS.str[0]
        df.PARTICULARS = df.PARTICULARS.apply(', '.join)
        df.PARTICULARS = df.PARTICULARS.str.replace(',', ' ')
        df.PARTICULARS = df.PARTICULARS.str.replace('-', ' ')
        df.TYPE = df.TYPE.str.replace('-', ' ')
        df.PARTICULARS = df.PARTICULARS.str.replace(':', ' ')
        df.TYPE = df.TYPE.str.replace(':', ' ')
        df.TYPE = df.TYPE.str.replace('\n', ' ', regex=True)
        df.TYPE = df.TYPE.str.replace('\r', ' ', regex=True)
        df.PARTICULARS = df.PARTICULARS.str.replace('.', ' ')
        df.PARTICULARS = df.PARTICULARS.str.replace('_', ' ')
        df.PARTICULARS = df.PARTICULARS.str.replace('\n', ' ', regex=True)
        df.PARTICULARS = df.PARTICULARS.str.replace('/', ' ', regex=True)
        df.PARTICULARS = df.PARTICULARS.str.replace('\r', ' ', regex=True)
        df.PARTICULARS = df.PARTICULARS.str.replace('+', ' ')
        df.PARTICULARS = df.PARTICULARS.apply(lambda x: x.strip())

    def get_transaction_type(i):

        if i == 'ECOM PUR':
            return 'ECOM'
        elif i == 'VISA MERCH Refund':
            return 'RFND'
        elif 'BRN PYMT CARD' in i:
            return 'CC'
        elif 'Dr Card Charges ANNUAL 4691XXXXXXXX4257' in i or 'Consolidated Charges' in i:
            return 'BTAX'
        elif 'Service Tax' in i:
            return 'BTAX'
        elif 'BY CASH' in i or 'ATM CASH' in i:
            return 'CASH'
        elif 'PUR' in i:
            return 'PUR'
        elif '110010100193993' in i or 'Int.Pd to' in i:
            return 'INTP'
        elif 'GST' in i:
            return 'GST'
        elif 'CTF' in i:
            return 'CTF'
        elif 'REFUND' in i:
            return 'RFND'
        elif 'BRN CLG CHQ' in i:
            return 'CHQ'
        elif 'By Clg' in i:
            return 'CHQ'
        elif 'BHIM' in i or 'UPI' in i:
            return 'UPI'
        elif 'BRN TO' in i or 'BRN BY' in i or 'BY CASH' in i:
            return 'CASH'
        elif 'EXCESS FUEL' in i:
            return 'RFND'
        elif 'BRN NEFT' in i:
            return 'NEFT'
        elif 'RTGS' in i or 'TO' in i:
            return 'RTGS'
        elif 'BRN OW' in i:
            return 'REJ'
        elif 'BY' in i:
            return 'NEFT'
        elif 'INB IFT' in i or 'INB' in i:
            return 'INB'
        elif 'POS' in i:
            return 'POS'
        elif 'IMPS' in i or 'IMPS PA':
            return 'IMPS'
        else:
            return i

    df.PARTICULARS = df.PARTICULARS.apply(remove_digits)
    print(f'Preprocessed dataset shape: {df.shape}')
    preprocess_bankdata(df)
    df.TYPE = df.TYPE.apply(get_transaction_type)
    df.TYPE = df.TYPE.str.replace('Int.Pd    to     ', 'INTP', regex=True)
    print(f'\n{len(df.TYPE.unique())} Unique Transaction Types found: {df.TYPE.unique()}')

    return df


# df = pre_process_input_document(df)


def view_target_labels(df, head=5):

    df.CAT.value_counts().plot(kind='bar', figsize=(14,5), title='TARGET CLASS DATA BALANCE')
    plt.show()


        

def train_model(df, es_flag=False):

    print(f'df.shape: {df.shape}')
    start = time.time()
    
    X = df.PARTICULARS
    df.CAT.dropna(inplace=True)
    y = df.CAT.astype(str)
    print(f'Total Unique Labels in Train: {len(df.CAT.unique())}')
    df.CAT.unique(), X.shape

    le = LabelEncoder()
    
    y = le.fit_transform(y).astype(str)
    le_name = 'target_label_encoder.sav'
    pickle.dump(le, open(le_name, 'wb'))

    print(f'type(X): {type(X)}, X.shape: {X.shape}')

    cv = TfidfVectorizer(analyzer='word')
    
    Xcv = cv.fit_transform(X).toarray()
    cv_name = 'vectorizer.sav'
    pickle.dump(cv, open(cv_name, 'wb'))
    print(f'Xcv.shape: {Xcv.shape}, type(Xcv): {type(Xcv)}')
    
    from imblearn.over_sampling import SMOTE
    import seaborn as sns

    sm = SMOTE(random_state=22)

    X = df.PARTICULARS
    df.CAT.dropna(inplace=True)
    y = df.CAT.astype(str)
    print(f'Total Unique Labels in Train: {len(df.CAT.unique())}')
    df.CAT.unique(), X.shape

    le = LabelEncoder()

    y = le.fit_transform(y).astype(str)
    le_name = 'target_label_encoder.sav'
    pickle.dump(le, open(le_name, 'wb'))

    print(f'type(X): {type(X)}, X.shape: {X.shape}')


    cv = TfidfVectorizer(analyzer='word')

    Xcv = cv.fit_transform(X).toarray()
    cv_name = 'vectorizer.sav'
    pickle.dump(cv, open(cv_name, 'wb'))
    print(f'Xcv.shape: {Xcv.shape}, type(Xcv): {type(Xcv)}')

    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    Xtrain, ytrain = sm.fit_resample(Xcv, y.ravel())
    Xtrain = pd.DataFrame(Xtrain)
    ytrain = pd.DataFrame(ytrain)
        
    print(f'Xtrain: {Xtrain.shape}, ytrain: {ytrain.shape}')
    print(f'Post SMOTE: {Xcv.shape}, {y.shape}')
    
    X_train, X_test, y_train, y_test = train_test_split(Xtrain, ytrain, test_size=0.15, random_state=22)
    
    
    print(f'X_train.shape: {X_train.shape}')
    print(f'X_test.shape: {X_test.shape}')
    print(f'y_train.shape: {y_train.shape}')
    print(f'y_test.shape: {y_test.shape}')
    
    X_train = np.asarray(X_train).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)

    y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

    print(f'type(X_test): {type(X_test)}, type(y_test): {type(y_test)}')
    
    solver = 'adam'
    bs = 300
    epochs = 50
    patience = 5
    bestmodel = 'model_ann_mdlchkpt_best.h5'
    es = EarlyStopping(patience=patience)    

    nn = Sequential()
    nn.add(Dense(units=512, input_dim=X_train.shape[1], activation='relu'))
    nn.add(Dropout(0.25))
    nn.add(Dense(units=1024, activation='relu'))
    nn.add(Dropout(0.25))
    nn.add(Dense(units=2048, activation='relu'))
    nn.add(Dropout(0.25))
    nn.add(Dense(units=1024, activation='relu'))
    nn.add(Dropout(0.25))
    nn.add(Dense(units=512, activation='relu'))
    nn.add(Dropout(0.25))
    nn.add(Dense(units=len(df.CAT.unique()), activation='softmax'))

    nn.compile(optimizer=solver, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    if es_flag:
        mc = ModelCheckpoint(filepath=bestmodel, monitor='val_loss', save_best_only=True)    
        nn.fit(X_train, y_train, batch_size=bs, epochs=epochs, callbacks=[es, mc], validation_data=(X_test, y_test))   
    else:
        nn.fit(X_train, y_train, batch_size=bs, epochs=epochs, callbacks=[es], validation_data=(X_test, y_test))   

    from sklearn.metrics import  accuracy_score, classification_report
    y_pred = nn.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    accuracy = round(accuracy_score(y_test, y_pred), 3)   
    print(f'\nModel Accuracy: \033[1;32m{accuracy}\033[0m')

    print(f'\nTotal Labels in Training: {len(df.CAT.unique())}')    
    print(df.CAT.unique())
    print(f'\nTotal Unique Labels Predicted: {len(set(y_pred))}')             
    print(set(y_pred))    
    y_pred = le.inverse_transform(y_pred)
    print(set(y_pred))    

    temp1 = list(df.CAT.unique())
    temp2 = list(set(y_pred))
    diff= list(set(temp1) - set(temp2))
    print(f'\n {len(diff)} Labels not predicted: {diff}')

    model_name = f'model_ann_{int(accuracy*100)}.h5'
    nn.save(model_name)
    print(f'{model_name} saved to disk. ')

    if es_flag:
        print(f'\nModel Saved to disk: ANN Model \033[1;30m{bestmodel}\033[0m')
    else:
        print(f'\nModel Saved to disk: ANN Model \033[1;30m{model_name}\033[0m')
    
    print(f'Model Saved to disk: Vectorizer \033[1;30m{cv_name}\033[0m')
    print(f'Model Saved to disk: Label Encoder Mapping \033[1;30m{le_name}\033[0m')

    end = time.time()
    print(f'\nTime taken to Build Predictor model: {int(end - start)}s')

    return model_name, accuracy, cv_name, le_name



def load_test_data(testfile, model_name, cv_name, le_name, upper=7, lower=34):

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

    dt = pd.read_csv(testfile, na_values=[' '])
    print(f'\nLoaded Test file shape: {dt.shape}')
    dt = pre_process_input_document(dt)

    nn = load_model(model_name)
    cv = pickle.load(open(cv_name, 'rb'))
    le = pickle.load(open(le_name, 'rb'))

    U = dt.PARTICULARS
    Ucv = cv.transform(U).toarray()

    solver = 'adam'
    loss = 'sparse_categorical_crossentropy'
    nn.compile(optimizer=solver, loss=loss, metrics=['accuracy'])

    dt['PRED_CAT'] = nn.predict_classes(Ucv)
    dt['PRED_CAT'] = le.inverse_transform(dt['PRED_CAT'])
    dt.DATE = dt.DATE.astype(str)

    dt.drop(0, inplace=True)
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


# ------------------------- MAIN ------------------------------------           # Uncomment entire section for testing.
    
#
# testfile = 'AruAxis_test_march.csv'
# model_name = 'model_ann_98.h5'
# cv_name = 'vectorizer.sav'
# le_name = 'label_encoder.sav'

# df = load_training_data(testfile)
# df = pre_process_input_document(df)
# view_target_labels(df)
# model_name, accuracy, cv_name, le_name = train_model(df)                     # Keep commented for testing.


# res = load_test_data(testfile, model_name, cv_name, le_name)
# print(res)



