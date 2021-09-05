from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dropout, Dense
import pandas as pd
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time

start = time.time()

df = pd.read_csv('bsa_training_data.csv')
print(f'df.shape: {df.shape}')

X = df.PARTICULARS
df.Label.dropna(inplace=True)
y = df.Label.astype(str)
print(f'Total Unique Labels in Train: {len(df.Label.unique())}')

le = LabelEncoder()

y = le.fit_transform(y).astype(str)
le_name = r'Saved Models/target_label_encoder.sav'
pickle.dump(le, open(le_name, 'wb'))

print(f'type(X): {type(X)}, X.shape: {X.shape}')

cv = TfidfVectorizer(analyzer='word')
Xcv = cv.fit_transform(X).toarray()
cv_name = 'vectorizer.sav'
pickle.dump(cv, open(cv_name, 'wb'))
print(f'Xcv.shape: {Xcv.shape}, type(Xcv): {type(Xcv)}')



sm = SMOTE(random_state=22)

X = df.PARTICULARS
df.Label.dropna(inplace=True)
y = df.Label.astype(str)
print(f'Total Unique Labels in Train: {len(df.Label.unique())}')

le = LabelEncoder()

y = le.fit_transform(y).astype(str)
le_name = r'Saved Models/target_label_encoder.sav'
pickle.dump(le, open(le_name, 'wb'))
print(f'type(X): {type(X)}, X.shape: {X.shape}')

cv = TfidfVectorizer(analyzer='word')

Xcv = cv.fit_transform(X).toarray()
cv_name = r'Saved Models/vectorizer.sav'
pickle.dump(cv, open(cv_name, 'wb'))
print(f'Xcv.shape: {Xcv.shape}, type(Xcv): {type(Xcv)}')

from imblearn.over_sampling import SMOTE

sm2 = SMOTE(random_state=42, k_neighbors=1)
Xtrain, ytrain = sm2.fit_resample(Xcv, y.ravel())
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

y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
y_test = np.asarray(y_test).astype('float32').reshape((-1, 1))

print(f'type(X_test): {type(X_test)}, type(y_test): {type(y_test)}')

# ---- FULLY CONNECTED SEQUENTIAL ANN ------------- ]
solver = 'adam'
bs = 300
epochs = 50
patience = 5

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
nn.add(Dense(units=len(df.Label.unique()), activation='softmax'))

nn.compile(optimizer=solver, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
nn.fit(X_train, y_train, batch_size=bs, epochs=epochs, callbacks=[es], validation_data=(X_test, y_test))

from sklearn.metrics import  accuracy_score
y_pred = nn.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

accuracy = round(accuracy_score(y_test, y_pred), 3)
print(f'\nModel Accuracy: \033[1;32m{accuracy}\033[0m')

print(f'\nTotal Labels in Training: {len(df.Label.unique())}')
print(df.Label.unique())
print(f'\nTotal Unique Labels Predicted: {len(set(y_pred))}')
print(set(y_pred))
y_pred = le.inverse_transform(y_pred)
print(set(y_pred))

model_name = fr'Saved Models/model_ann_{int(accuracy*100)}.h5'
nn.save(model_name)

end = time.time()
print('\033[1;34m\nModel Training Summary\033[0m')
print(f'1. Model Saved to disk: ANN - \033[0;33m{model_name}\033[0m')
print(f'2. Model Saved to disk: Vectorizer - \033[0;33m{cv_name}\033[0m')
print(f'3. Model Saved to disk: Label Encoder Mapping - \033[0;33m{le_name}\033[0m')
print(f'4. Labels predicted: \033[0;32m{len(set(y_pred))} of total {len(df.Label.unique())}\033[0m ')
print(f'5. Time taken to Fit Neural Network: \033[0;34m{int(end - start)}s\033[0m')
print(f'6. Model Accuracy: \033[1;32m{accuracy}\033[0m')
print('\033[1;31m--END\033[0m')
