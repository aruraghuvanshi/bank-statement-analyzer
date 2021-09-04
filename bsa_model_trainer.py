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

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=22)

X = df.PARTICULARS
df.Label.dropna(inplace=True)
y = df.Label.astype(str)
print(f'Total Unique Labels in Train: {len(df.Label.unique())}')

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

sm = SMOTE(random_state=42, k_neighbors=1)
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

y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
y_test = np.asarray(y_test).astype('float32').reshape((-1, 1))

print(f'type(X_test): {type(X_test)}, type(y_test): {type(y_test)}')
# X = df.PARTICULARS
# # y = df.Label
#
# # le = LabelEncoder()
# # y = le.fit_transform(y).astype(str)
#
# cv_name = r'Saved Models\vectorizer.sav'
# le_name = r'Saved Models\target_label_encoder.sav'
#
# # pickle.dump(le, open(le_name, 'wb'))
#
# print(f'type(X): {type(X)}, X.shape: {X.shape}')
#
# cv = TfidfVectorizer(analyzer='word')
# Xcv = cv.fit_transform(X).toarray()
# pickle.dump(cv, open(cv_name, 'wb'))
# print(f'Xcv.shape: {Xcv.shape}, type(Xcv): {type(Xcv)}')
#
# sm = SMOTE(random_state=22)
# df.Label.dropna(inplace=True)
# y = df.Label.astype(str)
# print(f'Total Unique Labels in Train: {len(df.Label.unique())}')
#
# le = LabelEncoder()
# yle = le.fit_transform(y).astype(str)
# print(f'yle: {yle}')
# pickle.dump(le, open(le_name, 'wb'))
#
# X_train, X_test, y_train, y_test = train_test_split(Xcv, yle, test_size=0.15, random_state=22)
#
# print(f'X_train.shape: {X_train.shape}')
# print(f'X_test.shape: {X_test.shape}')
# print(f'y_train.shape: {y_train.shape}')
# print(f'y_test.shape: {y_test.shape}')
#
# X_train = np.asarray(X_train).astype(np.float32)
# X_test = np.asarray(X_test).astype(np.float32)
#
# y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
# y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
#
# print(f'type(X_test): {type(X_test)}, type(y_test): {type(y_test)}')
#
# ---- FULLY CONNECTED SEQUENTIAL ANN ------------- ]
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
nn.add(Dense(units=len(df.Label.unique()), activation='softmax'))

nn.compile(optimizer=solver, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

mc = ModelCheckpoint(filepath=bestmodel, monitor='val_loss', save_best_only=True)
nn.fit(X_train, y_train, batch_size=bs, epochs=epochs, callbacks=[es, mc], validation_data=(X_test, y_test))

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

model_name = fr'Saved Models\model_ann_{int(accuracy*100)}.h5'
nn.save(model_name)
print(f'Model Saved to disk: ANN \033[1;30m{model_name}\033[0m')
print(f'Model Saved to disk: Vectorizer \033[1;30m{cv_name}\033[0m')
print(f'Model Saved to disk: Label Encoder Mapping \033[1;30m{le_name}\033[0m')

end = time.time()
print(f'\nTime taken to Build Predictor model: \033[0;34m{int(end - start)}s\033[0m')
print('\n\033[1;31m--END\033[0m')