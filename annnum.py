import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
data=pd.read_csv('E:\\heart.csv')
cols=data.shape[1]
x=data.iloc[ : ,0:cols-1].values
y=data.iloc[ : ,cols-1:cols].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
model = Sequential()
model.add(Dense(6, activation='softmax'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
from sklearn.metrics import confusion_matrix, accuracy_score,plot_confusion_matrix,roc_curve,auc
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5,validation_split=0.2)
y_pred = model.predict(X_test)


y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test,y_pred)
accuracy_score(y_test, y_pred)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
y_pred = model.predict(X_test).ravel()
nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(y_test  , y_pred)
auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='Neural Network (auc = %0.3f)' % auc_keras)
