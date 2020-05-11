#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:04:50 2020

@author: ziyunchebn
"""

import sys
import pandas as pd
import numpy as np

def load_data(dataset):
  dataset.dropna(inplace=True)
  dataset['Open']=pd.to_numeric(dataset['Open'],downcast='float')
  dataset['High']=pd.to_numeric(dataset['High'],downcast='float')
  dataset['Low']=pd.to_numeric(dataset['Low'],downcast='float')
  #dataset['Weighted Price']=pd.to_numeric(dataset['Weighted Price'],downcast='float')

  num_steps=5
  dataset_multi = series_to_supervised(dataset, num_steps, 1)

  dataset_multi_np=dataset_multi.values
  #print(dataset_multi)
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  dataset_multi_np = sc.fit_transform(dataset_multi_np)
  
  nn_features=dataset_multi_np[:,0:dataset_multi_np.shape[1]-dataset.shape[1]]
  nn_output=dataset_multi_np[:,dataset_multi_np.shape[1]-2*dataset.shape[1]:]

  nn_labels=np.divide(nn_output[:,dataset.shape[1]+4],nn_output[:,4])>1
  #print(dataset.shape[1]+4)
  #nn_output[:,[29,4]]
  predict_features=dataset_multi_np[-1,dataset.shape[1]:]
  return nn_features,nn_labels,nn_output[:,[4,dataset.shape[1]+4]],predict_features

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def generate_rand_training(nn_features_scaled,nn_labels,num_training):
  #np.c_[nn_features_scaled, nn_labels]
  #select_array=np.random.choice(nn_features_scaled.shape[0], num_training, replace=False)
  concat_data=np.c_[nn_features_scaled, nn_labels]
  np.random.shuffle(concat_data)
  #print(concat_data)
  training, test = concat_data[:num_training,:], concat_data[num_training:,:]
  features_train = training[:,:nn_features_scaled.shape[1]]
  labels_train = training[:,-1]
  features_test =  test[:,:nn_features_scaled.shape[1]]
  labels_test = test[:,-1]
  #print(labels_train)
  #print(labels_test)
  return features_train,features_test,labels_train,labels_test

def generate_order_training(nn_features_scaled,nn_labels,num_training):
  features_train = nn_features_scaled[:num_training]
  features_test = nn_features_scaled[num_training:]
  labels_train = nn_labels[:num_training]
  labels_test = nn_labels[num_training:]
  #print(labels_train)
  #print(labels_test)
  return features_train,features_test,labels_train,labels_test

def fit_LSTM_multi(nn_features,nn_labels,num_steps,num_training):

  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  nn_features_scaled = sc.fit_transform(nn_features)
  feature_num=int(nn_features.shape[1]/num_steps)
  #nn_features_scaled=nn_features_scaled.reshape((nn_features_scaled.shape[0],num_steps,feature_num))
  #nn_labels=nn_labels.reshape((nn_labels.shape[0],1,nn_labels.shape[1]))
  features_train,features_test,labels_train,labels_test=generate_rand_training(nn_features_scaled,nn_labels,num_training)
  features_train=features_train.reshape((features_train.shape[0],num_steps,feature_num))
  features_test=features_test.reshape((features_test.shape[0],num_steps,feature_num))
  #features_train = nn_features_scaled[:num_training]
  #features_test = nn_features_scaled[num_training:]
  #labels_train = nn_labels[:num_training]
  #labels_test = nn_labels[num_training:]
  #Dependencies
  import keras
  from keras.models import Sequential
  from keras.layers import Dense
  from keras.callbacks import History
  from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, Flatten, Dropout
  # Neural network

  model = Sequential()
  model.add(LSTM(units=16, input_shape=(num_steps,feature_num)))
  model.add(Dropout(0.2))
  #model.add(LSTM(units=50, batch_input_shape=(8,1,23)))
  #model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))
  history = History()
  model.summary()
  model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

  from sklearn.model_selection import KFold
  results = []
  kf = KFold(n_splits=10)
  for train_idx, val_idx in kf.split(features_train, labels_train):
      X_train = features_train[train_idx]
      y_train = labels_train[train_idx]
      X_val = features_train[val_idx]
      y_val = labels_train[val_idx]
      hist = model.fit(X_train, y_train, batch_size = 8, epochs = 100, validation_data = (X_val, y_val), verbose = 1, callbacks=[history])
      results.append(hist.history)

  print(results)

  score = model.evaluate(features_test, labels_test,verbose=1)
  print(score)
  return hist

def fit_NN_multi(nn_features,nn_labels,num_training):
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  nn_features_scaled = sc.fit_transform(nn_features)
  #nn_labels=nn_labels.reshape((nn_labels.shape[0],1,nn_labels.shape[1]))
  features_train,features_test,labels_train,labels_test=generate_rand_training(nn_features_scaled,nn_labels,num_training)
  #print(features_train.shape)
  #print(features_test.shape)
  #print(labels_train.shape)
  #print(labels_test.shape)  
  import keras
  from keras.models import Sequential
  from keras.layers import Dense
  from keras.callbacks import History
  from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, Flatten, Dropout
  # Neural network

  model = Sequential()
  model.add(Dense(16, input_dim=features_train.shape[1], activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(1,activation='sigmoid'))
  history = History()
  
  print(model.summary())

  model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])  

  from sklearn.model_selection import KFold
  results = []
  kf = KFold(n_splits=2)
  for train_idx, val_idx in kf.split(features_train, labels_train):
      X_train = features_train[train_idx]
      y_train = labels_train[train_idx]
      X_val = features_train[val_idx]
      y_val = labels_train[val_idx]
      hist = model.fit(X_train, y_train, batch_size = 8, epochs = 100, validation_data = (X_val, y_val), verbose = 1, callbacks=[history])
      results.append(hist.history)

  #print(results)
  predictions = model.predict(features_test)
  score = model.evaluate(features_test, labels_test,verbose=1)
  print(score)

  prediction_dummy=predictions>0.5
  prediction_dummy=prediction_dummy.reshape(labels_test.shape)
  accuracy_score=np.equal(prediction_dummy,labels_test).sum()/prediction_dummy.shape[0]
  return accuracy_score,prediction_dummy,labels_test

def rebal_model(nn_features,nn_labels):
  results=[]
  rebal_days=2160
  num_training=int(rebal_days/3*2)
  step_size=nn_features.shape[0]//rebal_days
  print("num_training: ",num_training)
  for i in range(step_size-1):
    print("running step: ",i)
    nn_features_step=nn_features[(i*rebal_days):(i*rebal_days+rebal_days),:]
    nn_labels_step=nn_labels[(i*rebal_days):(i*rebal_days+rebal_days)]
    #score=fit_NN_multi(nn_features_step,nn_labels_step,num_training)
    accuracy_score,predictions,prediction_dummy,labels_test=fit_rf(nn_features_step,nn_labels_step,num_training)
    results.append(accuracy_score)
    
  return results


def rebal_model_consecutive(nn_features,nn_labels):
  results=[]
  rebal_window=501
  num_training=500
  #num_training=
  #step_size=nn_features.shape[0]//rebal_days
  print("num_training: ",num_training)
  for i in range(nn_features.shape[0]-rebal_window):
    print("running step: ",i)
    nn_features_step=nn_features[i:i+rebal_window,:]
    nn_labels_step=nn_labels[i:i+rebal_window]
    accuracy_score,predictions,prediction_dummy,labels_test=fit_rf(nn_features_step,nn_labels_step,num_training)
    #score=fit_NN_multi(nn_features_step,nn_labels_step,num_training)
    results.append(accuracy_score)
    #sys.exit()
    print_accuracy(results)
  return results


def fit_rf(nn_features,nn_labels,num_training,predict_features):
  '''
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  nn_features_scaled_rf = sc.fit_transform(nn_features)
  '''
  features_train,features_test,labels_train,labels_test=generate_rand_training(nn_features,nn_labels,num_training)
  #features_train,features_test,labels_train,labels_test=generate_order_training(nn_features,nn_labels,num_training)
  # Import the model we are using
  from sklearn.ensemble import RandomForestRegressor
  # Instantiate model with 1000 decision trees
  rf = RandomForestRegressor(n_estimators = 200, random_state = 42)
  # Train the model on training data
  rf.fit(features_train, labels_train);
  predictions = rf.predict(features_test)
  prediction_dummy=predictions>0.5
  #score=rf.score(predictions,labels_test)
  # Calculate the absolute errors
  #errors = abs(predictions - labels_test)
  # Print out the mean absolute error (mae)
  #print('Mean Absolute Error:', round(np.mean(errors), 2))
  accuracy_score=np.equal(prediction_dummy,labels_test).sum()/prediction_dummy.shape[0]
  prediction_label=rf.predict(predict_features.reshape(1,-1))>0.5
  print('prediction accuracy',accuracy_score)
  return accuracy_score,predictions,prediction_dummy,labels_test,prediction_label




