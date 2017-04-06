#!/usr/bin/python
import numpy as np
import scipy.sparse
import pickle
import xgboost as xgb
#--------Read a csv file and obtainig a DMatrix------------------
import pandas
df=pandas.read_csv("pima-data-orig.csv")
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
train=train.as_matrix()
test=test.as_matrix()
train_data=train[:,:-1]
test_data=test[:,:-1]
train_label=train[:,-1]
test_label=test[:,-1]
dtrain = xgb.DMatrix(train_data,label=train_label)
dtest = xgb.DMatrix(test_data,label=test_label)

# specify parameters via map, definition are same as c++ version
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }

# specify validations set to watch performance
watchlist  = [(dtest,'eval'), (dtrain,'train')]
results=dict()
num_round = 100
bst = xgb.train(param, dtrain, num_round, watchlist,evals_result=results)

# this is prediction
preds = bst.predict(dtest)
labels = dtest.get_label()
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))

#-------------To set up evaluvation of models----------------------------------

