#!/usr/bin/python
import numpy as np
import scipy.sparse
import pickle
import xgboost as xgb
import pylab
import time
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

#param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic','eval_metric':'logloss','eval_metric':'error' }

param = [('max_depth', 2), ('objective', 'binary:logistic'), ('eval_metric', 'logloss'), ('eval_metric', 'error'),('eval_metric','auc'),('missing',0)]
 
# specify validations set to watch performance
watchlist  = [(dtest,'eval'), (dtrain,'train')]
results=dict()
num_round = 50
bst = xgb.train(param, dtrain, num_round, watchlist,evals_result=results)

# this is prediction
preds = bst.predict(dtest)
labels = dtest.get_label()
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))

#-------------To set up evaluvation of models----------------------------------

#Print model report:
print "\nModel Report"
print "\nTrain-Accuracy"
ter=results['train']['error'][-1]
tauc=results['train']['auc'][-1]
print (1-float(ter))*100
print "AUC Score(Train)",tauc
print "\nTest-Accuracy"
er=results['eval']['error'][-1]
auc=results['eval']['auc'][-1]
print (1-float(er))*100
print "AUC Score(Test)",auc

#-------------------Graphical Plotting----------------------------
'''pylab.ion()
x=0
for i in results['train']['error']:
    pylab.plot(x,float(i),marker='o', linestyle='--')
    x+=1
    pylab.draw()
    time.sleep(0.2)'''

pylab.plot(range(len(results['train']['error'])),results['train']['error'])
pylab.show()

pylab.plot(range(len(results['train']['auc'])),results['train']['auc'])
pylab.show()

#---------------Feature Importance--------------------------


