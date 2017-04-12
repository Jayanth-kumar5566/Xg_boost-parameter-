import numpy as np

def logregobj(pred, label):
    pred = 1.0 / (1.0 + np.exp(-pred))
    grad = pred - label
    hess = pred * (1.0-pred)
    return grad, hess

I_l=[(0,0),(0,0)] #a tuple with(pred,label)
I_r=[(0,1),(0,1)]
I=[(0,0),(0,0),(0,1),(0,1)]

def fun(x):
    pred=x[0]
    label=x[1]
    (g,h)=logregobj(pred,label)
    #return (g**2,h+1)
    return (g,h+1)
#---------------------------------------------------
Num=0
Den=0

for i in I_l:
    (num,den)=fun(i)
    Num+=num
    Den+=den

A = Num/Den
print A
#---------------------------------------------

Num=0
Den=0

for i in I_r:
    (num,den)=fun(i)
    Num+=num
    Den+=den

B = (Num)/Den
print B
#--------------------------------------------------

Num=0
Den=0

for i in I:
    (num,den)=fun(i)
    Num+=num
    Den+=den

C = (Num)/Den

gain=(0.5*(A+B-C))-1

print gain
