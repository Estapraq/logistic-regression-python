
#this program is to read the hand writting pictures (only zero and one)
#and apply logistic Regression. This is failed because we had 65
#features and 18 versions of 0's and 1's
import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
from PIL import Image #PYTHON IMAGE LIBRARY
from functools import reduce #for reduce to be defines
np.seterr(over='ignore')#ignore warning that is already taken care of
from math import log
from math import pow
import pandas as pd
import pylab # to show the graph
import scipy.optimize as opt #we need it for the fminunc function


def sigmoid(z):
    return 1/(1+ np.exp(-z))



    
def costfcn(theta, X, y):
    #theta=np.reshape(theta, (len(theta),1)) iit is not working
    #we have to convert to matrix
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    first= np.multiply(-y,np.log(sigmoid(X*theta.T))) 
    second = np.multiply((1-y),np.log(1-sigmoid(X*theta.T))) 
    return np.sum(first-second)/(len(X)) 


def grad(theta, X, y):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    error = sigmoid(X*theta.T) - y # difference between label and prediction
    features= int(theta.ravel().shape[1])
    print()
    grad= np.zeros(features)
    
    for i in range(features):
        temp = np.multiply(error, X[:,i]) # gradient vector
        grad[i]=np.sum(temp)/len(X)
    
    return grad    

    
def ReadImage():
    newArray=[]
    label=[]
    AveArray=[]
    images01=range(0,2)#we will read only 0, 1 pictures
    ImageVersions=range(1,10) #1-9 versions
    for eachPic01 in images01:
        for eachVersion in ImageVersions:
            label.append(eachPic01)
            imagePath='images/numbers/'+str(eachPic01)+'.'+str(eachVersion)+'.png'
            ImageRead= Image.open(imagePath)
            ImageArray=np.array(ImageRead)#Send to get the mean metho
            for row in ImageArray:
                for pix in row:
                    AverRBG=reduce(lambda x, y: x+y, pix[:3])/len(pix[:3])#we need a library fpor reduce
                    AveArray.append(AverRBG)
            
    newArray=np.reshape(AveArray,(18, 64))
    return newArray, label 
   
   
#add a unit vector to the matrix Array       
[Array, label]=ReadImage()
n,m= Array.shape
x0=np.ones((n,1))
Xnew= np.hstack((x0, Array))
#print(Xnew)
theta=np.zeros(65)
print(costfcn(theta, Xnew, label))

#fminnunc won't work unless you convert the arguments to matrices
X=np.array(Xnew)
y=np.array(label)
#minimize the theta using fminunc function
minimize_theta= opt.fmin_tnc(func=costfcn, x0=theta, fprime=grad, args=(X,y))

print('theta is \n', minimize_theta[0], costfcn(theta, X, y))


print(sigmoid(X*minimize_theta[0].T))

#Array.shape, theta.shape , label.shape #shape doesn't work because it ius list. should be changed to a matrix


#m=len(Array)
#n=len(Array[0])
#check the costfcn
#X=[[1,2], [1,2], [2,2]]
#th=[1,1]
#y=[1,1,0]
#print(costfcn(th, X, label))

#check sigmoid
#nums=np.arange(-10, 10, step=1)
#fig, ax=plt.subplots(figsize=(8,6))
#ax.plot(nums, sigmoid(nums), 'r')
#pylab.show()
