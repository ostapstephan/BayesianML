# python 3
# Ostap Voynarovskiy And Carena Toy
# Prof. Keene
# Project 3

import matplotlib.pyplot as plt 
import numpy as np
import scipy.io
from sklearn import metrics
from sklearn.model_selection import train_test_split

from scipy.io import arff 
import pandas as pd 

'''
Implement two different linear classifiers:
a Gaussian generative model
a logistic regression classifier. 
    you must implement the IRLS algorithm. 
    You will need to use a 3rd basis function to separate the circles data, 
    but you can operate directly in the feature space for the unimodal data.

Report your % correct classification (it should be very high for these simple datasets). 

Plot a ROC curve for both of your classifiers, Plot the decision boundary for both your classifiers for the unimodal data. (you can skip this for the circle one).


'''

def plot(x=np.empty([2]), t=np.empty([2]), title='', m = None, b = None):
    if t == np.empty([2]):
        data = scipy.io.loadmat("mlData.mat")
        x, y = data['circles'][0][0][0].T
        t = data['circles'][0][0][1].squeeze() # 400,1 => 400
    else:
        x, y = x[:,:2].T
    if m != None: 
        lsp = np.linspace(-3,3,1000)
        plt.plot(lsp,lsp*m+b)
    

    plt.scatter(x[t==0],y[t==0])
    plt.scatter(x[t==1],y[t==1])
    
    plt.title(title)
    plt.axis('square')
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y',labelpad=30, fontsize=10, rotation=0)
    plt.show()
    print(str(title).replace('\n','_').replace(' ','_'))
    plt.matplotlib.pyplot.savefig('plots/'+str(title).replace('\n','_').replace(' ','_')+'.pdf',)
    plt.clf()

def trainAndInferGGM(x_train, t_train, x_test):

    N = len(t_train) 
    N1 = sum(t_train) 
    N2 = sum(1-t_train) 

    mu1 = 1/N1 * t_train @ x_train 
    mu2 = 1/N2 * (1-t_train) @ x_train 

    S1 = 1/N1*(x_train-mu1).T@(x_train-mu1)
    S2 = 1/N2*(x_train-mu2).T@(x_train-mu2)
    S = (N1/N)*S1 + (N2/N)*S2
    
    c1 = scipy.stats.multivariate_normal(mu1, S) 
    c2 = scipy.stats.multivariate_normal(mu2, S) 

    p1 = c1.pdf(x_test)
    p2 = c2.pdf(x_test)
    
    pred = np.zeros(len(x_test))    
    pred[p1>p2]= 1

    mp = (mu2[0]+mu1[0]) / 2 ,(mu2[1]+mu1[1]) / 2
    m = (mu2[1]-mu1[1])/(mu2[0]-mu1[0])
    m = -1/m
    b = mp[1] - m*mp[0]
    
    return pred, m, b 

def sigmoid(x):
    return  (1 + np.exp(-x))**-1


def trainAndInferIRLS(x_train, t_train, x_test, circles=True, n_iter=2):

    if circles == True:
        phi = np.concatenate( (np.ones((x_train.shape[0],1)), x_train, np.expand_dims(x_train[:,0]**2+x_train[:,1]**2,1)), axis =1 )
    else:
        phi = np.concatenate( (np.ones((x_train.shape[0],1)), x_train ), axis =1 )

    # phi.shape = [n,m]
    # weight vector
    w = np.random.normal(0, 1, [ phi.shape[1] ]) 
    # w = [m]

    for i in range(n_iter):
        # y_pred = [N]
        y_pred = sigmoid( w @ phi.T)
        
        # R = [n,n]
        R = np.expand_dims(y_pred,1) @ np.expand_dims( (1-y_pred),1).T
        R = R*np.eye(R.shape[0],R.shape[1])

        # z = [n,1]
        z = phi@w - np.linalg.pinv(R)@ ( y_pred - t_train )

        # UPDATE WEIGHTS
        w = np.linalg.pinv(phi.T @ R @phi) @ phi.T @ R @ z

    # phi = np.concatenate((np.ones((x_test.shape[0],1)),x_test ),axis =1 )
    if circles == True:
        phi = np.concatenate((np.ones((x_test.shape[0],1)),x_test,np.expand_dims(x_test[:,0]**2+x_test[:,1]**2,1) ),axis =1 )
    else:
        phi = np.concatenate((np.ones((x_test.shape[0],1)),x_test ),axis =1 )


    y_pred = sigmoid( w@phi.T )
    t_hat= np.zeros( (y_pred.shape) )
    t_hat[y_pred > .5] = 1
    return t_hat, w 
    
def plotROC(x_test, t_hat, title= ''):
    fpr, tpr, thresholds = metrics.roc_curve(t_test,t_hat)
    roc_auc = metrics.auc(fpr, tpr)

    plt.clf()
    plt.title( title +' Receiver Operating Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc, linewidth=10)
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.matplotlib.pyplot.savefig('plots/'+str(title).replace('\n','_').replace(' ','_')+'.pdf',)
    plt.clf()

def accuracy(t_hat, t_test ):
    return  (len(t_hat)- sum(abs(t_hat-t_test))  )/len(t_hat) *100

if __name__ == '__main__':
    data = scipy.io.loadmat("mlData.mat")

    # Load the data for the unimodal 
    x = data['unimodal'][0][0][0]
    t = data['unimodal'][0][0][1].squeeze() 
    
    # Train the gaussian generative model for the unimodal 
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.33, random_state=42)
    t_hat,m,b= trainAndInferGGM(x_train, t_train, x_test )

    # Plot the results for the unimodal data
    plot(x_test, t_hat, f'Unimodal Data\n Accuracy: {accuracy(t_hat, t_test)}%',m,b) 
    plotROC(x_test, t_hat, 'Unimodal Data')    
      
    # Load the data for the circles
    x = data['circles'][0][0][0]
    t = data['circles'][0][0][1].squeeze() 
    
    # Train the gaussian generative model for the circles w/o third feat
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.33, random_state=42)
    t_hat,_,_ = trainAndInferGGM(x_train, t_train, x_test )

    # Plot the results for the circles w/0 third feat
    plot(x_test, t_hat, f'Circles Data Without a Third Feature\n Accuracy: {accuracy(t_hat, t_test)}%')
    plotROC(x_test, t_hat, 'Circles Data Without a Third Feature')

    # Train the gaussian generative model for the circles with third feat
    x = np.concatenate((x,np.expand_dims(x[:,0]**2+x[:,1]**2,1)),axis =1 )
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.33, random_state=42)
    t_hat,_,_ = trainAndInferGGM(x_train, t_train, x_test)

    # Plot the results for the circles with the third feat
    plot(x_test, t_hat, f'Circles Data With a Third Feature\n Accuracy: {accuracy(t_hat, t_test)}%' )
    plotROC(x_test, t_hat, 'Circles Data With a Third Feature')    
   
    # Load the clean data for the unimodal dataset 
    x = data['unimodal'][0][0][0]
    t = data['unimodal'][0][0][1].squeeze() 
 
    # Train the logistic regression model for the circles
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.33, random_state=42)
    t_hat, w = trainAndInferIRLS(x_train, t_train, x_test, False ,10 )

    # Plot the results for the circles with the third feat
    plot(x_test, t_hat, f'Unimodal Data IRLS\n Accuracy: {accuracy(t_hat, t_test)}%' )
    plotROC(x_test, t_hat, 'Unimodal Data IRLS')    
    
    # Load the clean data for the unimodal dataset 
    x = data['circles'][0][0][0]
    t = data['circles'][0][0][1].squeeze() 
 
    # Train the logistic regression model for the circles
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.33, random_state=42)
    t_hat,_ = trainAndInferIRLS(x_train, t_train, x_test, True, 10 )

    # Plot the results for the circles with the third feat
    plot(x_test, t_hat, f'Circles Data IRLS\n Accuracy: {accuracy(t_hat, t_test)}%' )
    plotROC(x_test, t_hat, 'Circles Data IRLS')    
    

    ##########################################################
    ##          banknote authentication dataset            ### 
    ##########################################################

    df = pd.read_csv('banknote_authentication.txt')         

    x = np.asarray(df[['a','b','c','d']])
    t = np.asarray(df['target'])

    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2, random_state=42)
    t_hat, _ = trainAndInferIRLS(x_train, t_train, x_test, True, 10)

    plotROC(x_test, t_hat, f'Banknote Authentication Dataset Accuracy: {accuracy(t_hat, t_test)}%')
    






















