import matplotlib.pyplot as plt 
import numpy as np
import scipy.io
from sklearn import metrics
from sklearn.model_selection import train_test_split

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

def plot(x=np.empty([2]), t=np.empty([2]), title=''):
    if t == np.empty([2]):
        data = scipy.io.loadmat("mlData.mat")
        x, y = data['circles'][0][0][0].T
        t = data['circles'][0][0][1].squeeze() # 400,1 => 400
    else:
        x, y = x[:,:2].T

    plt.scatter(x[t==0],y[t==0])
    plt.scatter(x[t==1],y[t==1])
    
    plt.title(title)
    plt.axis('square')
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y',labelpad=30, fontsize=10, rotation=0)
    plt.show()

def trainAndInferGGM(x_train, t_train, x_test):
    N = len(t_train) 
    N1 = sum(t_train) 
    N2 = sum(1-t_train) 

    mu1 = 1/N1 * t_train @ x_train 
    mu2 = 1/N2 * (1-t_train) @ x_train 
    print(mu1, mu2, N1, N2)

    S1 = 1/N1*(x_train-mu1).T@(x_train-mu1)
    S2 = 1/N2*(x_train-mu2).T@(x_train-mu2)
    S = (N1/N)*S1 + (N2/N)*S2
    
    c1 = scipy.stats.multivariate_normal(mu1, S) 
    c2 = scipy.stats.multivariate_normal(mu2, S) 

    p1 = c1.pdf(x_test)
    p2 = c2.pdf(x_test)
    
    pred = np.zeros(len(x_test))    
    pred[p1>p2]= 1

    return pred

def trainAndInferIRLS(x_train, t_train, x_test, n_iter):
    # phi.shape = [n,m]
    phi =  x_train
    yhat = 



    
def plotROC(x_test, t_hat,title= ''):
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



if __name__ == '__main__':

    data = scipy.io.loadmat("mlData.mat")

    # Load the data for the unimodal 
    x = data['unimodal'][0][0][0]
    t = data['unimodal'][0][0][1].squeeze() 
    
    # Train the gaussian generative model for the unimodal 
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.33, random_state=42)
    t_hat = trainAndInferGGM(x_train, t_train, x_test )

    # Plot the results for the unimodal data
    plot(x_test, t_hat, 'Unimodal Data')    
    plotROC(x_test, t_hat, 'Unimodal Data')    
    
    # Load the data for the circles
    x = data['circles'][0][0][0]
    t = data['circles'][0][0][1].squeeze() 
    
    # Train the gaussian generative model for the circles w/o third feat
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.33, random_state=42)
    t_hat = trainAndInferGGM(x_train, t_train, x_test )

    # Plot the results for the circles w/0 third feat
    plot(x_test, t_hat,'Circles Data Without a Third Feature')
    plotROC(x_test, t_hat, 'Circles Data Without a Third Feature')

    # Train the gaussian generative model for the circles with third feat
    x = np.concatenate((x,np.expand_dims(x[:,0]**2+x[:,1]**2,1)),axis =1 )
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.33, random_state=42)
    t_hat = trainAndInferGGM(x_train, t_train, x_test)

    # Plot the results for the circles with the third feat
    plot(x_test, t_hat, 'Circles Data With a Third Feature' )
    plotROC(x_test, t_hat, 'Circles Data With a Third Feature')    
    

    # Load the clean data for the unimodal dataset 
    x = data['circles'][0][0][0]
    t = data['circles'][0][0][1].squeeze() 
 
    # Train the logistic regression model for the circles
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.33, random_state=42)
    t_hat = trainAndInferIRLS(x_train, x_test, t_train )


















