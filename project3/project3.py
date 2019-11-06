import matplotlib.pyplot as plt 
import numpy as np
import scipy.io
from sklearn.metrics import roc_curve
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

def plot(x=np.empty([2]), t=np.empty([2]) ):
    if t == np.empty([2]):
        data = scipy.io.loadmat("mlData.mat")
        x, y = data['circles'][0][0][0].T
        t = data['circles'][0][0][1].squeeze() # 400,1 => 400
    else:
        x, y = x[:,:2].T


    plt.scatter(x[t==0],y[t==0])
    plt.scatter(x[t==1],y[t==1])
    plt.show()

def trainAndInfer(x_train, x_test, t_train):
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




if __name__ == '__main__':

    data = scipy.io.loadmat("mlData.mat")

    # plot()
    # x, y = data['circles'][0][0][0].T
    x = data['circles'][0][0][0]
    t = data['circles'][0][0][1].squeeze() # 400,1 => 400
    
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.33, random_state=42)
    t_hat = trainAndInfer(x_train, x_test, t_train )

    plot(x_test, t_hat)
 
    x = np.concatenate((x,np.expand_dims(x[:,0]**2+x[:,1]**2,1)),axis =1 )
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.33, random_state=42)
    t_hat = trainAndInfer(x_train, x_test, t_train )

    plot(x_test, t_hat)


    
    




