# python 3
# Ostap Voynarovskiy 
# Prof. Keene
# Project 5

import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import metrics
from sklearn.model_selection import train_test_split

'''
Implement EM on a Gaussian mixture model in 1 and 2 dimensions with K = 3.
The choice of means, covariance and pi is up to you. 
The algorithm is laid out explicitly in equations 9.23-9.28.
 
For the 1-d case, produce a plot that shows a histogram of your generated observations, 
and overlay on that histogram the pdf you found. 
Plot this at algorithm init, and a couple other times as the algorithm converges. 
If you feel ambitious make a movie. If you want to see the algorithm break, 
artificially introduce a data point that exactly equals one of the means of the distribution.
 
For 2-D, create a plot similar to 9.8, but with K = 3. If you want to get fancy, make it a 3-D plot.
'''



class EM:
    def __init__(self, seed, N ):
        np.random.seed(seed)
        self.N = N
        
        
                
    def e(self,mu,sigma,pie,):       
        # gamma_z = pie*
        pass
        

    def m(self):
        mu = []
        sig = []
        pie = []
        # pass
    
    def gen_data(self, meanRange=(-10,10), ndims=1):
        if ndims == 1:
            # 1d case 
            mu0 = np.random.uniform(meanRange[0],meanRange[1],(1))
            mu1 = np.random.uniform(meanRange[0],meanRange[1],(1))
            mu2 = np.random.uniform(meanRange[0],meanRange[1],(1))

            sig0 = np.random.uniform(.1,meanRange[1],(1))
            sig1 = np.random.uniform(.1,meanRange[1],(1))
            sig2 = np.random.uniform(.1,meanRange[1],(1))

            n0 = np.random.randint(self.N/6,self.N/2)
            n1 = np.random.randint(self.N/6,self.N/2)
            n2 = self.N- n0 - n1

            print(n0,n1,n2)  
            print(sig0,sig1,sig2)  
            print(mu0,mu1,mu2)  

            data_0 = np.random.normal(mu0,sig0,n0)
            data_1 = np.random.normal(mu1,sig1,n1)
            data_2 = np.random.normal(mu2,sig2,n2)

            x = np.concatenate([data_0,data_1,data_2])
            mu  = np.array([mu0,mu1,mu2])
            sig = np.array([sig0,sig1,sig2])
            pie = np.array([n0,n1,n2])

        elif ndims > 1:
            # ndims > 1 case 
            mu0 = np.random.uniform(meanRange[0],meanRange[1],(ndims))
            mu1 = np.random.uniform(meanRange[0],meanRange[1],(ndims))
            mu2 = np.random.uniform(meanRange[0],meanRange[1],(ndims))

            sig0 = np.random.uniform(meanRange[0],meanRange[1],[ndims for x in range(ndims)])
            sig1 = np.random.uniform(meanRange[0],meanRange[1],[ndims for x in range(ndims)])
            sig2 = np.random.uniform(meanRange[0],meanRange[1],[ndims for x in range(ndims)])
            
            n0 = np.random.randint(self.N/6,self.N/2)
            n1 = np.random.randint(self.N/6,self.N/2)
            n2 = self.N - n0 - n1

            data_0 = np.random.multivariate_normal(mu0,sig0,(n0))
            data_1 = np.random.multivariate_normal(mu1,sig1,(n1))
            data_2 = np.random.multivariate_normal(mu2,sig2,(n2))

            x   = np.concatenate([data_0,data_1,data_2])
            mu  = np.concatenate([mu0,mu1,mu2])
            sig = np.concatenate([sig0,sig1,sig2])
            pie = np.concatenate([n0,n1,n2])

        else:
            raise ValueError(f'Number of dimentions must be greater than or equal to 1, you passed: {ndims}')

        return x, mu, sig, pie
    




    
if __name__ == '__main__':
    seed = 7
    N = 3000 # num data points per cat
    em_alg= EM(seed, N)
    data, mu, sig, pie =  em_alg.gen_data((-10,10), ndims=1)
    print(data.shape)
    plt.hist(data,40,(-10,10)) 

    # draw true curve
    x = np.linspace(-10,10,1000)
    y =(pie[0]*multivariate_normal.pdf(x, mu[0], sig[0] ) + pie[1]*multivariate_normal.pdf(x, mu[1], sig[1] ) + pie[2]*multivariate_normal.pdf(x, mu[2], sig[2] ))

    plt.plot(x,y)

    plt.show()







