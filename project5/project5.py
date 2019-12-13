# python 3
# Ostap Voynarovskiy 
# Prof. Keene
# Project 5

import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import multivariate_normal as mv_norm
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

    def e(self,mu,sigma,pie):       
        gamma = np.zeros(mu.shape[0])  
        d=[]
        for k in range(mu.shape[0]):
            d.append(pie[k]*mv_norm(x,mu[k],sig[k])))
        denominator=sum(d)
        for k in range(mu.shape[0]):
            gamma[k]= pie[k]*mv_norm(x,mu[k],sig[k])/denominator

        # gamma's shape == [n,k]
        return gamma


    def m(self,):
        #TODO figure this one out
        mu = 1/N
        sig = []
        pie = []
        # pass
    
    def gen_data(self, meanRange=(-10,10), ndims=1):
        ndims = int(ndims)
        if ndims == 1:
            # 1d case 
            mu0 = np.random.uniform(meanRange[0],meanRange[1],(1))
            mu1 = np.random.uniform(meanRange[0],meanRange[1],(1))
            mu2 = np.random.uniform(meanRange[0],meanRange[1],(1))
            # mu0,mu1,mu2=-7,0,7

            sig0 = np.random.uniform(.1,meanRange[1],(1))
            sig1 = np.random.uniform(.1,meanRange[1],(1))
            sig2 = np.random.uniform(.1,meanRange[1],(1))

            n0 = np.random.randint(self.N/6,self.N/2)
            n1 = np.random.randint(self.N/6,self.N/2)
            n2 = self.N- n0 - n1

            print('Pie',n0,n1,n2)  
            print('mu',mu0,mu1,mu2)  
            print('sig',sig0,sig1,sig2)  

            data_0 = np.random.normal(mu0,np.sqrt(sig0),n0)
            data_1 = np.random.normal(mu1,np.sqrt(sig1),n1)
            data_2 = np.random.normal(mu2,np.sqrt(sig2),n2)

            x = np.concatenate([data_0,data_1,data_2]) 
            mu  = np.array([mu0,mu1,mu2])
            sig = np.array([sig0,sig1,sig2])
            pie = np.array([n0,n1,n2])

        elif ndims > 1:
            print("n>1")
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
            
            print('Pie', n0,n1,n2)  
            print('mu',  mu0,mu1,mu2)  
            print('sig', sig0,sig1,sig2)  

            data_0 = np.random.multivariate_normal(mu0,sig0,(n0))
            data_1 = np.random.multivariate_normal(mu1,sig1,(n1))
            data_2 = np.random.multivariate_normal(mu2,sig2,(n2))

            '''
            # for data in [data_0,data_1,data_2]:
                # x = data[:,0]
                # y = data[:,1]
                # plt.scatter(x,y)
            # plt.show()
            '''
            x   = np.concatenate([data_0,data_1,data_2])
            mu  = np.array([mu0,mu1,mu2])
            sig = np.array([sig0,sig1,sig2])
            pie = np.array([n0,n1,n2])
            
        else:
            raise ValueError(f'Number of dimentions must be greater than or equal to 1, you passed: {ndims}')

        return x, mu, sig, pie
   
if __name__ == '__main__':
    seed = 7
    N = 30000# num data points per cat
    em_alg= EM(seed, N)
    data, mu, sig, pie =  em_alg.gen_data((-10,10), ndims=1)
    print(data.shape)
    bins = 40
    bounds = (-10,10)
    plt.hist(data,bins,bounds) 

    # draw true curve
    x = np.linspace(-10,10,1000)
    y = (bounds[1]-bounds[0])/bins* (pie[0]*multivariate_normal.pdf(x, mu[0], sig[0] ) + pie[1]*multivariate_normal.pdf(x, mu[1], sig[1] ) + pie[2]*multivariate_normal.pdf(x,mu[2],sig[2]))

    plt.plot(x,y)
    
    plt.show()
    plt.clf()
    
    
    ########################################
    ########## 2D 3 Cluster Case ###########
    ########################################
    seed = 7
    N = 3000
    
    em_alg= EM(seed, N)
    data, mu, sig, pie =  em_alg.gen_data((-10,10), ndims=2)
    
    # bins = 40
    # bounds = (-10,10)
    
    # draw true curve
    # x = np.linspace(-10,10,1000)
    # y = (bounds[1]-bounds[0])/bins* (pie[0]*multivariate_normal.pdf(x, mu[0], sig[0] ) + pie[1]*multivariate_normal.pdf(x, mu[1], sig[1] ) + pie[2]*multivariate_normal.pdf(x, mu[2], sig[2] ))

    # plt.plot(x,y)

    # plt.show()


