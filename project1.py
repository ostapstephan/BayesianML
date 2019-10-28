# python 3
# Ostap Voynarovskiy
# Prof. Keene
# Project 1

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.special

from scipy.stats import norm 


# Simulation 1 you should generate Bernoulli random variables and estimate the probability p. 
def ML_Estimate(x):
    # ML estimate eq. 2.7
    return np.mean(x)

def posterior(m,l,a,b):
    return (m+a)/(m+a+l+b)

# Conjugate Prior
def betaConjPriorUpdate(mu,m,l,a,b): 
    # update eq for the bayesian estimate eq. 2.18
    # m = num (x=1)
    # l = num (x=0) # Num(samples)-Num(x==1)
    # a = hyper parameter #can be interpreted as effective # of observations of x=1 in the prior
    # b = hyper parameter #can be interpreted as effective # of observations of x=0 in the prior( but they dont need to be integers
    e =  (np.random.gamma(m+a+l+b) / (np.random.gamma(m+a) * np.random.gamma(l+b)))
    f = mu**(m+a-1) 
    g = (1-mu)**(l+b-1)
    return  e*f*g

def mse(pred, actual):
    return (actual-pred)**2

    
def bernoulli():
    # ML Estimate 
    true_mu = .7
    N = 100
    data = np.zeros(N)
    data[np.random.random(N)<true_mu]=1

    # bad Guess
    plt.rcParams['figure.figsize'] = (20,12)

    a = 3
    b = 14

    ag = 3
    bg = 4
    mse_bad = []
    mse_good = []
    MLE = ML_Estimate(data[:])

    for i in range(N):
        m = sum(data[:i])
        l = i-m
        mse_bad.append (mse(posterior(m,l,a,b),MLE ))
        mse_good.append(mse(posterior(m,l,ag,bg), MLE))

    numSamples = np.linspace(1,N,N)
    plt.plot(numSamples,mse_bad)
    plt.plot(numSamples,mse_good)

    plt.xlabel('Num Observations', fontsize=10)
    ylab = plt.ylabel('MSE',labelpad=30, fontsize=10)
    ylab.set_rotation(0) 
           
    plt.title(f'Mean Squared Error Bernoulli')


    plt.tight_layout()
    plt.savefig(f'MSE_Bernoulli.pdf')
    plt.clf()

    f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row') 

    plt.rcParams['figure.figsize'] = (20,12)
    mu = np.linspace(0,1,num=1000)
    a = 7
    b = 3
    loc = 221
    for i in range(N+1):
        if i==0 or i==10 or i==66 or i==100: 
            m = data[:i].sum()
            l = i-m
            density = betaConjPriorUpdate(mu,m,l,a,b)
            plt.subplot(loc) 
            plt.plot(mu,density) 
            plt.xlabel('Mu', fontsize=10)
            ylab = plt.ylabel('P(mu|m,l,a,b)',labelpad=30, fontsize=10)
            ylab.set_rotation(0) 
            plt.title(f'N = {i} Peak at mu = {round(mu[np.argwhere(density==max(density))][0][0],3)}')
            loc+=1

    f.suptitle("Bernoulli Case")
    plt.tight_layout()
    f.subplots_adjust(top=0.88, wspace = 0.2, hspace = 0.2 )

    plt.savefig(f'PDF_Bernoulli.pdf')


##### Gaussian Stuff ##### 
def gauss_ml_est(d):
    return np.mean(d)

def gaussianBayes(mu_ml, sig, mu_0, sig_0, N):
    Mu_n = (sig**2/((N*sig_0**2)+sig**2))*mu_0 + ((N*sig_0**2)/(N*sig_0**2+sig**2))*mu_ml
    sig_n = (1/(sig_0**2) + N/sig**2)**-1
    sig_n = np.sqrt(sig_n)
    return  Mu_n, sig_n

def gaussian_plots():
    data = np.random.normal(4,1,size=(100))
    
    mu = 3 
    sigma = .5

    mu_0  = 5
    sig_0 = 1

    mu_space = np.linspace (0,10,1001)
    data = np.random.normal(mu,sigma,(100) )

    plt.rcParams['figure.figsize'] = (20,12)

    f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row') 
    
    loc = 221
    for i in [0,5,20,100]:
        plt.subplot(loc) 
        
        if i == 0:
            ml_Est = 0 
        else: 
            ml_Est = np.mean(data[:i])

        Mu_n, sig_n = gaussianBayes(ml_Est, sigma, mu_0, sig_0, i)
        prob = norm(Mu_n, sig_n)
        plt.plot(mu_space,prob.pdf(mu_space)) 

        plt.xlabel('Mu', fontsize=10)
        ylab = plt.ylabel('P(mu|x)',labelpad=30, fontsize=10)
        ylab.set_rotation(0) 
        plt.title(f"N = {i}")
        loc+=1

    f.suptitle("Gaussian Case")
    plt.tight_layout()

    f.subplots_adjust(top=0.92, wspace = 0.2, hspace = 0.2 )
    # plt.savefig(f'p1.pdf')
    plt.show()




if __name__ == '__main__':
    bernoulli()
    gaussian_plots()

