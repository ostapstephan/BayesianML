# python 3
# Ostap Voynarovskiy
# Prof. Keene
# Project 1

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.special

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

    print(sum(data))
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
        
        print( m, l, a, b, posterior(m,l,a,b), sep=',')

        mse_bad.append (mse(posterior(m,l,a,b),MLE ))
        mse_good.append(mse(posterior(m,l,ag,bg), MLE))

        # mse_bad.append(  np.mean(mse_good+ [mse(posterior(m,l,a,b),MLE )] ))
        # mse_good.append( np.mean(mse_good+ [ mse(posterior(m,l,ag,bg), MLE) ]))#ML_Estimate(data[:])))

    numSamples = np.linspace(1,N,N)
    plt.plot(numSamples,mse_bad)
    plt.plot(numSamples,mse_good)

    plt.xlabel('Num Observations', fontsize=10)
    ylab = plt.ylabel('MSE',labelpad=30, fontsize=10)
    ylab.set_rotation(0) 
           
    plt.title(f'Mean Squared Error Bernoulli')


    plt.tight_layout()
    plt.savefig(f'p2.pdf')
    plt.clf()

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
            plt.title(f'Peak at mu = {round(mu[np.argwhere(density==max(density))][0][0],3)}')
            loc+=1

    plt.tight_layout()
    plt.savefig(f'p1.pdf')




##### Gaussian Stuff ##### 

def gauss_ml_est(d):
    return np.mean(d)

    
def gaussianBayes(sig,sig_0, mu_ml, mu_0, N, ):
    Mu_n = (sig**2/(N*sig_0**2+sig**2))*mu_0 + ((N*sig**2)/(N*sig_0**2+sig**2))*mu_ml
    # 1/sig_n**2 = 1/sig_0**2 + N/sig**2
    return  Mu_n#, 1/sig_n

# data = np.random.normal(4,1,size=(100))


# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
# ax1.plot(x, y)
# ax1.set_title('Sharing x per column, y per row')
# ax2.scatter(x, y)
# ax3.scatter(x, 2 * y ** 2 - 1, color='r')
# ax4.plot(x, 2 * y ** 2 - 1, color='r')






# fig, axs = plt.subplots(1, 1)
# We can set the number of bins with the `bins` kwarg
# axs.hist(train, bins=100)

# plt.show()


if __name__ == '__main__':
    bernoulli()

