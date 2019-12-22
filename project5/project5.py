# python 3
# Ostap Voynarovskiy 
# Prof. Keene
# Project 5

import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import multivariate_normal 
from scipy.stats import norm 
from sklearn import metrics
from sklearn.model_selection import train_test_split

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
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

def confidence_ellipse(mean,cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    This function is an adapted version of the one found here 
    https://matplotlib.org/3.1.0/gallery/statistics/confidence_ellipse.html
            
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

class EM:
    def __init__(self,  N, seed, n_dims):
        np.random.seed(seed)
        self.N = N
        self.n_dims = n_dims

    def e(self,x,mu,sigma,pie):       
        gamma = np.zeros([mu.shape[0],x.shape[0]])  
        d=[]
        for k in range(mu.shape[0]):
            vec = norm.pdf(x,mu[k],sigma[k])
            d.append( pie[k]*vec )
        denominator=sum(d)
        for k in range(mu.shape[0]):
            vec = norm.pdf(x,mu[k],sigma[k])
            gamma[k] = pie[k] * vec /denominator

        # gamma's shape == [k,n] 
        # k = num classes
        # n = num samples
        return gamma.T
    
    def m(self,x,gamma):
        N = np.sum(gamma,0) # N should be shape [k]
        mu = (1/N * (x@gamma))
        sigma = np.ones([3])

        for k in range(mu.shape[0]):
            sum_gamma_and_mu=[]
            for n in range(x.shape[0]):
                sum_gamma_and_mu.append( gamma[n,k]* (x[n]-mu[k])* (x[n]-mu[k]).T)
            sum_gamma_and_mu = np.array(sum_gamma_and_mu)
            sigma[k] = np.sqrt( 1/N[k] * np.sum(sum_gamma_and_mu,0) )
        pie = N / np.sum(N)
        return mu, sigma, pie

    def e_2d(self,x,mu,sigma,pie):       
        gamma = np.zeros([mu.shape[0],x.shape[0]])  
        d=[]
        # calculate the responsibility params for each of the 3 clusters
        for k in range(mu.shape[0]):
            vec = multivariate_normal(mu[k],sigma[k])
            d.append( pie[k]*vec.pdf(x) ) #scalar*scalar
        denominator=sum(d)
        for k in range(mu.shape[0]):
            vec = multivariate_normal(mu[k],sigma[k])
            gamma[k] = (pie[k]*vec.pdf(x))/denominator
        return gamma.T

    def m_2d(self,x,gamma):
        N = np.sum(gamma, 0) # N should be the shape of the number of clusters 
        # Gamma is  [n,k] and so we should have k mu's
        mu=[[] for x in range(gamma.shape[1])]

        # calculate mu for each cluster
        for k in range(gamma.shape[1]):
            mu[k] = 1/N[k] * (gamma[:,k] @ x)

        mu = np.array(mu)
        sigma = np.ones([3,2,2])

        # calculate Cov mat and Pi
        for k in range(mu.shape[0]):
            sum_gamma_and_mu=[]
            for n in range(x.shape[0]):
                a = np.expand_dims((x[n]-mu[k]), 1)
                sum_gamma_and_mu.append( gamma[n,k] * a@a.T )
            
            sum_gamma_and_mu = np.array(sum_gamma_and_mu)
            sigma[k] = 1/N[k] *  np.sum(sum_gamma_and_mu,0)

        pie = N / np.sum(N)
        return mu, sigma, pie

    def train(self,x,k,n_iter):
        meanRange=(-10,10)
        sig = [] 
        if self.n_dims==1:
            #one dim case
            mu  = np.array([np.random.uniform(meanRange[0],meanRange[1],(self.n_dims)) for j  in range(k) ]).flatten()
            sig = np.array([np.random.uniform(0.1,meanRange[1],[self.n_dims for x in range(self.n_dims)]) for j in range(k)]).flatten()

            pie = np.array([len(x)/(len(x)*k) for j in range(k)]).flatten() 
            # its probably safe to assume there are equal number of samples from each source
            mu_,sig_,pie_= [],[],[]

            # put down your estimate for the random guess with no training 
            if n_iter[0] == 0:
                mu_,sig_,pie_= [mu],[sig],[pie]

            for i in range(n_iter[-1]):
                gamma = self.e(x,mu,sig,pie) 
                mu, sig, pie = self.m(x,gamma) 
                if i+1 in n_iter:
                    mu_.append(mu)
                    sig_.append(sig)
                    pie_.append(pie)
        else:
            mu  = np.array([np.random.uniform(meanRange[0],meanRange[1],(self.n_dims)) for j  in range(k) ])
            size = [self.n_dims for x in range(self.n_dims)]
            s = np.array([np.random.uniform(0.1,meanRange[1],size) for j in range(k)])
            sig = [ x @ x.T for x in s ]

            pie = np.array([len(x)/(len(x)*k) for j in range(k)]).flatten() 
            # its probably safe to assume there are equal number of samples from each source as a preliminary estimate

            mu_,sig_,pie_= [],[],[]

            # put down your est for the initial guess with no training 
            if n_iter[0] == 0:
                mu_,sig_,pie_= [mu],[sig],[pie]
            
            for i in range(n_iter[-1]):
                gamma = self.e_2d(x,mu,sig,pie) 
                mu, sig, pie = self.m_2d(x,gamma) 
                if i+1 in n_iter:
                    mu_.append(mu)
                    sig_.append(sig)
                    pie_.append(pie)
                        
        return np.array(mu_), np.array(sig_), np.array(pie_)
        
    def gen_data(self, meanRange=(-10,10), ndims=1):
        ndims = int(ndims)
        if ndims == 1:
            # 1d case 
            mu0,mu1,mu2= -9,0, 9
            sig0,sig1,sig2  = .5,1,2

            n0 = np.random.randint(self.N/6,self.N/2)
            n1 = np.random.randint(self.N/6,self.N/2)
            n2 = self.N- n0 - n1

            data_0 = np.random.normal(mu0,np.sqrt(sig0),n0)
            data_1 = np.random.normal(mu1,np.sqrt(sig1),n1)
            data_2 = np.random.normal(mu2,np.sqrt(sig2),n2)

            x = np.concatenate([data_0,data_1,data_2]) 
            mu  = np.array([mu0,mu1,mu2])
            sig = np.array([sig0,sig1,sig2])
            
            pie = np.array([n0,n1,n2])/self.N

        elif ndims ==2:
            mu0 = np.random.uniform(meanRange[0],meanRange[1],(ndims))
            mu1 = np.random.uniform(meanRange[0],meanRange[1],(ndims))
            mu2 = np.random.uniform(meanRange[0],meanRange[1],(ndims))
                
            sig0 = np.random.uniform(meanRange[0]/3,meanRange[1]/3,[ndims for x in range(ndims)]) 
            sig1 = np.random.uniform(meanRange[0]/3,meanRange[1]/3,[ndims for x in range(ndims)])
            sig2 = np.random.uniform(meanRange[0]/3,meanRange[1]/3,[ndims for x in range(ndims)]) 

            # I need to make sure the cov is positive semidefinite   
            sig0 = sig0@sig0.T
            sig1 = sig1@sig1.T
            sig2 = sig2@sig2.T

            n0 = np.random.randint(self.N/6,self.N/2)
            n1 = np.random.randint(self.N/6,self.N/2)
            n2 = self.N - n0 - n1
            
            data_0 = np.random.multivariate_normal(mu0,sig0,(n0))
            data_1 = np.random.multivariate_normal(mu1,sig1,(n1))
            data_2 = np.random.multivariate_normal(mu2,sig2,(n2))

            x   = np.concatenate([data_0,data_1,data_2])
            mu  = np.array([mu0,mu1,mu2])
            sig = np.array([sig0,sig1,sig2])
            pie = np.array([n0,n1,n2])/(np.sum([n0,n1,n2]))

        else:
            raise ValueError(f'Number of dimentions must be greater than or equal to 1, you passed: {ndims}')
        return x, mu, sig, pie
    
def One_D():
    seed = 8
    N = 30000 # num data points 
    ndims=1
    em_alg= EM( N,seed ,ndims)
    data, mu, sig, pie =  em_alg.gen_data((-10,10), ndims=ndims)

    bins = 40
    bounds = (-10,10)

    k = 3
    N_train_steps = [ 0,1,3,15 ]
    mu_, sig_, pie_ = em_alg.train(data,k,N_train_steps)

    # Draw the actual distribution
    x = np.linspace(-10,10,1000)
    y = ((bounds[1]-bounds[0])/bins) * (pie[0]*multivariate_normal.pdf(x, mu[0], sig[0] ) +\
            pie[1]*multivariate_normal.pdf(x, mu[1], sig[1] )  +\
            pie[2]*multivariate_normal.pdf(x,mu[2],sig[2]))
    
    plt.rcParams['figure.figsize'] = (16,9) # make the plot bigger
    plt.show ()
    for loc in range(4) :
        print('Number of training steps:', N_train_steps[loc])
        print('pie',pie, pie_[loc])
        print('mu', mu, mu_[loc])
        print('sig',sig, sig_[loc])
        # plot the distribution based on the parameters we estimated
        y_ = ((bounds[1]-bounds[0])/bins) * (pie_[loc][0]*multivariate_normal.pdf(x, mu_[loc][0], sig_[loc][0] ) +\
                pie_[loc][1]*multivariate_normal.pdf(x, mu_[loc][1], sig_[loc][1] ) +\
                pie_[loc][2]*multivariate_normal.pdf(x, mu_[loc][2],sig_[loc][2]))

        plt.subplot(2,2,loc+1)
        plt.title(f'{N_train_steps[loc]} Training Steps')

        plt.hist(data,bins,bounds)  # plot data hist
        plt.plot(x, y*N)            # plot true dist
        plt.plot(x, y_*N)           # plot prediction @ n steps

    plt.matplotlib.pyplot.savefig(f'plots/GMM_1D_Case.png',dpi=800)
    plt.show()
    plt.clf()
       
def Two_D():
    seed = 24
    N = 3000  # num data points 
    ndims = 2
    em_alg = EM( N, seed ,ndims )
    data, mu, sig, pie =  em_alg.gen_data((-10,10), ndims=ndims)

    plt.rcParams['figure.figsize'] = (16,9) # make the plot bigger
    bins = 40
    bounds = (-10,10)

    # calculate predictions and output results at each of N_train_steps
    k = 3
    N_train_steps = [ 0,1,10,30 ]
    mu_, sig_, pie_ = em_alg.train(data,k,N_train_steps)

    print('stuff',mu_.shape, sig_.shape)
    plt.rcParams['figure.figsize'] = (16,9) # make the plot bigger
    for loc in range(4) :
                
        ax = plt.subplot(2,2,loc+1)
        ax.set_title(f'2D case {N_train_steps[loc]} Training Steps')
        x = data[:,0]
        y = data[:,1]
        plt.scatter(x,y,linewidths=2)

        # PLOT THE TRUE ELLIPSES
        for c in range(3):
            # Plot the true ellipse 
            confidence_ellipse(mu[c], sig[c], ax, n_std=1, edgecolor='blue')

        # Draw the predicted confidence ellipses   
        for c in range(3):
            # Plot the true ellipse colors 
            confidence_ellipse(mu_[loc][c], sig_[loc][c], ax, n_std=1, edgecolor='orange')
        
        plt.gca().set_aspect('equal', adjustable='box')
        ax = plt.gca()
        plt.xlim(-25, 25)
        plt.ylim(-15.5, 15.5)

        print('Number of training steps:', N_train_steps[loc])
        print('pie', pie_[loc])
        print('mu', mu_[loc])
        print('sig', sig_[loc])
        
    plt.matplotlib.pyplot.savefig(f'plots/GMM_2D_Case.png',dpi=800)
    plt.show()
    plt.clf()


if __name__ == '__main__':
    
    ########## 1D 3 Cluster Case ###########
    One_D()

    ########## 2D 3 Cluster Case ###########
    Two_D()
    

