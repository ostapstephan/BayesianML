# python 3
# Ostap Voynarovskiy 
# Prof. Keene
# Project 5

import matplotlib.pyplot as plt 
import numpy as np
import scipy.misc
import scipy.special
import scipy.stats 

from scipy.stats import multivariate_normal 
from scipy.stats import norm 
from sklearn import metrics
from sklearn.model_selection import train_test_split




def part1():
    '''        
    First part of the sampling methods project is to perform rejection sampling  
    to draw samples from a Gaussian mixture model. 

    Use the same mixture model from the EM project, and I suggest using regular normal 
    RV for the proposal distribution. Draw samples using this method and plot a 
    histogram of the generated samples against your GMM.
    '''

    def calculate_p(val):
        # Take in an array of numbers we are trying to eval the function at  
        mu0, mu1, mu2= -5, 0,7
        sig0,sig1,sig2  = 2, 1.5, 1.0
        p =  norm.pdf(val,mu0,sig0) + norm.pdf(val,mu1,sig1) +norm.pdf(val,mu2,sig2)
        return p

    # Rejection Sampling
    N_gaussian_samp =1000 
    t = np.linspace(-15,15, N_gaussian_samp)
    
    # p is what we are trying to sample
    p = calculate_p(t)

    # make an envelope gaussian 
    mu, sigma = 0,7
    q = norm.pdf(t,mu,sigma) 

    # calculate scaling factor 
    k = np.max(p/q)
    q *= k

    # sample it 
    num_sample_loc  = 10000 # This is larger to reduce noise on histogram 
    num_samp_height = 1000  # making this large will reduce the noise on the rejection sampling at a point
    sample_loc      = np.sort(np.random.normal( loc=mu, scale=sigma, size=(num_sample_loc))) 
     
    # calc p and q at these 2 points
    sample_height = norm.pdf(sample_loc,mu,sigma)*k 
    thresh = calculate_p(sample_loc)

    q_samples = np.array([np.random.uniform(0, high, size = num_samp_height) for high in sample_height]).T
    accepted_percent  = np.sum(thresh>q_samples,0)/num_samp_height 
    function_estimate = sample_height*accepted_percent
    
    plt.title('Estimation of the Distribution')
    plt.plot(sample_loc,function_estimate)
    plt.show()
    plt.clf()

    p_samp =  np.concatenate([ np.array(np.ones(v)*sample_loc[c]) for c,v in enumerate( np.sum(thresh>q_samples,0) )] ).flatten()

    plt.rcParams['figure.figsize'] = (16,9) # make the plot bigger
    plt.hist(p_samp, 75, (-15,15) ,density=True)

    plt.plot(t,p/3)
    plt.plot(t,q/3)

    # plt.ylim(bottom=-1,top= 1) 
    plt.xlim(left =-15,right=15)

    plt.title('Rejection Sampling')
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y',labelpad=10, fontsize=10, rotation=0)
    plt.legend(('Function we are trying to sample',f'Envelope (k = {k})','Histogram of the sampled distribution',))
    plt.savefig('plots/Rejection_Sampling_Envelope.png')
    
    plt.show()









'''        
Part two of the sampling methods mini-project is to re-do the first part of your 
linear regression project using MCMC to find an estimate for the weights.

Reuse your project 2 to generate the same training data. Just do this for 25 training 
samples

Use Equation 3.10 as the likelihood function, to be used with the training samples 
you generated. You may select any distribution you want for the prior on the weights, 
and recall that the posterior density on the weights w is proportional to the likelihood 
x prior

Use the Metropolis algorithm as defined in equation 11.33 to compute an estimate 
of the weights.

A few practical tips - you'll need to use the log of the posterior, i.e.  log (likelihood 
x prior) instead of the actual probabilities, due to numerical precision problems 
that will crop up with 25 training observations.


Remember to give the Markov chain a chance to 'burn in' and let it run for a few 
hundred samples or so before you start using those samples to compute an average 
on the weights.

The actual 'burn in' time is dependent on the proposal distribution you choose.

The proposal distribution you use can be different for each step of the algorithm 
- note this is different than in rejection sampling. A common choice is to recenter 
the proposal distribution on the previous stored sample.  See the bottom of page 
541- top of 542 for a more detailed explanation of this.

'''

# Linear regression 

def GenData(x, A0,A1, sigma):
    # gen data along the line A0 + x*A1 and add mean 0 std sigma noise on top
    rand_noise =  np.random.normal(0, sigma,size=(len(x)) )
    Target = A0 + x*(A1) + rand_noise 
    return Target

def likelihood(x, t, w, beta):
    # calculate the likelihood
    iota = np.concatenate((np.ones((len(x),1)), x*np.ones((len(x),1))), axis = 1)
    p = scipy.stats.norm(w@iota.T, beta**(-1))    
    return np.squeeze(p.pdf(t))

def posterior(w,x,t,alpha,beta):
    # iota = np.expand_dims(np.concatenate((np.ones(x.shape[0]), x),axis=1 ),0)
    iota = np.column_stack((np.ones(x.shape[0]), x))

    sn = np.linalg.inv(alpha*np.eye(iota.shape[1]) + beta*iota.T@iota)
    mn = (beta* (sn@iota.T)@t)
    posterior = scipy.stats.multivariate_normal(np.squeeze(mn), sn) 
    return posterior.pdf(w), posterior

def part2():
    # Generate Data to model (linear model)
    x_data  = np.random.uniform(-1,1,(25))
    a0,a1 = -0.3, 0.5
    sigma = .2 
    tn = GenData( x_data, a0, a1, sigma)
    
    alpha = 2
    beta = (1/sigma)**2
    
    # Plotting Parameters to make it legible 
    plt.rcParams['figure.figsize'] = (14,20) # make the plot bigger
   
    # First plot is empty but you need to put the title
    loc = 1
    plt.subplot(4,3,loc)
    plt.title('Likelihood')
    plt.axis('off')
    plt.axis('square')
    
    # Plot the Prior dist. (a multivariate gaussian)
    loc = 2  
    plt.subplot(4,3,loc) 

    # generate the meshgrid to plot contour of the prior
    x, y = np.mgrid[-1:1:.001, -1:1:.001]
    grid = np.empty(x.shape + (2,)) 
    
    # plot prior 
    grid[:, :, 0] = x
    grid[:, :, 1] = y 
    rv = scipy.stats.multivariate_normal([0,0],(alpha**(-1))*np.eye(2)) 
    z = rv.pdf(grid)
    plt.contourf(x, y, z, levels=50, cmap = 'jet')
    plt.plot(a0,a1,'+')

    plt.xlabel('W0', fontsize=10)
    plt.ylabel('W1',labelpad=30, fontsize=10, rotation=0)
    plt.title('Prior/Posterior')
    plt.axis('square')
    
    # Plot the dataspace of the prior
    # (sample the weights plot the equations)
    loc = 3
    plt.subplot(4,3,loc) 
    w = rv.rvs(6)
    x1 = np.linspace(-1,1,num=100)
    
    for i in range(len(w)):
        #loop 6 times and print the samples of the w mat as lines
        yTemp = w[i,0] + x1*w[i,1]
        plt.plot(x1,yTemp)


    plt.axis('square')
    plt.ylim(bottom=-1,top= 1) 
    plt.xlim(left =-1,right=1)
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y',labelpad=10, fontsize=10, rotation=0)
    plt.title('Data Space')
    
    # Plot the Likelihood
    loc = 4
    plt.subplot(4,3,loc) 
     
    w0, w1 = np.mgrid[-1:1:.01, -1:1:.01]
    grid = np.empty(w0.shape + (2,)) 
    
    grid[:, :, 0] = w0
    grid[:, :, 1] = w1

    plt.contourf(w0, w1, likelihood( [x_data[0]], [tn[0]], grid, sigma**-1), levels=50, cmap = 'jet')
    plt.plot(a0,a1,'+')
     
    plt.xlabel('W0', fontsize=10)
    plt.ylabel('W1',labelpad=30, fontsize=10, rotation=0)
    plt.title('Likelihood')
    plt.axis('square')
    
    loc = 5
    plt.subplot(4,3,loc) 

    w0, w1 = np.mgrid[-1:1:.01, -1:1:.01]
    grid = np.empty(w0.shape + (2,)) 
    grid[:, :, 0] = w0
    grid[:, :, 1] = w1

    z, post = posterior(grid, x_data[0:1], tn[0:1], alpha, sigma**-1)
    plt.contourf(w0, w1, z , levels=50, cmap = 'jet')
    plt.plot(a0,a1,'+')

    plt.xlabel('W0', fontsize=10)
    plt.ylabel('W1',labelpad=30, fontsize=10, rotation=0)
    plt.title('Prior/Posterior')
    plt.axis('square')

    loc = 6
    plt.subplot(4,3,loc) 
    w = post.rvs(6)
    x1 = np.linspace(-1,1,num=100)
     

    for i in range(len(w)):
        yTemp = w[i,0] + x1*w[i,1]
        plt.plot(x1,yTemp)

    plt.plot( x_data[0], tn[0], 'bo')

    plt.axis('square')
    plt.ylim(bottom=-1,top= 1) 
    plt.xlim(left =-1,right=1)
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y',labelpad=10, fontsize=10, rotation=0)
    plt.title('Data Space')
  
    # Plot the Likelihood
    loc = 7
    plt.subplot(4,3,loc) 
     
    w0, w1 = np.mgrid[-1:1:.01, -1:1:.01]
    grid = np.empty(w0.shape + (2,)) 
    
    grid[:, :, 0] = w0
    grid[:, :, 1] = w1

    plt.contourf(w0, w1, likelihood( [x_data[1]], [tn[2]], grid, sigma**-1), levels=50, cmap = 'jet')
    plt.plot(a0,a1,'+')
     
    plt.xlabel('W0', fontsize=10)
    plt.ylabel('W1',labelpad=30, fontsize=10, rotation=0)
    plt.title('Likelihood')
    plt.axis('square')
    
    loc = 8
    plt.subplot(4,3,loc) 
    
    w0, w1 = np.mgrid[-1:1:.01, -1:1:.01]
    grid = np.empty(w0.shape + (2,)) 
    grid[:, :, 0] = w0
    grid[:, :, 1] = w1

    z, post = posterior(grid, x_data[0:2], tn[0:2], alpha, sigma**-1)
    plt.contourf(w0, w1, z , levels=50, cmap = 'jet')
    plt.plot(a0,a1,'+')

    plt.xlabel('W0', fontsize=10)
    plt.ylabel('W1',labelpad=30, fontsize=10, rotation=0)
    plt.title('Prior/Posterior')
    plt.axis('square')

    loc = 9
    plt.subplot(4,3,loc) 
    w = post.rvs(6)
    x1 = np.linspace(-1,1,num=100)
     
    for i in range(len(w)):
        yTemp = w[i,0] + x1*w[i,1]
        plt.plot(x1,yTemp)

    plt.plot( x_data[0:2], tn[0:2] ,'bo')
    plt.axis('square')
    plt.ylim(bottom=-1,top= 1) 
    plt.xlim(left =-1,right=1)
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y',labelpad=10, fontsize=10, rotation=0)
    plt.title('Data Space')
    
    # Plot the Likelihood
    loc = 10
    plt.subplot(4,3,loc) 
     
    w0, w1 = np.mgrid[-1:1:.01, -1:1:.01]
    grid = np.empty(w0.shape + (2,)) 
    
    grid[:, :, 0] = w0
    grid[:, :, 1] = w1

    plt.contourf(w0, w1, likelihood( [x_data[-1]], [tn[-1]], grid, sigma**-1), levels=50, cmap = 'jet')
    plt.plot(a0,a1,'+')
     
    plt.xlabel('W0', fontsize=10)
    plt.ylabel('W1',labelpad=30, fontsize=10, rotation=0)
    plt.title('Likelihood')
    plt.axis('square')
    
    loc = 11
    plt.subplot(4,3,loc) 
    
    w0, w1 = np.mgrid[-1:1:.01, -1:1:.01]
    grid = np.empty(w0.shape + (2,)) 
    grid[:, :, 0] = w0
    grid[:, :, 1] = w1

    z, post = posterior(grid, x_data[0:], tn[0:], alpha, sigma**-1)
    plt.contourf(w0, w1, z , levels=50, cmap = 'jet')
    plt.plot(a0,a1,'+')

    plt.xlabel('W0', fontsize=10)
    plt.ylabel('W1',labelpad=30, fontsize=10, rotation=0)
    plt.title('Prior/Posterior')
    plt.axis('square')

    loc = 12
    plt.subplot(4,3,loc) 
    w = post.rvs(6)
    x1 = np.linspace(-1,1,num=100)
     
    for i in range(len(w)):
        yTemp = w[i,0] + x1*w[i,1]
        plt.plot(x1,yTemp)
    
    plt.plot( x_data[0:], tn[0:],'bo')

    plt.axis('square')
    plt.ylim(bottom=-1,top= 1) 
    plt.xlim(left =-1,right=1)
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y',labelpad=10, fontsize=10, rotation=0)
    plt.title('Data Space')
    
    plt.tight_layout()
    plt.savefig('project_2_graphs/3_7.pdf')



if __name__ == '__main__':
    
    ########## Rejection Sampling ###########
    part1()

    ########## 2D 3 Cluster Case ###########
    # part2()
    
