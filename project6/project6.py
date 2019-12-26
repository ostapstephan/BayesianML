# python 3
# Ostap Voynarovskiy  
# Prof. Keene
# Project 6

# Isaac Alboucai helped me figure out the metropolis hastings 
# part of the assignment so I am citing him

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
    plt.clf()







'''        
Part two of the sampling methods mini-project is to re-do the first part of your 
linear regression project using MCMC to find an estimate for the weights.

Reuse your project 2 to generate the same training data. Just do this for 25 training 
samples

Use Equation 3.10 as the likelihood function, to be used with the training samples 
you generated. You may select any distribution you want for the prior on the weights, 
and recall that the posterior density on the weights w is proportional to the 
likelihood * prior

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


def loglikelyhood(tn,w,phi,beta):
    # Function to calculate the loglikelyhood for use in the metropolis-hastings algorithm 
    return np.sum(np.log(norm.pdf(tn,(w@phi.T).T,np.sqrt(1/beta))))

def Metropolis_Hastings( burn_in, total_samples, t, phi, beta, init):
    #Function to run the Metropolis-Hastings algorithm
    prop_mean = np.array([0,0])

    #prior distribution mean and covaraiance
    prior_mean = np.array([0,0])
    prior_cov = np.eye(2)
    prop_cov = np.eye(2)*.2
    curr_weights = prop_mean

    count = 0
    weights = []

    # initialize var
    numer = 0
    denom = 0


    #  the metropolis-hastings algorithm
    while(len(weights)< total_samples):
        if count%100==10:
            print(count)
            print(prob)

        prev_weights = curr_weights
        count+=1
        
        curr_weights = np.random.multivariate_normal(mean = prop_mean, cov = prop_cov)

        if init:
            # On the first run the likelyhood plotting the  
            numer = 0
            denom = 0
        else:
            numer = loglikelyhood(t,np.array(curr_weights),phi,beta) 
            denom = loglikelyhood(t,np.array(prev_weights),phi,beta)

        numer+= np.log(multivariate_normal.pdf(curr_weights,mean=prior_mean,cov=prior_cov)) \
                +np.log(multivariate_normal.pdf(curr_weights,mean=prev_weights,cov=prop_cov))

        denom+= np.log(multivariate_normal.pdf(prev_weights,mean=prior_mean,cov=prior_cov))\
                + np.log(multivariate_normal.pdf(prev_weights,mean=curr_weights,cov=prop_cov))

        # logs make division into subtraction 
        prob = numer-denom 

        if(prob>=0):  
            # if p>1 keep it 
            if(count > burn_in): 
                #only keep if we're past the min burn in amount
                weights.append(curr_weights)
        else: 
            # if p<1 keep w probability p 
            if(prob > np.log(np.random.rand(1)) ):
                if(count > burn_in):
                    weights.append(curr_weights)
            else:
                count-=1
                curr_weights=prev_weights
    return weights, np.mean(weights,axis=0)


def GenData(x, A0,A1, sigma):
    # gen data along the line A0 + x*A1 and add mean 0 std sigma noise on top
    rand_noise =  np.random.normal(0, sigma,size=(len(x)) )
    Target = A0 + x*(A1) + rand_noise 
    return Target
    
def likelihood(x, t, w, beta):
    # calculate the likelihood for the single point
    iota = np.concatenate((np.ones((len(x),1)), x*np.ones((len(x),1))), axis = 1)
    p = scipy.stats.norm(w@iota.T, beta**(-1))    
    return np.squeeze(p.pdf(t))

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
    
    burn_in=1000
    total_samples= 10000
    phi = None # this will be ignored since this is the prior 
    init = True
    w,w_avg = Metropolis_Hastings(burn_in,total_samples,tn[0:1],phi, beta,init )
    w = np.array(w) 

    x,y= w[:,0],w[:,1]
    plt.plot(a0,a1,'+')

    plt.hist2d(x, y, bins=100, range=[[-1,1],[-1,1]] ,cmap = 'jet')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlabel('W0', fontsize=10)
    plt.ylabel('W1',labelpad=30, fontsize=10, rotation=0)
    plt.title('Prior/Posterior')
    plt.axis('square')

    
    # Plot the dataspace of the prior
    # (sample the weights plot the equations)
    loc = 3
    plt.subplot(4,3,loc) 
    ws =  w[np.random.randint(0,len(w),7),:]
    x1 = np.linspace(-1,1,num=100)
    
    for i in range(len(ws)):
        #loop 7 times and print the samples of the w mat as lines
        yTemp = ws[i,0] + x1*ws[i,1]
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

    # one data point
    burn_in=1000
    total_samples= 10000
    phi = np.column_stack((np.ones(x_data[0:1].shape[0]), x_data[0:1]))
    init = False 
    w,w_avg = Metropolis_Hastings(burn_in,total_samples,tn[0:1],phi, beta,init )
    w = np.array(w) 

    x,y= w[:,0],w[:,1]
    plt.plot(a0,a1,'+')

    plt.hist2d(x, y, bins=100, range=[[-1,1],[-1,1]] ,cmap = 'jet')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlabel('W0', fontsize=10)
    plt.ylabel('W1',labelpad=30, fontsize=10, rotation=0)
    plt.title('Prior/Posterior')
    plt.axis('square')

    loc = 6
    plt.subplot(4,3,loc) 
    # plot Dataspace
    ws =  w[np.random.randint(0,len(w),7),:]
    x1 = np.linspace(-1,1,num=100)
    
    for i in range(len(ws)):
        #loop 7 times and print the samples of the w mat as lines
        yTemp = ws[i,0] + x1*ws[i,1]
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
    
    # two data points
    burn_in=1000
    total_samples= 10000
    phi = np.column_stack((np.ones(x_data[0:2].shape[0]), x_data[0:2]))
    init = False 
    w,w_avg = Metropolis_Hastings(burn_in,total_samples,tn[0:2],phi, beta,init )
    w = np.array(w) 

    x,y= w[:,0],w[:,1]
    plt.plot(a0,a1,'+')

    plt.hist2d(x, y, bins=100, range=[[-1,1],[-1,1]] ,cmap = 'jet')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlabel('W0', fontsize=10)
    plt.ylabel('W1',labelpad=30, fontsize=10, rotation=0)
    plt.title('Prior/Posterior')
    plt.axis('square')

    loc = 9
    plt.subplot(4,3,loc) 
    ws =  w[np.random.randint(0,len(w),7),:]
    x1 = np.linspace(-1,1,num=100)
    
    for i in range(len(ws)):
        #loop 7 times and print the samples of the w mat as lines
        yTemp = ws[i,0] + x1*ws[i,1]
        plt.plot(x1,yTemp)

    plt.plot( x_data[0:2], tn[0:2], 'bo')

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
    # all training data points
    burn_in=1000
    total_samples= 10000
    phi = np.column_stack((np.ones(x_data[:].shape[0]), x_data[:]))
    init = False 
    w,w_avg = Metropolis_Hastings(burn_in,total_samples,tn[:],phi, beta,init )
    w = np.array(w) 

    x,y= w[:,0],w[:,1]
    plt.plot(a0,a1,'+')

    plt.hist2d(x, y, bins=100, range=[[-1,1],[-1,1]] ,cmap = 'jet')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlabel('W0', fontsize=10)
    plt.ylabel('W1',labelpad=30, fontsize=10, rotation=0)
    plt.title('Prior/Posterior')
    plt.axis('square')

    loc = 12
    plt.subplot(4,3,loc) 
    ws =  w[np.random.randint(0,len(w),7),:]
    x1 = np.linspace(-1,1,num=100)
    
    for i in range(len(ws)):
        #loop 7 times and print the samples of the w mat as lines
        yTemp = ws[i,0] + x1*ws[i,1]
        plt.plot(x1,yTemp)

    plt.plot( x_data[:], tn[:], 'bo')

    plt.axis('square')
    plt.ylim(bottom=-1,top= 1) 
    plt.xlim(left =-1,right=1)
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y',labelpad=10, fontsize=10, rotation=0)
    plt.title('Data Space')

    plt.tight_layout()
    plt.savefig('plots/3_7.pdf')



if __name__ == '__main__':
    
    ########## Rejection Sampling ###########
    part1()

    ########## MCMC linear Regression  ###########
    part2()
    
