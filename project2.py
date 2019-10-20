# python 3
# Ostap Voynarovskiy And Carena Toy
# Prof. Keene
# Project 2


import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.special
import scipy.stats  

'''
Reproduce figures 3.7 and 3.8 in the textbook.
The exact data they used is described in the text. 
Note your figure will look slightly different because they rely on random draws from the dataset.
'''

def GenData(x, A0,A1, sigma):
    # gen data along the line A0 + x*A1 and add mean 0 std sigma noise on top
    rand_noise =  np.random.normal(0, sigma,size=(len(x)) )
    Target = A0 + x*(A1) + rand_noise 
    return Target

def likelyhood(x, t, w, beta):
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

if __name__ == '__main__':
    # Generate Data to model (linear model)
    x_data  = np.random.uniform(-1,1,(20))
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
    print('empty grid',grid.shape)
    
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
    
    # Plot the dataspace 
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

    plt.contourf(w0, w1, likelyhood( [x_data[0]], [tn[0]], grid, sigma**-1), levels=50, cmap = 'jet')
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

    plt.contourf(w0, w1, likelyhood( [x_data[1]], [tn[2]], grid, sigma**-1), levels=50, cmap = 'jet')
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

    plt.contourf(w0, w1, likelyhood( [x_data[-1]], [tn[-1]], grid, sigma**-1), levels=50, cmap = 'jet')
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




'''
Stretch goal for the brave: 
Read section 3.5 on the evidence approximation and use the 
approximation to find the 'best' values of alpha and beta. 
This can be done with eqn's 3.98 and 3.99.
'''
