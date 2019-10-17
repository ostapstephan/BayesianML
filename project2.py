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
    Target = A0+ x*A1 +np.random.normal(0, sigma)  
    return Target

def likelyhood( x, t, w, beta):
    # p = np.exp( 0.5*N*ln(beta)-0.5*N*ln(2*np.pi)-beta*e_d)
    # W = meshgrid  
    iota = np.concatenate((np.ones((len(x),1)), x*np.ones((len(x),1))), axis = 1)
    p = scipy.stats.normal(w.T*iota.pdf(t), beta**(-1))    
    return p

if __name__ == '__main__':
    # Generate Data to model (linear model)
    x_data  = np.random.uniform(-1,1,(20))
    a0,a1 = -0.3, 0.5
    sigma = .2 
    tn = GenData(x_data, a0, a1, sigma)
     
    
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
    
    grid[:, :, 0] = x
    grid[:, :, 1] = y 
    rv = scipy.stats.multivariate_normal([0,0],(alpha**(-1))*np.eye(2)) 
    z = rv.pdf(grid)
    plt.contourf(x, y, z, levels=50, cmap = 'jet')

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
    plt.title('Prior/Posterior')
    
    # Plot the Likelyhood
    loc = 4
    plt.subplot(4,3,loc) 
     
    w0, w1 = np.mgrid[-1:1:.01, -1:1:.01]
    grid = np.empty(w0.shape + (2,)) 

    x_data[0] 
    t[0]
    plt.contourf(w0, w1, likelyhood( x[0], t[0], grid, sigma**-1), levels=50, cmap = 'jet')
     
    plt.xlabel('W0', fontsize=10)
    plt.ylabel('W1',labelpad=30, fontsize=10, rotation=0)
    plt.title('Prior/Posterior')
    plt.axis('square')


    


     
    plt.tight_layout()
    plt.savefig('project_2_graphs/3_7.pdf')




'''
Stretch goal for the brave: 
Read section 3.5 on the evidence approximation and use the 
approximation to find the 'best' values of alpha and beta. 
This can be done with eqn's 3.98 and 3.99.
'''
