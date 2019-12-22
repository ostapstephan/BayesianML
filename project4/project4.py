# python 3
# Ostap Voynarovskiy 
# Prof. Keene
# Project 4

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import scipy.io
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from scipy.io import arff 

'''
Part 1: 
    Re-do the plotting predictions of your linear regression 
    using the Gaussian Process method from section 6.4. 
    Stretch goal:Try to learn the hyperparameters by maximizing eqn 6.70.
Part 2: 
    Re-do your linear classification assignment using a Support Vector Machine. 
    Do this for the circles dataset, and whatever other dataset you chose from UCI or whathaveyou.  
    Implement the grid search method outlined in the paper, 
    plot an ROC and report your % correct. 
    Compare this to your results from the linear classification project.


Results:
    The SVM algorithim outperformed the plain linear regression and made it 
    easier to achieve better results. This is because it didn't require you to  
    create features to separate the data better. The SVM was able to classify
    the Circles and the banknote datasets perfectly, which was not the case with 
    the plain linear regression for the more complex banknote authentication 
    dataset. SVM's are more useful as the data becomes less interpretable to
    humans and the problems, while still linearly separabl

'''

def plot(x=np.empty([2]), t=np.empty([2]), title='', m = None, b = None):
    if t == np.empty([2]):
        data = scipy.io.loadmat("mlData.mat")
        x, y = data['circles'][0][0][0].T
        t = data['circles'][0][0][1].squeeze() # 400,1 => 400
    else:
        x, y = x[:,:2].T

    if m != None: 
        lsp = np.linspace(-3,3,1000)
        plt.plot(lsp,lsp*m+b)

    plt.scatter(x[t==0],y[t==0])
    plt.scatter(x[t==1],y[t==1])
    
    plt.title(title)
    plt.axis('square')
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y',labelpad=30, fontsize=10, rotation=0)
    plt.matplotlib.pyplot.savefig('plots/'+str(title).replace('\n','_').replace(' ','_')+'.png',)
    plt.show()
    plt.clf()

def plotROC(t_test, t_hat, title= ''):
    fpr, tpr, thresholds = metrics.roc_curve(t_test,t_hat)
    roc_auc = metrics.auc(fpr, tpr)

    plt.clf()
    plt.title( title +' Receiver Operating Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc, linewidth=10)
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.matplotlib.pyplot.savefig('plots/'+str(title).replace('\n','_').replace(' ','_')+'.png')
    plt.show()
    plt.clf()

def accuracy(t_hat, t_test ):
    return  (len(t_hat)- sum(abs(t_hat-t_test))  )/len(t_hat) *100

 
def GaussianProc():
    def format(plt,x=np.array([]),t=np.array([]),color = 'g',title = ''):
        if (x != np.array([])) and (t != np.array([])):
            plt.plot(x,t,color=color,linewidth=1)
        plt.ylim(bottom=-1.5,top= 1.5) 
        plt.xlim(left =0,right=1)
        plt.xlabel('X', fontsize=10)
        plt.ylabel('t',labelpad=10, fontsize=10, rotation=0)
        plt.title(title)

    def k_func(xn,xm,tVec):
        # calculate the grahm matrix for an element of the matriz xn,xm K
        return tVec[0]*np.exp((-tVec[1]/2)*np.linalg.norm(np.expand_dims((xn-xm),0), axis=0 )**2) + tVec[2] + tVec[3]*xn.T*xm

    def k_mat(x,thetaVector):
        # calculate k matrix
        k = np.zeros((x.shape[0],x.shape[0]))
        for n,xn in enumerate(x):
            for m,xm in enumerate(x):
                k[n,m]= k_func(xn,xm,thetaVector)
        return k

    def k_vec(x,x_n1,thetaVector):
        # Calculate the k vector
        k = np.zeros(x.shape[0])
        for n,xn in enumerate(x):
            k[n]= k_func(xn,x_n1,thetaVector)
        return k

    def get_c_mat(x, k, beta):
        # get covariance matrix by adding (beta*identity) to the Grahm Matrix (K)
        C = k + (beta**-1)*np.eye(x.shape[0])
        return C 

    def calc_M_S(x_n1, x, t, beta):
        '''
        Predict Mean and Standard Deviation at a point 
        '''
        tvec = [1,64,0,5]  
        k_m = k_mat(x, tvec)
        c = k_func(x_n1, x_n1, tvec) + beta**-1
        C = get_c_mat(x,k_m, beta)
        k_v = k_vec(x,x_n1,tvec)
        # C_n1 = np.array([[C,k_v],[k_v.T,c]])
        mean =   k_v.T @ np.linalg.inv(C) @ t
        var = c - k_v.T @ np.linalg.inv(C) @ k_v
        return mean,var
    
    plt.rcParams['figure.figsize'] = (16,9) 

    sigma = .2 
    alpha = 2
    beta = (1/sigma)**2
    
    loc = 1
    name = ['N=1','N=2','N=4','N=25']
    for N in [1,2,4,25]: 
        # choose quadrant to plot in 
        plt.subplot(2,2,loc) 
        
        # generate true data
        x_ = np.linspace(0,1,num=100)
        t_ = np.sin(2*np.pi*x_)
         
        # how many randon ints specifies num data points trained on
        r = np.random.randint(0, 99, (N) )

        g_mean = []
        g_var = []
        for x_n1 in x_:
            m_n,var_n= calc_M_S(x_n1, x_[r], t_[r], beta)
            g_mean.append(m_n)
            g_var.append(var_n)

        plt.plot(x_, g_mean,linewidth=1) # plot pred
        plt.plot(x_[r], t_[r],'o', markersize=3) #plot data points 
        high =[g_mean[i]+g_var[i] for i in range(len(g_mean))]
        low = [g_mean[i]-g_var[i] for i in range(len(g_mean))]
        plt.fill_between(x_, high, low, color = 'pink') #plot variance
        format(plt,x_,t_,title = name[loc-1]) # plots true sin wave
        loc+=1
    
    plt.tight_layout()
    plt.savefig('plots/kernel_regression.png')

    plt.clf()

def svm_process():
    '''
    Transform data to the format of an SVM package
    Conduct simple scaling on the data
    Consider the RBF kernel 
    Use cross-validation to find the best parameter C and γ
    Use the best parameter C and γ to train the whole training set
    Test
    '''

    # Transform data to the format of an SVM package
    data = scipy.io.loadmat("mlData.mat")
    
    # Load the data for the circles
    x = data['circles'][0][0][0]
    t = data['circles'][0][0][1].squeeze()
    
    # Conduct simple scaling on the data
    x = (x/x.max(axis=0)) #normalize 
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.20, random_state=42) 
    x_train, x_val, t_train, t_val   = train_test_split(x_train, t_train, test_size=0.10, random_state=42) # 72:8:20 train,val,test

    # c = pentalty parameter
    # gamma = kernel parameters
    # kernel type
    i,j = 20,20
    C_opt= [  2**(x-5)for x in range(i)]
    Gamma_opt= [2**(x-15) for x in range(j)]
    acc = np.zeros((i,j))

    # Consider the RBF kernel and use cross-validation to find the best parameter C and γ
    for ci,vi in enumerate(C_opt):
        for cj,vj in enumerate(Gamma_opt):
            clf = SVC( C=vi, kernel='rbf',gamma=vj)
            clf.fit(x_train, t_train)
            t_hat = clf.predict(x_val)
            a  = accuracy(t_hat, t_val )
            acc[ci,cj] = a 

    # Use the best parameter C and γ to train the whole training set
    c_best,gamma_best = (np.unravel_index(np.argmax(acc),acc.shape))
    clf = SVC(C=C_opt[c_best], kernel='rbf',gamma=Gamma_opt[gamma_best])
    clf.fit(np.concatenate((x_train,x_val)),np.concatenate( (t_train,t_val)))

    # Test
    t_hat = clf.predict(x_test)
    a = accuracy(t_hat, t_test )

    # Plot results
    plot(x_test, t_hat,title=f'SVM\n Best Parameters: kernel = rbf, C={C_opt[c_best]}, gamma={Gamma_opt[gamma_best]} \n accuracy:{a}% ')
    plotROC(t_test, t_hat, 'Circles Data SVM')    

    ##########################################################
    ##          banknote authentication dataset            ### 
    ##########################################################
    df = pd.read_csv('banknote_authentication.txt')         

    x = np.asarray(df[['a','b','c','d']])
    t = np.asarray(df['target'])

    # Conduct simple scaling on the data
    x = (x/x.max(axis=0)) #normalize 

    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.20, random_state=42) 
    x_train, x_val, t_train, t_val   = train_test_split(x_train, t_train, test_size=0.10, random_state=42) 

    i,j = 20,20
    C_opt= [ 2**(x-5)for x in range(i)]
    Gamma_opt= [2**(x-15) for x in range(j)]
    acc = np.zeros((i,j))
    # Consider the RBF kernel and use cross-validation to find the best parameter C and γ
    for ci,vi in enumerate(C_opt):
        for cj,vj in enumerate(Gamma_opt):
            clf = SVC( C=vi, kernel='rbf',gamma=vj)
            clf.fit(x_train, t_train)
            t_hat = clf.predict(x_val)
            a  = accuracy(t_hat, t_val )
            acc[ci,cj] = a 

    # Use the best parameter C and γ to train the whole training set
    c_best,gamma_best = (np.unravel_index(np.argmax(acc),acc.shape))
    clf = SVC(C=C_opt[c_best], kernel='rbf',gamma=Gamma_opt[gamma_best])
    clf.fit(np.concatenate((x_train,x_val)),np.concatenate( (t_train,t_val)))

    # Test
    t_hat = clf.predict(x_test)
    a = accuracy(t_hat, t_test )

    # Plot results
    plot(x_test, t_hat,title=f'Banknote Authentication SVM \n Best Parameters: kernel = rbf, C={C_opt[c_best]}, gamma={Gamma_opt[gamma_best]}. Accuracy:{a}% ')
    plotROC(t_test, t_hat, 'Banknote Authentication Data SVM')    

if __name__ == '__main__':
    # Part 1
    GaussianProc()

    # Part 2
    svm_process()


    
