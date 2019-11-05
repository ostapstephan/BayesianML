import matplotlib.pyplot as plt 
import numpy as np
import scipy.io
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    data = scipy.io.loadmat("mlData.mat")

    x, y = data['circles'][0][0][0].T
    t = data['circles'][0][0][1].squeeze() # 400,1 => 400



    plt.scatter(x[t==0],y[t==0])
    plt.scatter(x[t==1],y[t==1])
    plt.show()

