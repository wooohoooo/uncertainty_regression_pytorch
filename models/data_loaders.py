
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def load_data(scale_y = True):
    """loads kaggle housing price dataset
    removes non-numerical values
    removes outliers
    scales dataset"""
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = pd.read_csv('train.csv').select_dtypes(include=numerics).dropna()
    
    #remmove outliers
    q = df["SalePrice"].quantile(0.99)
    df = df[df["SalePrice"] < q]

    #df
    
    print(df.shape)
    
    y = df['SalePrice']
    X = df.drop('SalePrice',axis=1)
    
    def normalize(df,minmax=False):
        if minmax:
            return (df-df.min())/(df.max()-df.min())
        
        return (df-df.mean())/df.std()

    
    # normalize
    X = normalize(X)
    if scale_y:
        y = normalize(y)
        
        

    return X.values,y.values


def generate_data(datalen=1000,noise_level=0.2,padding_frac=0.1, out_of_sample = True) -> np.array:
    """returns numpy arrays X and y that can be used as basis for regression problem"""
    
    padding_size = int(padding_frac * datalen)
    
    # original X, datalen points between 0 and 1
    X_long = np.linspace(0,1,datalen)
    
    #actual X that we use - padded left and right
    X = X_long[padding_size:]
    X = X[:len(X) - padding_size]  
    

    if out_of_sample:
        # add the first and last datapoints back to X to have out of sample datapoint
        X = np.insert(X,0,X_long[0])
        X = np.append(X,X_long[-1])

    # define the frequencies of the sinoid
    freq1 = 0.1
    freq2 = 0.0375
    
    # make some noise!
    noise = np.random.randn(len(X)) * noise_level
    
    # the function that generates y
    def generate_y(X_):
        """nonlinear case"""
        X_ = X_ * 200
        y1 = np.sin(X_ * freq1) 
        y2 = np.sin(X_ * freq2) 
        return y1 + y2
    
    def generate_y_linear(X_):
        """linear case"""
        return X_
    
    # the original function values
    y_long = generate_y(X_long)
    
    # it all comes together: generated function plus noise
    y = generate_y(X) + noise
    X = np.expand_dims(X,1)
    X_long = np.expand_dims(X_long,1)
    
    y = np.expand_dims(y,1)
    y_long = np.expand_dims(y_long,1)

    return X, y, X_long, y_long



def get_X_y(toy):
    """obtain X, y and N depending on <toy>
    either calls generate_data_
    or load_data"""
    if not toy:
        X,y = load_data()
        N = X.shape[0]
        plt.plot(list(range(len(y))), y, ls="none", color="green", label="dataset unsorted",marker="_")
        plt.plot(list(range(len(y))), np.sort(y), ls="none", color="purple", label="dataset sorted by y value for easy visualisation",marker="_")
        plt.ylabel('house price (normalized)')
        plt.xlabel('house identifier (not actual X)')
        plt.legend()


        y = np.expand_dims(y,1)
        return X,y, N
    
    N = 100
    X,y,X_long,y_long = generate_data(N,0.3)
    plt.plot(X,y,'x',label='observed values',c='black')
    plt.plot(X_long, y_long,label='generating function',c='r')
    plt.title('the Dataset')
    plt.xlabel('this is between 0 and 1')
    plt.ylabel('this is a combination of two sioids and a bit of noise')
    plt.legend()
    return X,y, N
