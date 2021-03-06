
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

    
def normalize(df,minmax=False):
        if minmax:
            return (df-df.min())/(df.max()-df.min())
        
        return (df-df.mean())/df.std()

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
    
    #print(df.shape)
    
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


    
# the function that generates y
def generate_y(X_):
    """nonlinear case"""
        
        
    # define the frequencies of the sinoid
    freq1 = 20#0.1
    freq2 = 7.5#0.0375
    
    #X_ = X_ * 200
    y1 = np.sin(X_ * freq1) 
    y2 = np.sin(X_ * freq2) 
    return y1 + y2
    
def generate_y_linear(X_):
        """linear case"""
        return X_

def generate_data(datalen=1000,noise_level=0.2,padding_frac=0.1, out_of_sample = True, generator_function = generate_y) -> np.array:
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


    # make some noise!
    seed = 42424
    np.random.seed(seed)
    noise = np.random.randn(len(X)) * noise_level

    
    # the original function values
    y_long = generator_function(X_long)
    y_long = normalize(y_long)
    
    # it all comes together: generated function plus noise
    y = generator_function(X) + noise
    y = normalize(y)

    
    X = np.expand_dims(X,1)
    X_long = np.expand_dims(X_long,1)
    
    y = np.expand_dims(y,1)
    y_long = np.expand_dims(y_long,1)
    
    

    return X, y, X_long, y_long



def get_X_y(toy,seed=42,out_of_sample = False,plot=False,fname=None):
    """obtain X, y and N depending on <toy>
    either calls generate_data_
    or load_data
    generates cross validation samples"""
    if not toy:
        X,y = load_data()
        N = X.shape[0]
        if plot:
            fig = plt.figure()
            plt.plot(list(range(len(y))), y, ls="none", color="green", label="dataset unsorted",marker="_")
            plt.plot(list(range(len(y))), np.sort(y), ls="none", color="black", label="dataset sorted by y value for easy visualisation",marker="x",ms=7)
            
            #, ls="none", marker="x", color="black", label="observed",ms =7
            
            #plt.ylabel('house price (normalized)')
            #plt.xlabel('house identifier (not actual X)')
            #plt.legend()
            if fname is not None:
                fig.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format='pdf',
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)    
                
                
                
                
                
                
                

        y = np.expand_dims(y,1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        output_dims = X_train.shape[1]
        return X_train, X_test, y_train, y_test, N, output_dims
    
    
    N = 100
    X,y,X_long,y_long = generate_data(N,0.3,out_of_sample = False) # oos now below
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

              
        
    if out_of_sample:

        X_test = np.insert(X_test,0,0)
        X_test = np.append(X_test,1)
        X_test = np.expand_dims(X_test,1)
        y_test = np.insert(y_test,0,0)
        y_test = np.append(y_test,1)
        y_test = np.expand_dims(y_test,1)
        
        #print(X_test.shape)
        #print(y_test.shape)
    if plot:
        fig = plt.figure()
        plt.plot(X_train,y_train,ls="none", marker="x", color="blue", label="train set",ms =7)
        plt.plot(X_test,y_test,ls="none", marker="x", color="black", label="test set",ms =7)
        plt.plot(X_long, y_long,ls=":", color="black", label="generating function")

        #plt.legend()
        if fname is not None:
            fig.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format='pdf',
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)  
         
    output_dims = X_train.shape[1]
    #print(X_train.shape,X_test.shape)
    return X_train, X_test, y_train, y_test, N, output_dims


def generate_y_x3(X):
    return X**3

def get_X_y_small_toy(seed,datalen=20,plot=False,fname=None):
    """returns numpy arrays X and y that can be used as basis for regression problem"""
    
    
    
    np.random.seed(seed)
    # define data space
    X_train = np.linspace(-4,4,datalen)
    
    X_test = np.insert(X_train,0,-6)
    X_test = np.append(X_test,6)
    
    noise = np.random.randn(len(X_test)) * 3**2
    
    y_original = generate_y_x3(X_test)
    y_test = y_original + noise
    y_train = y_test[1:-1]
    plt.plot(X_test,y_original)
    plt.plot(X_test,y_test)
    plt.plot(X_train,y_train)
    
    X_train = np.expand_dims(X_train,1)
    X_test = np.expand_dims(X_test,1)
    y_train = np.expand_dims(y_train,1)
    y_test = np.expand_dims(y_test,1)
    output_dims = X_train.shape[1]
    
    
    if plot:
        fig = plt.figure()
        plt.plot(X_test,y_test,ls="none", marker="x", color="black", label="test set",ms =7)
        plt.plot(X_train,y_train,ls="none", marker="x", color="blue", label="train set",ms =7)
        plt.plot(X_test, y_original,ls=":", color="black", label="generating function")
        #plt.title('small synthetic dataset')
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.legend()
        if fname is not None:
            fig.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format='pdf',
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None) 

    return X_train, X_test, y_train, y_test, datalen, output_dims

