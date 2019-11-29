import numpy as np


def generate_data(datalen=1000,noise_level=0.2,padding_size=10) -> np.array:
    """returns numpy arrays X and y that can be used as basis for regression problem"""
    
    X_long = np.linspace(0,1,datalen)
    X = X_long[padding_size:]
    X = X[:len(X) - padding_size]  
    


    X = np.insert(X,0,X_long[0])
    X = np.append(X,X_long[-1])

    freq1 = 0.2
    freq2 = 0.15

    freq1 = 0.1
    freq2 = 0.0375
    noise = np.random.randn(len(X)) * noise_level
    
    def generate_y(X_):
        """nonlinear case"""
        X_ = X_ * 200
        y1 = np.sin(X_ * freq1) 
        y2 = np.sin(X_ * freq2) 
        return y1 + y2
    
    def generate_y_linear(X_):
        """linear case"""
        return X_/datalen
    
    y = generate_y(X) + noise
    y_long = generate_y(X_long)
    
    return X, y, X_long, y_long