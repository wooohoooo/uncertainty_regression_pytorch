import numpy as np
import scipy.stats as stats


def safe_ln(x):
    return np.log(x+0.0001)


def compute_cobeau(y,y_pred,y_std):
    error = np.sqrt((y-y_pred)**2)
    return stats.pearsonr(error, y_std)


def compute_nlpd(y,y_pred,y_std):
    nlpd = 1/2 * ((y-y_pred)**2 / (y_std +0.0001)) + safe_ln(y_std)
    l = np.mean(nlpd)
    return l
    #return -1/2 *np.mean( safe_ln(y_std) + ((y_pred - y)**2/(y_std+0.0001)))


    
def compute_error(y,y_pred):
    return np.mean(np.sqrt((y-y_pred)**2))