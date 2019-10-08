import numpy as np
import scipy.stats as stats


def compute_cobeau(y,y_pred,y_std):
    error = np.sqrt((y-y_pred)**2)
    return stats.pearsonr(error, y_std)


def compute_nlpd(y,y_pred,y_std):
    nlpd = 1/2 * (y-y_pred)**2 / y_std + np.log(y_std)
    l = np.mean(nlpd)
    return l
    
def compute_error(y,y_pred):
    return np.mean(np.sqrt((y-y_pred)**2))