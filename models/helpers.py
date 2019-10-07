import numpy as np
import matplotlib.pyplot as plt
from measures import compute_cobeau, compute_nlpd
import seaborn as sns


#ToDo: Propagate them through the methods
iters = 100
l2 = 1
n_std = 4



def plot_uncertainty_kaggle(model,X,y,n_std=4,raw=False, sort=True,iters=100):
    X_ = np.arange(len(y))
    
    if sort==True:
        index = np.argsort(y.squeeze())
    else: 
        #just use index [0,1,2,3,4,...]
        index = X_


    fig, ax = plt.subplots(1,1)
    if raw:
        try:
            y_mean, y_std = model.uncertainty_function(X, iters, l2=l2,raw=True)
        except Exception as e:
            print(f"this network has no raw uncertainty. Please consider using DropoutEnsembles instead, {e}")
    else:
            y_mean, y_std = model.uncertainty_function(X, iters, l2=l2)

    print(f'variance in plot is {y_std[index]}')

    ax.plot(X_, y[index], ls="none", marker="x", color="black", label="observed",ms =7)
    #ax.plot(X_, y, ls="-", color="r", label="true")
    ax.plot(X_, y_mean[index], ls="none", color="purple", label="mean",marker="_")
    ax.errorbar(X_, y_mean[index] , yerr=y_std[index], label='unctertainty',color="purple",alpha=0.1,marker="_",uplims=True, lolims=True,fmt='none')
    

#     for i in range(n_std):
#         ax.fill_between(
#             X_.squeeze(),
#             y_mean.squeeze() - y_std.squeeze() * ((i+1)/2),
#             y_mean.squeeze() + y_std.squeeze() * ((i+1)/2),
#             color="purple",
#             alpha=0.1
#         )




    ax.legend()
    sns.despine(offset=10)
    
    
    print(compute_cobeau(y.squeeze(),y_mean.squeeze(),y_std.squeeze()))
    print(compute_nlpd(y.squeeze(),y_mean.squeeze(),y_std.squeeze()))
    
def plot_uncertainty_toy(model,X,y,n_std=4,raw=False,all_predictions=True,iters=100):


    fig, ax = plt.subplots(1,1)


    if all_predictions:
        y_mean, y_std, outputs = model.uncertainty_function(X, iters, l2=l2,all_predictions=all_predictions)
        print(outputs.shape)
        for i,prediction in enumerate(outputs.T):
            ax.plot(X,prediction,alpha=0.3)
            
    else:
        if raw:
            try:
                #y_mean, y_std = model.uncertainty_function(X_long, iters, l2=l2,raw=True)
                y_mean, y_std = model.uncertainty_function(X, iters, l2=l2,raw=True)
            except Exception as e:
                print(f"this network has no raw uncertainty. Please consider using DropoutEnsembles instead, {e}")
        else:
                #y_mean, y_std = model.uncertainty_function(X_long, iters, l2=l2)
                y_mean, y_std = model.uncertainty_function(X, iters, l2=l2)

    ax.plot(X, y, ls="none", marker="x", color="black", alpha=0.5, label="observed")
    #ax.plot(X_long, y_long, ls="-", color="r", label="true")
    ax.plot(X, y_mean, ls="-", color="purple", label="mean")


    for i in range(n_std):
        ax.fill_between(
            X.squeeze(),
            y_mean.squeeze() - y_std.squeeze() * ((i+1)/2),
            y_mean.squeeze() + y_std.squeeze() * ((i+1)/2),
            color="purple",
            alpha=0.1
        )




    ax.legend()
    sns.despine(offset=10)
    
    print(compute_cobeau(y.squeeze(),y_mean.squeeze(),y_std.squeeze()))
    print(compute_nlpd(y.squeeze(),y_mean.squeeze(),y_std.squeeze()))

def plot_uncertainty(model,X,y,toy=False, n_std=4,raw=False, sort=True, all_predictions=True):
    """decide which plot function is appropriate"""
    if not toy:
        plot_uncertainty_kaggle(model,X,y, n_std=4,raw=False, sort=True)
    else:
        plot_uncertainty_toy(model,X,y, n_std=4,raw=False, all_predictions=all_predictions)
    

    
    
    