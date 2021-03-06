import numpy as np
import matplotlib.pyplot as plt
from measures import compute_cobeau, compute_nlpd, compute_error
import seaborn as sns


#ToDo: Propagate them through the methods
iters = 10
l2 = 1
n_std = 4
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import IPython



def showcase_code(pyfile,class_name = False,showcase=False):
    """shows content of py file"""
    
    if showcase:

        with open(pyfile) as f:
            code = f.read()
            
        if class_name:
            #1. find beginning (class + <name>)
            index = code.find(f'class {class_name}')
            code = code[index:]
            
            #2. find end (class (new class!) or end of script)
            end_index = code[7:].find('class')
            code = code[:end_index]
            
            

        formatter = HtmlFormatter()
        return IPython.display.HTML('<style type="text/css">{}</style>{}'.format(
            formatter.get_style_defs('.highlight'),
            highlight(code, PythonLexer(), formatter)))
    pass


def plot_uncertainty_kaggle(model,X,y,n_std=4,raw=False, sort=True,iters=100,fname=None):
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

    #print(f'variance in plot is {y_std[index]}')

    ax.plot(X_, y[index], ls="none", marker="x", color="black", label="observed",ms =7)
    #ax.plot(X_, y, ls="-", color="r", label="true")
    ax.plot(X_, y_mean[index], ls="none", color="purple", label="mean",marker="_")
    ax.errorbar(X_, y_mean[index] , yerr=y_std[index]*4, label='unctertainty',color="purple",alpha=0.3,marker="_",fmt='none')
    #ax.errorbar(X_, y_mean[index]*4 , yerr=y_std[index]*4, label='unctertainty',color="purple",alpha=0.6,marker="_",fmt='none')
    




    if fname is None:
        ax.legend()
    sns.despine(offset=10)
    
    if fname is not None:
        fig.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)    
#     print(f'cobeau: {compute_cobeau(y.squeeze(),y_mean.squeeze(),y_std.squeeze())}')
#     print(f'nlpd: {compute_nlpd(y.squeeze(),y_mean.squeeze(),y_std.squeeze())}.\n nlpds of just mean and just std of the model:')
#     print(compute_nlpd(y.squeeze(),y.squeeze().mean(), y.squeeze().std())) # https://www.mendeley.com/viewer/?fileId=03696e80-bc97-8d5d-1369-9366d576b414&documentId=1725878a-471c-39b9-88e8-b8c7c4d2ff0e p14 evaluating predictive uncertainty challenge
#     print(compute_nlpd(y.squeeze(),y_mean.squeeze(), (y-y_mean).squeeze())) # https://www.mendeley.com/viewer/?fileId=03696e80-bc97-8d5d-1369-9366d576b414&documentId=1725878a-471c-39b9-88e8-b8c7c4d2ff0e p14 evaluating predictive uncertainty challenge
#     print(f'error: {compute_error(y.squeeze(),y_mean.squeeze())}')
    
    return fig
    
def plot_uncertainty_toy(model,X,y,n_std=4,raw=False,all_predictions=True,iters=10,generating_function = None, fname=None):


    fig, ax = plt.subplots(1,1)
    index = np.argsort(X.squeeze())


    y_mean, y_std = model.uncertainty_function(X, iters, l2=l2)
    
    

            

    ax.plot(X[index], y[index], ls="none", marker="x", color="black", alpha=0.9, label="observed")
    ax.plot(X[index], y_mean[index], ls='none', color="black", label="test set prediction",marker='X')



        
    #X_original = np.expand_dims(np.linspace(0,1,100),1)
    X_original = np.expand_dims(np.linspace(round(X.min()),round(X.max()),100),1)    

    
    if all_predictions:
        y_original_mean, y_original_std, outputs = model.uncertainty_function(X_original, iters, l2=l2,all_predictions=all_predictions)
        #print(f' this many models: {outputs.shape}')
        for i,prediction in enumerate(outputs.T):
            #print(prediction)
            if i == 0:
                ax.plot(X_original,prediction,alpha = 0.2,c='grey',label='ensemble member prediction')
            ax.plot(X_original,prediction,alpha = 0.2,c='grey')
    else:
        y_original_mean, y_original_std = model.uncertainty_function(X_original, iters, l2=l2)

    ax.plot(X_original, y_original_mean, ls="-", color="purple", label="mean")
    if generating_function is not None:
        ax.plot(X_original, generating_function(X_original), ls=":", color="black", label="generating function")



    for i in range(n_std):
        ax.fill_between(
            X_original.squeeze(),
            y_original_mean.squeeze() - y_original_std.squeeze() * ((i+1)/2),
            y_original_mean.squeeze() + y_original_std.squeeze() * ((i+1)/2),
            color="purple",
            alpha=0.1
        )
        

    if fname is None:
        ax.legend()
    sns.despine(offset=10)
    
    print(f'cobeau: {compute_cobeau(y.squeeze(),y_mean.squeeze(),y_std.squeeze())}')
    print(f'nlpd: {compute_nlpd(y.squeeze(),y_mean.squeeze(),y_std.squeeze())}.\n nlpds of just mean and just std of the model:')
    print(compute_nlpd(y.squeeze(),y.squeeze().mean(), y.squeeze().std())) # https://www.mendeley.com/viewer/?fileId=03696e80-bc97-8d5d-1369-9366d576b414&documentId=1725878a-471c-39b9-88e8-b8c7c4d2ff0e p14 evaluating predictive uncertainty challenge
    print(compute_nlpd(y.squeeze(),y_mean.squeeze(), (y-y_mean).squeeze())) # https://www.mendeley.com/viewer/?fileId=03696e80-bc97-8d5d-1369-9366d576b414&documentId=1725878a-471c-39b9-88e8-b8c7c4d2ff0e p14 evaluating predictive uncertainty challenge
    print(f'error: {compute_error(y.squeeze(),y_mean.squeeze())}')
    
    if fname is not None:
        fig.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    
    return fig
    
    
def plot_uncertainty(model,X,y,toy=False, n_std=4,raw=False, sort=True, all_predictions=True,generating_function=False,fname=None):
    """decide which plot function is appropriate"""
    
    
    
    if not toy:
        fig = plot_uncertainty_kaggle(model,X,y, n_std=4,raw=False, sort=True,fname=fname)
    else:
        if generating_function:
            fig = plot_uncertainty_toy(model,X,y, n_std=4,raw=False, all_predictions=all_predictions,generating_function =generating_function,fname=fname)
        else:
            fig = plot_uncertainty_toy(model,X,y, n_std=4,raw=False, all_predictions=all_predictions,fname=fname)
            
    return fig
        

        
        
        
        
def plot_mean_std(X_train,y_train,X_test,y_test, generating_function, N):
    """helper function to plot and evaluate simply taking mean and std of target variable"""
    

    fig, ax = plt.subplots(1,1)

    
    X_original = np.linspace(0,1,N)
    y_original = generating_function(X_original)
    ax.plot(X_original, y_original, ls="-", color="r", label="true")

    predictive_mean = np.ones(X_original.shape) * np.mean(y_train.squeeze())
    predictive_uncertainty = np.ones(X_original.shape) *np.std(y_train.squeeze())
    
    
    ax.plot(X_original, predictive_mean, 'X')
    for i in range(n_std):
        ax.fill_between(
            X_original.squeeze(),
            predictive_mean.squeeze() - predictive_uncertainty.squeeze() * ((i+1)/2),
            predictive_mean.squeeze() + predictive_uncertainty.squeeze() * ((i+1)/2),
            color="purple",
            alpha=0.1
        )
    


    
#     print(f'cobeau: {compute_cobeau(y.squeeze(),y_mean.squeeze(),y_std.squeeze())}')
#     print(f'nlpd: {compute_nlpd(y.squeeze(),y_mean.squeeze(),y_std.squeeze())}.\n nlpds of just mean and just std of the model:')
#     print(compute_nlpd(y.squeeze(),y.squeeze().mean(), y.squeeze().std())) # https://www.mendeley.com/viewer/?fileId=03696e80-bc97-8d5d-1369-9366d576b414&documentId=1725878a-471c-39b9-88e8-b8c7c4d2ff0e p14 evaluating predictive uncertainty challenge
#     print(compute_nlpd(y.squeeze(),y_mean.squeeze(), (y-y_mean).squeeze())) # https://www.mendeley.com/viewer/?fileId=03696e80-bc97-8d5d-1369-9366d576b414&documentId=1725878a-471c-39b9-88e8-b8c7c4d2ff0e p14 evaluating predictive uncertainty challenge
#     print(f'error: {compute_error(y.squeeze(),y_mean.squeeze())}')

    
    
    
def plot_generating_function(X,y,generating_function,noise_level,N):
    """helper function to plot and evaluate the generating function"""
    
    fig, ax = plt.subplots(1,1)

    
    index = np.argsort(X.squeeze())
    y_mean = generating_function(X.squeeze())
    y_std = np.ones(y_mean.shape) * noise_level
    X_original = np.linspace(0,1,N)
    y_original = generating_function(X_original)
    
    ax.plot(X[index], y[index], ls="none", marker="x", color="black", alpha=0.5, label="observed")
    ax.plot(X_original, y_original, ls="-", color="r", label="true")
    ax.plot(X[index], y_mean[index], ls="-", color="purple", label="mean",marker='X')


    for i in range(n_std):
        ax.fill_between(
            X[index].squeeze(),
            y_mean[index].squeeze() - y_std[index].squeeze() * ((i+1)/2),
            y_mean[index].squeeze() + y_std[index].squeeze() * ((i+1)/2),
            color="purple",
            alpha=0.1
        )



    
    print(f'cobeau: {compute_cobeau(y.squeeze(),y_mean.squeeze(),y_std.squeeze())}')
    print(f'nlpd: {compute_nlpd(y.squeeze(),y_mean.squeeze(),y_std.squeeze())}.\n nlpds of just mean and just std of the model:')
    print(compute_nlpd(y.squeeze(),y.squeeze().mean(), y.squeeze().std())) # https://www.mendeley.com/viewer/?fileId=03696e80-bc97-8d5d-1369-9366d576b414&documentId=1725878a-471c-39b9-88e8-b8c7c4d2ff0e p14 evaluating predictive uncertainty challenge
    print(compute_nlpd(y.squeeze(),y_mean.squeeze(), (y-y_mean).squeeze())) # https://www.mendeley.com/viewer/?fileId=03696e80-bc97-8d5d-1369-9366d576b414&documentId=1725878a-471c-39b9-88e8-b8c7c4d2ff0e p14 evaluating predictive uncertainty challenge
    print(f'error: {compute_error(y.squeeze(),y_mean.squeeze())}')

    