from data_loaders import get_X_y, generate_y
toy = True
from measures import compute_cobeau, compute_nlpd, compute_error

from helpers import plot_uncertainty

import numpy as np
import torch

seed = 42 #424
np.random.seed(seed)
torch.manual_seed(seed)
#NOTE! This only works for non cudnn. gpu needs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from helpers import showcase_code

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16,8)

import os
import seaborn as sns
import time
import pandas as pd
iters = 100
l2 = 1
n_std = 4

class Experimentator(object):
    def __init__(self,num_experiments,num_epochs,model_type,toy,seed=None,generator_function=None):
        self.toy = toy
        self.generator_function = generator_function or False
        self.num_experiments = num_experiments
        self.num_epochs = num_epochs
        self.model_type = model_type
        self.seed = seed or 42
        
        
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        #NOTE! This only works for non cudnn. gpu needs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.X_train, self.X_test, self.y_train, self.y_test, self.N, self.output_dims  = get_X_y(self.toy,seed=self.seed)
    
        

        self.stats_dict = {'pre_training':{'means':[],
                                      'stds':[],
                                      'outcomes':[]
                                     },
                      'post_training':{'means':[],
                                      'stds':[],
                                      'outcomes':[]
                                      },
                      'training':{'losses':[],
                                  'final_losses':[],
                                  'training_times':[]
                                 },

                      'analysis':{'test_errors':[],
                                  'cobeau':[],
                                  'cobeau_p':[],
                                  'nlpd':[],
                                  'prior_test_errors':[],
                                  'prior_nlpd':[],
                                  'prior_cobeau':[],
                                  'prior_cobeau_p':[]
                                  
                                 },
                      'models': []
                     }
        
        model_string = f'{model_type}'
        print(model_string)
        print('.models.' in model_string)
        if '.models.' in model_string:
            index_start = model_string.find('.models.')+len('.models.')
            index_stop = model_string.find("'>")
            self.model_name = model_string[index_start:index_stop]

        elif '.ensembles.' in model_string:
            index_start = model_string.find('.ensembles.')+len('.ensembles.')
            index_stop = model_string.find("'>") 
            self.model_name = model_string[index_start:index_stop]

        else: self.model_name = 'VanillaEnsemble'

        

    def run_experiment(self):
        for i in range(self.num_experiments):

            np.random.seed(self.seed + i*100000)
            torch.manual_seed(self.seed + i*100000)
            try:
                model = self.model_type(self.toy,self.output_dims,save_path=f'experiments/experiment_{i}_{self.model_name}_{self.toy}/')
            except Exception as e:
                #print(e)
                try:
                    model = self.model_type(self.toy,self.output_dims,dataset_lenght=self.X_train.shape[0])
                except Exception as e:
                    #print(e)
                    model = self.model_type(self.toy,self.output_dims)


            losslist = []
            try:
                mean, std, outcomes = model.uncertainty_function(self.X_test, iters, l2=l2,all_predictions=True)
                self.stats_dict['pre_training']['means'].append(mean)
                self.stats_dict['pre_training']['stds'].append(std)
                self.stats_dict['pre_training']['outcomes'].append(outcomes)
            except Exception as e:
                a= 0
                #print('pre-training information not available for methods that rely on ensembling through time')
            
            start = time.time()
            for i in range(self.num_epochs):
                loss = model.fit_model(self.X_train, self.y_train)
                losslist.append(loss)

                if i == self.num_epochs - 1:
                    try:
                        self.stats_dict['training']['final_losses'].append(loss.data.numpy())
                    except:
                        self.stats_dict['training']['final_losses'].append(loss)
                        
            time_now = time.time()
            self.stats_dict['training']['training_times'].append(time_now-start)
            #print(f'the training for {self.num_epochs} took {time_now-start} ')

            self.stats_dict['training']['losses'].append(losslist)
            plt.plot(losslist)

            mean, std, outcomes = model.uncertainty_function(self.X_test, iters, l2=l2,all_predictions=True)

            self.stats_dict['models'].append(model)

            self.stats_dict['post_training']['means'].append(mean)
            self.stats_dict['post_training']['stds'].append(std)
            self.stats_dict['post_training']['outcomes'].append(outcomes)


    
class ExperimentAnalyzer(object):
    def __init__(self,experiment: Experimentator):
        self.experiment = experiment
        self.stats_dict = experiment.stats_dict.copy()
        self.model_name = experiment.model_name
        self.X_train, self.X_test, self.y_train, self.y_test, self.N, self.output_dims, self.toy = experiment.X_train, experiment.X_test, experiment.y_train, experiment.y_test, experiment.N, experiment.output_dims, experiment.toy
        #index of non-outliers to make choosing which experiments to keep easy
        self.outlier_keep_index = list(range(self.experiment.num_experiments))
        self.fig_path = f'figures\\{self.model_name}_toy_{self.toy}_{self.experiment.num_experiments}\\'
        try:
            os.makedirs(os.getcwd() + '\\' + self.fig_path )
        except OSError:
            print ("Creation of the directory %s failed" % self.fig_path)
        else:
            print ("Successfully created the directory %s " % self.fig_path)    
    def plot_models(self,metric='test_errors'):
        
        assert len(self.stats_dict['analysis'][metric]) == len(self.stats_dict['models']), 'number of models and metrics isnt the same'
        
        metric_array = np.array(self.stats_dict['analysis'][metric])[self.outlier_keep_index]
        print(len(metric_array))
        best_model_index = (np.abs(metric_array-0)).argmin()
        worst_model_index =(np.abs(metric_array-0)).argmax()
        
        print(best_model_index,worst_model_index)
        
        best_model = np.array(self.stats_dict['models'])[self.outlier_keep_index][best_model_index]
        worst_model = np.array(self.stats_dict['models'])[self.outlier_keep_index][worst_model_index]
        
        best_fig = plot_uncertainty(best_model,self.X_test,self.y_test,self.toy,all_predictions=True)
        worst_fig = plot_uncertainty(worst_model,self.X_test,self.y_test,self.toy,all_predictions=True)

        best_fig.suptitle(f'{self.model_name} with {metric} closest to 0: {metric_array[best_model_index]}')
        worst_fig.suptitle(f'{self.model_name} with {metric} closest to 0: {metric_array[worst_model_index]}')
        #best_fig.figsize([16,9]) 
        #worst_fig.figsize([16,9]) 
        
        
        best_fig.savefig(self.fig_path +f'_{metric}_sample_models_min_fit',bbox_inches='tight', pad_inches=0)
        worst_fig.savefig(self.fig_path +f'_{metric}_sample_models_max_fit',bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.close()
        plt.clf()

        #plot_uncertainty(self.best_model,X_test,y_test,toy,all_predictions=True)
        #plot_uncertainty(self.worst_model,X_test,y_test,toy,all_predictions=True)
        
        
        
    def plot_outcomes(self):
        
        
        X_ = np.arange(len(self.y_test))
    
        if not self.toy:
            index = np.argsort(self.y_test.squeeze())
                    #print(index)
        else: 
                #just use index [0,1,2,3,4,...]
                index = X_
                X_ = self.X_test[index]
        
        plt.plot(X_,self.y_test[index],'x',label='original data')

        for i in range(len(np.array(self.stats_dict['post_training']['means'])[self.outlier_keep_index])):
            mean = np.array(self.stats_dict['post_training']['means'])[self.outlier_keep_index][i][index]
            std = np.array(self.stats_dict['post_training']['stds'])[self.outlier_keep_index][i][index]

            plt.plot(X_,np.array(self.stats_dict['post_training']['means'])[self.outlier_keep_index][i][index],'x',alpha = 0.3)
            plt.errorbar(X_,mean,yerr=std,marker='x',alpha = 0.3,fmt='none')
            #plt.errorbar(X_test, y_mean[index] , yerr=y_std[index], label='unctertainty',color="purple",alpha=0.1,marker="_",uplims=True, lolims=True,fmt='none')
        plt.legend()
        
        plt.savefig(self.fig_path + 'outcome', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
        plt.close()

    
 
    def get_outlier_indices(self,threshold=3):
        
        assert len(self.stats_dict['analysis']['test_errors']) > 0, 'please run analysis first to get unbiased numbers'
        self.analysis_dict_no_outliers = {}
        errors = np.array(self.stats_dict['analysis']['test_errors'])
        
        def get_outliers(array, threshold=threshold):
            zscore = (array - array.mean())/array.std()
            return np.where(np.abs(zscore) <= threshold)
        
        
        outlier_indices = get_outliers(errors,threshold)
        
        self.outlier_keep_index =  outlier_indices[0]
    
    def _analyze(self):
        
        #purge previous outcomes
        self.stats_dict['analysis']['test_errors'] = []
        self.stats_dict['analysis']['cobeau'] = []
        self.stats_dict['analysis']['cobeau_p'] = []
        self.stats_dict['analysis']['nlpd'] = []
        
        self.stats_dict['analysis']['prior_test_errors'] = []
        self.stats_dict['analysis']['prior_cobeau'] = []
        self.stats_dict['analysis']['prior_cobeau_p'] = []
        self.stats_dict['analysis']['prior_nlpd'] = []
        
        
            
        for mean,std in zip(np.array(self.stats_dict['post_training']['means']),  np.array(self.stats_dict['post_training']['stds'])):
                self.stats_dict['analysis']['test_errors'].append(compute_error(mean.squeeze(),self.y_test.squeeze()))
                self.stats_dict['analysis']['cobeau'].append(compute_cobeau(self.y_test.squeeze(),mean.squeeze(),std.squeeze())[0])  
                self.stats_dict['analysis']['cobeau_p'].append(compute_cobeau(self.y_test.squeeze(),mean.squeeze(),std.squeeze())[1])  
                self.stats_dict['analysis']['nlpd'].append(compute_nlpd(self.y_test.squeeze(),mean.squeeze(),std.squeeze()))  
        try:
            for mean,std in zip(np.array(self.stats_dict['pre_training']['means']), np.array(self.stats_dict['pre_training']['stds'])):
                    self.stats_dict['analysis']['prior_test_errors'].append(compute_error(mean.squeeze(),self.y_test.squeeze()))
                    self.stats_dict['analysis']['prior_cobeau'].append(compute_cobeau(self.y_test.squeeze(),mean.squeeze(),std.squeeze())[0])  
                    self.stats_dict['analysis']['prior_cobeau_p'].append(compute_cobeau(self.y_test.squeeze(),mean.squeeze(),std.squeeze())[1])  
                    self.stats_dict['analysis']['prior_nlpd'].append(compute_nlpd(self.y_test.squeeze(),mean.squeeze(),std.squeeze()))
        except:
            print('model without prior')



    
    def analysis(self,return_as_dataframe=True):
        self._analyze()
        #print(self.stats_dict['analysis'])
        #errors
        self.errors = np.array(self.stats_dict['analysis']['test_errors'])[self.outlier_keep_index]
        try:
            self.prior_errors = np.array(self.stats_dict['analysis']['prior_test_errors'])[self.outlier_keep_index]
        except:
            a = 0
        print(np.mean(self.errors), np.std(self.errors))

        #cobeau
        self.cobeau = np.array(self.stats_dict['analysis']['cobeau'])[self.outlier_keep_index]
        try:
            self.prior_cobeau = np.array(self.stats_dict['analysis']['prior_cobeau'])[self.outlier_keep_index]
        except:
            a = 0

        print(np.mean(self.cobeau), np.std(self.cobeau))

        #p - values cobeau
        self.p_val = np.array(self.stats_dict['analysis']['cobeau_p'])[self.outlier_keep_index]
        try:
            self.prior_p_val = np.array(self.stats_dict['analysis']['prior_cobeau_p'])[self.outlier_keep_index]
        except:
            a = 0

        print(np.mean(self.p_val), np.std(self.p_val))

    
        #nlpd
        self.nlpd = np.array(self.stats_dict['analysis']['nlpd'])[self.outlier_keep_index]
        try:
            self.prior_nlpd = np.array(self.stats_dict['analysis']['prior_nlpd'])[self.outlier_keep_index]
        except:
            a = 0
        print(np.mean(self.nlpd), np.std(self.nlpd))
        
        if return_as_dataframe:
            return pd.DataFrame.from_dict({'nlpd':self.nlpd,'errors':self.errors,'cobeau':self.cobeau,'cobeau_p_vals':self.p_val})
    
    def _create_comparisson_values(self):
        self.y_original = self.experiment.generator_function(self.X_test)

        self.original_function_error = compute_error(self.y_test,self.y_original)

        self.original_function_nlpd = compute_nlpd(self.y_test,self.y_original, np.ones(self.y_test.shape) *0.3)

        self.y_stupid_mean = np.ones(self.y_test.shape)* np.mean(self.y_train.squeeze())
        self.y_stupid_std = np.ones(self.y_test.shape)* np.std(self.y_train.squeeze())

        self.stupid_function_error = compute_error(self.y_test,self.y_stupid_mean)

        self.stupid_function_nlpd = compute_nlpd(self.y_test,self.y_stupid_mean,self.y_stupid_std)
        

        
    def plot_distribution_of_metrics(self):
        self.analysis()
        plt.figure()
        try:
            self._create_comparisson_values()
            
            sns.distplot(self.errors,label=f'distribution of errors',norm_hist =False)
            try:
                plt.axvline(np.mean(self.prior_error), 0,17,c='green',label=f'average prior error',linestyle='.')
            except:
                a =0

            if self.toy:
                plt.axvline(self.original_function_error, 0,17,c='green',label=f'perfect model error',linestyle=':')
            # check if the dict contains pre-training values
            
#              self.stats_dict['pre_training']['means'].append(mean)
#                 self.stats_dict['pre_training']['stds']
#             if len(self.stats_dict['pre_training']['means') > 0:
#                 #plt.axvline(compute_error(self,y_test))
#                                                    pass
            plt.axvline(self.stupid_function_error, 0,17,c='red',label=f'dumb model error',linestyle='--')
            plt.legend()
            #plt.show()
            plt.savefig(self.fig_path+ 'errors', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)

            
            plt.close()

            

        
        except Exception as e:
            print('no comparisson possible for error')
            print(e)
            sns.distplot(self.errors,label=f'distribution of errors',norm_hist =False)
            try:
                plt.axvline(np.mean(self.prior_error), 0,17,c='green',label=f'prior error',linestyle=':')
            except:
                a = 0

            plt.legend()
            #plt.show()
            plt.savefig(self.fig_path + 'errors', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
            plt.close()

            plt.figure()

        try:
            sns.distplot(self.nlpd,label=f'distribution of nlpd',norm_hist =False)
            if self.toy:
                plt.axvline(self.original_function_nlpd, 0,17,c='green',label=f'perfect model nlpd',linestyle=':')
            try:
                plt.axvline(np.mean(self.prior_nlpd), 0,17,c='green',label=f'average prior nlpd',linestyle='.')
            except:
                a = 0

            plt.axvline(self.stupid_function_nlpd, 0,17,c='red',label=f'dumb model nlpd',linestyle='--')
            plt.axvline(np.mean(self.errors),0,17,label='average errors of the model')
            plt.legend()
            #plt.show()
            plt.savefig(self.fig_path + 'nlpd', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
    
        except Exception as e:
            print('no comparisson possible for nlpd')
            print(e)
            sns.distplot(self.nlpd,label=f'distribution of nlpd',norm_hist =False)
            #plt.show()
            try:
                plt.axvline(np.mean(self.prior_nlpd), 0,17,c='green',label=f'prior nlpd',linestyle='.')
            except:
                a = 0
            plt.legend()

            plt.savefig(self.fig_path + 'nlpd', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
            plt.close()

            plt.figure()

        #data = norm.rvs(5,0.4,size=1000) # you can use a pandas series or a list if you want
        sns.distplot(self.cobeau,label=f'distribution of cobeaus',norm_hist =False)
        try:
            plt.axvline(np.mean(self.prior_cobeau), 0,17,c='green',label=f'average prior cobeau',linestyle='.')
        except:
            a = 0
        #plt.axvline(self.stupid_function_nlpd, 0,17,c='red',label=f'dumb model error')
        plt.legend()
        #plt.show()
        plt.savefig(self.fig_path + 'cobeau', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    
        runtime_metrics = self.stats_dict['training']['training_times']
        plt.close()

        plt.figure()

        sns.distplot(runtime_metrics,label=f'distribution of runtimes',norm_hist =False)
        plt.legend()
        #plt.show()
        plt.savefig(self.fig_path + 'runtimes', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
        plt.close()
    
    
    def plot_outlier_models(self):
        iters = 100
        l2 = 1
        n_std = 4

        X_ = np.arange(len(self.y_test))
    
        if not self.toy:
            index = np.argsort(self.y_test.squeeze())
                    #print(index)
        else: 
                #just use index [0,1,2,3,4,...]
                index = X_
                X_ = self.X_test[index]
                
                
        outlier_index = list(set(list(range(self.experiment.num_experiments))) - set(self.outlier_keep_index.tolist()))
        
        num_models = len(outlier_index)

        
        print(f'number of outliers: {num_models}')
        fig, axs = plt.subplots(max(num_models,2))
        fig.suptitle('Vertically stacked models removed due to being outliers')
        fig.set_size_inches(18.5, num_models*5)
        
        for i,model in enumerate(np.array(self.stats_dict['models'])[outlier_index]):
            y_mean, y_std = model.uncertainty_function(self.X_test, iters, l2=l2)


            axs[i].plot(X_, y_mean[index], marker='x',ls='None', color="purple", label="mean")
            axs[i].plot(X_, self.y_test[index], marker='x',ls='None', color="red", label="original data")




