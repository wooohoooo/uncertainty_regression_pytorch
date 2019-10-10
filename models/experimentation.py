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
import seaborn as sns


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
                                  'final_losses':[]
                                 },

                      'analysis':{'test_errors':[],
                                  'cobeau':[],
                                  'cobeau_p':[],
                                  'nlpd':[]
                                 },
                      'models': []
                     }
        
        model_string = f'{model_type}'
        index_start = model_string.find('.models.')+len('.models.')
        index_stop = model_string.find("'>")
        self.model_name = model_string[index_start:index_stop]

        

    def run_experiment(self):
        for i in range(self.num_experiments):

            np.random.seed(self.seed + i)
            torch.manual_seed(self.seed + i)
            try:
                model = self.model_type(self.toy,self.output_dims,save_path=f'experiments/experiment_{i}_{self.model_name}/')
            except Exception as e:
                print(e)
                try:
                    model = self.model_type(self.toy,self.output_dims,dataset_lenght=self.X_train.shape[0])
                except Exception as e:
                    print(e)
                    model = self.model_type(self.toy,self.output_dims)


            losslist = []
            try:
                mean, std, outcomes = model.uncertainty_function(self.X_test, iters, l2=l2,all_predictions=True)
                self.stats_dict['pre_training']['means'].append(mean)
                self.stats_dict['pre_training']['stds'].append(std)
                self.stats_dict['pre_training']['outcomes'].append(outcomes)
            except Exception as e:
                print('pre-training information not available for methods that rely on ensembling through time')

            for i in range(self.num_epochs):
                loss = model.fit_model(self.X_train, self.y_train)
                losslist.append(loss)

                if i == self.num_epochs - 1:
                    try:
                        self.stats_dict['training']['final_losses'].append(loss.data.numpy())
                    except:
                        self.stats_dict['training']['final_losses'].append(loss)

            self.stats_dict['training']['losses'].append(losslist)
            plt.plot(losslist)

            mean, std, outcomes = model.uncertainty_function(self.X_test, iters, l2=l2,all_predictions=True)

            self.stats_dict['models'].append(model)

            self.stats_dict['post_training']['means'].append(mean)
            self.stats_dict['post_training']['stds'].append(std)
            self.stats_dict['post_training']['outcomes'].append(outcomes)

            self.stats_dict['analysis']['test_errors'].append(compute_error(mean.squeeze(),self.y_test.squeeze()))
            self.stats_dict['analysis']['cobeau'].append(compute_cobeau(self.y_test.squeeze(),mean.squeeze(),std.squeeze())[0])  
            self.stats_dict['analysis']['cobeau_p'].append(compute_cobeau(self.y_test.squeeze(),mean.squeeze(),std.squeeze())[1])  
            self.stats_dict['analysis']['nlpd'].append(compute_nlpd(self.y_test.squeeze(),mean.squeeze(),std.squeeze()))  
    
    def plot_models(self,metric='test_errors'):
        
        assert len(self.stats_dict['analysis'][metric]) == len(self.stats_dict['models']), 'number of models and metrics isnt the same'
        
        best_model_index = np.argmin(self.stats_dict['analysis'][metric])
        worst_model_index = np.argmax(self.stats_dict['analysis'][metric])
        
        print(best_model_index,worst_model_index)
        
        best_model = self.stats_dict['models'][best_model_index]
        worst_model = self.stats_dict['models'][worst_model_index]
        
        plot_uncertainty(best_model,self.X_test,self.y_test,self.toy,all_predictions=True)
        plot_uncertainty(worst_model,self.X_test,self.y_test,self.toy,all_predictions=True)
        #plot_uncertainty(self.best_model,X_test,y_test,toy,all_predictions=True)
        #plot_uncertainty(self.worst_model,X_test,y_test,toy,all_predictions=True)
        
        
        
    def plot_outcomes(self):
        plt.plot(self.X_test,self.y_test,'x',label='original data')

        for i in range(len(self.stats_dict['post_training']['means'])):
            mean = self.stats_dict['post_training']['means'][i]
            std = self.stats_dict['post_training']['stds'][i]

            plt.plot(self.X_test,self.stats_dict['post_training']['means'][i],'x',alpha = 0.3)
            plt.errorbar(self.X_test,mean,yerr=std,marker='x',alpha = 0.3,fmt='none')
            #plt.errorbar(X_test, y_mean[index] , yerr=y_std[index], label='unctertainty',color="purple",alpha=0.1,marker="_",uplims=True, lolims=True,fmt='none')
    
    
    def analysis(self):
        
        #errors
        self.errors = self.stats_dict['analysis']['test_errors']
        print(np.mean(self.errors), np.std(self.errors))

        #cobeau
        self.cobeau = self.stats_dict['analysis']['cobeau']
        print(np.mean(self.cobeau), np.std(self.cobeau))

        #p - values cobeau
        self.p_val = self.stats_dict['analysis']['cobeau_p']
        print(np.mean(self.p_val), np.std(self.p_val))

    
        #nlpd
        self.nlpd = self.stats_dict['analysis']['nlpd']
        print(np.mean(self.nlpd), np.std(self.nlpd))
    
    def _create_comparisson_values(self):
        self.y_original = self.generator_function(self.X_test)

        self.original_function_error = compute_error(self.y_test,self.y_original)

        self.original_function_nlpd = compute_nlpd(self.y_test,self.y_original, np.ones(self.y_test.shape) *0.3)

        self.y_stupid_mean = np.ones(self.y_test.shape)* np.mean(self.y_train.squeeze())
        self.y_stupid_std = np.ones(self.y_test.shape)* np.std(self.y_train.squeeze())

        self.stupid_function_error = compute_error(self.y_test,self.y_stupid_mean)

        self.stupid_function_nlpd = compute_nlpd(self.y_test,self.y_stupid_mean,self.y_stupid_std)
        
        
    def plot_distribution_of_metrics(self):
        self.analysis()
        try:
            self._create_comparisson_values()
            
            sns.distplot(self.errors,label=f'distribution of errors',norm_hist =False)
            plt.axvline(self.original_function_error, 0,17,c='green',label=f'perfect model error',linestyle=':')
            plt.axvline(self.stupid_function_error, 0,17,c='red',label=f'dumb model error',linestyle='--')
            plt.legend()
            plt.show()
            

            

        
        except Exception as e:
            print('no comparisson possible for error')
            print(e)
            sns.distplot(self.errors,label=f'distribution of errors',norm_hist =False)
            plt.legend()
            plt.show()
        
        try:
            sns.distplot(self.nlpd,label=f'distribution of nlpd',norm_hist =False)
            plt.axvline(self.original_function_nlpd, 0,17,c='green',label=f'perfect model nlpd',linestyle=':')
            plt.axvline(self.stupid_function_nlpd, 0,17,c='red',label=f'dumb model nlpd',linestyle='--')
            plt.legend()
            plt.show()
            
        except Exception as e:
            print('no comparisson possible for nlpd')
            print(e)
            sns.distplot(self.nlpd,label=f'distribution of nlpd',norm_hist =False)
            plt.legend()
            plt.show()


        #data = norm.rvs(5,0.4,size=1000) # you can use a pandas series or a list if you want
        sns.distplot(self.cobeau,label=f'distribution of cobeaus',norm_hist =False)
        #plt.axvline(self.original_function_nlpd, 0,17,c='green',label=f'perfect model error')
        #plt.axvline(self.stupid_function_nlpd, 0,17,c='red',label=f'dumb model error')
        plt.legend()
        plt.show()

        