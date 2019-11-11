from nn_models.base_model import SimpleModel
import torch
from tqdm import tqdm, trange
from pathlib import Path
import numpy as np

class SaverModel(SimpleModel):
    def __init__(self,toy ,n_dims_input,p=0.00, decay=0.005, non_linearity=torch.nn.LeakyReLU, num_epochs_per_save=100,save_path = 'dummytest/',n_models_to_keep=20):
        super(SaverModel, self).__init__(toy ,n_dims_input,p=0.00, decay=decay, non_linearity=non_linearity)
        self.num_epochs_per_save = num_epochs_per_save
        self.current_epoch = 0
        
        #make sure save path exists
        path = Path().absolute()/save_path
        path.mkdir(exist_ok=True)
        self.save_path = path
        self.n_models_to_keep = n_models_to_keep

        self.model_paths = []
        
        
        
    def fit_model(self, X_obs,y_obs):
        """saves model each X iterations, stores path in model_paths"""
        
        y = torch.autograd.Variable(torch.Tensor(y_obs), requires_grad=False)
        y_pred = self(X_obs)
        self.optimizer.zero_grad()
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()

        #if epoch number is multiple of nsave epochs
        if self.current_epoch % self.num_epochs_per_save == 0:
            #create epoch path
            epoch_path = self.save_path/f"epoch{self.current_epoch}"
            
            #save model
            torch.save(self.state_dict(),epoch_path)
            
            #append to model_path_list if less than n_models_to_keep models, else replace
            self.model_paths.append(epoch_path)
            if len(self.model_paths) > self.n_models_to_keep:
                self.model_paths.pop(0)

                
  
        
        #increment current epoch
        self.current_epoch +=1
        return loss
    
    def load_saved_model(self,path):
        self.load_state_dict(torch.load(path))

        
    def uncertainty_function(self,X, iters, l2, range_fn=trange,raw_var=False,all_predictions=False):
        return self.ensemble_uncertainity_estimate(X=X, iters=iters, l2=l2, range_fn=trange,all_predictions=all_predictions)
    
    def weighted_avg_and_std(self,values, weights):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
        """
        
        print(values.shape,np.array(weights).shape)

        average = np.average(values, axis=1, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values-average)**2,axis=0, weights=weights)
        return (average, np.sqrt(variance))

    def ensemble_uncertainity_estimate(self,X, iters, l2=0.005, range_fn=trange,all_predictions=False):
        outputs = []
        weights = []
        
        for i,path in enumerate(self.model_paths):
            self.load_saved_model(path)
            outputs.append(self(X).data.numpy())
            weights.append(i)
            
        
        outputs = np.hstack(outputs)
        
        
        
        
        y_mean = np.average(outputs, axis=1,weights = weights )
        y_variance = outputs.var(axis=1)
        
        #y_mean, y_variance = self.weighted_avg_and_std(outputs, weights)
        #tau = l2 * (1-self.dropout_p) / (2*N*self.decay)
        #y_variance += (1/tau)
        y_std = np.sqrt(y_variance)# + (1/tau)
        y_std = outputs.std(axis=1)
        if all_predictions:
            return y_mean, y_std, outputs
        return y_mean, y_std
            
        
        
class BobstrapEnsemble(SaverModel):
    def __init__(self,toy ,n_dims_input,p=0.00, decay=0.005, non_linearity=torch.nn.LeakyReLU, num_epochs_per_save=100,save_path = 'dummytestbob/',n_models_to_keep=20,bootstrap_p_positive=0.7):
        super(BobstrapEnsemble, self).__init__(toy ,n_dims_input,p=0.00, decay=decay, non_linearity=non_linearity, num_epochs_per_save=num_epochs_per_save,save_path = save_path,n_models_to_keep=n_models_to_keep)
        
        
        self.current_dataset_indices = None
        self.bootstrap_p_positive = bootstrap_p_positive

      
        
    def fit_model(self, X_obs,y_obs):
        """saves model each X iterations, stores path in model_paths"""
        
        # if the model is saved, draw a new dataset to train on, too.
        if self.current_epoch % self.num_epochs_per_save == 0:
            self.current_dataset_indices = np.random.choice(a=[True, False], size=len(y_obs), p=[self.bootstrap_p_positive, 1-self.bootstrap_p_positive])

        X_obs, y_obs = X_obs[self.current_dataset_indices], y_obs[self.current_dataset_indices]
        
        y = torch.autograd.Variable(torch.Tensor(y_obs), requires_grad=False)
        y_pred = self(X_obs)
        self.optimizer.zero_grad()
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()

        #if epoch number is multiple of nsave epochs
        if self.current_epoch % self.num_epochs_per_save == 0:
            #create epoch path
            epoch_path = self.save_path/f"epoch{self.current_epoch}"
            
            #save model
            torch.save(self.state_dict(),epoch_path)
            
            #append to model_path_list if less than n_models_to_keep models, else replace
            self.model_paths.append(epoch_path)
            if len(self.model_paths) > self.n_models_to_keep:
                self.model_paths.pop(0)

                
  
        
        #increment current epoch
        self.current_epoch +=1
        return loss
    
    
class SnapshotHybridModel(SaverModel):
    
    def ensemble_uncertainity_estimate(self,X, iters, l2=0.005, range_fn=trange,all_predictions=False):
        outputs = []
        weights = []
        
        for i,path in enumerate(self.model_paths):
            self.load_saved_model(path)
            outputs.append(self(X).data.numpy())
            weights.append(i)
            
        
        outputs = np.hstack(outputs)
        
        
        
        #mean is now imply the last snapsot
        y_mean = outputs[:,-1]#self.load_saved_model(self.model_paths[-1])(X).data.numpy()        
        y_variance = outputs.var(axis=1)
        
        y_std = outputs.std(axis=1)#np.sqrt(y_variance)# + (1/tau)
        if all_predictions:
            return y_mean, y_std, outputs
        return y_mean, y_std
    
class DropoutModel(SimpleModel):
    def __init__(self,toy ,n_dims_input,p=0.05, decay=0.005, non_linearity=torch.nn.LeakyReLU):
        super(DropoutModel, self).__init__(toy ,n_dims_input,p=p, decay=decay, non_linearity=non_linearity)

    def ensemble_uncertainity_estimate(self,X, iters, l2=0.005, range_fn=trange, raw=False,all_predictions=False):
        outputs = np.hstack([self(X).data.numpy() for i in range_fn(iters)])
        y_mean = outputs.mean(axis=1)
        y_std = outputs.std(axis=1)
#         N = len(y_std)
#         print(y_mean, y_std)
        
        
        
        
        
        
#         #from gal
#         predictive_variance = outputs.var(axis=1)
#         l=1
#         tau = l**2 * (1 - self.dropout_p) / (2 * N * self.decay)
#         predictive_variance += tau**-1
#         y_std = predictive_variance


        if all_predictions:
            return y_mean, y_std, outputs

        return y_mean, y_std
        
    
    def uncertainty_function(self,X, iters, l2, range_fn=trange,raw=False,all_predictions=False):
        return self.ensemble_uncertainity_estimate(X, iters=iters, l2=l2, range_fn=trange,raw=raw,all_predictions=all_predictions)
    
    
    
    def fit_ensemble(self, X_abs, y_abs):
        losslist = []
        for model,optimizer in zip(self.models,self.optimizers):
                        
            losslist.append(self.fit_model(model, optimizer, X_abs,y_abs))
    
        
    def fit_model(self, X_obs,y_obs):
        y = torch.autograd.Variable(torch.Tensor(y_obs), requires_grad=False)
        y_pred = self(X_obs)
        self.optimizer.zero_grad()
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss
