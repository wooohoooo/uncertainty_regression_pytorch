import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm, trange
from pathlib import Path

N = 100


class SimpleModel(torch.nn.Module):
    def __init__(self,p=0.05, decay=0.001, non_linearity=torch.nn.LeakyReLU):
        super(SimpleModel, self).__init__()
        self.dropout_p = p
        self.decay = decay
        self.criterion = torch.nn.MSELoss()

        self.f = torch.nn.Sequential(
            torch.nn.Linear(1,20),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(20,20),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(20, 10),
            non_linearity(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(10,1)
        )
        
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            weight_decay=self.decay) 
        
    def forward(self, X):
        X = Variable(torch.Tensor(X), requires_grad=False)
        return self.f(X)
    
    def ensemble_uncertainity_estimate(self,X, iters, l2=0.005, range_fn=trange, raw_var=False):
        outputs = np.hstack([self(X[:, np.newaxis]).data.numpy() for i in range_fn(iters)])
        y_mean = outputs.mean(axis=1)
        y_variance = outputs.var(axis=1)
        if not raw_var:
            tau = l2 * (1-self.dropout_p) / (2*N*self.decay)
            y_variance += (1/tau)
        y_std = np.sqrt(y_variance) #+ (1/tau)
        return y_mean, y_std

    
    def uncertainty_function(self,X, iters, l2, range_fn=trange,raw_var=False):
        return self.ensemble_uncertainity_estimate(X=X, iters=iters, l2=l2, range_fn=trange,raw_var=raw_var)
    
    
    
    def fit_ensemble(self, X_abs, y_abs):
        losslist = []
        for model,optimizer in zip(self.models,self.optimizers):
                        
            losslist.append(self.fit_model(model, optimizer, X_abs,y_abs))
    
    
    def fit_model(self, X_obs,y_obs):
        y = Variable(torch.Tensor(y_obs[:, np.newaxis]), requires_grad=False)
        y_pred = self(X_obs[:, np.newaxis])
        self.optimizer.zero_grad()
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss

    
class SaverModel(SimpleModel):
    def __init__(self,p=0.00, decay=0.001, non_linearity=torch.nn.LeakyReLU, num_epochs_per_save=100,save_path = 'test/',n_models_to_keep=10):
        super(SaverModel, self).__init__(p=0.00, decay=0.001, non_linearity=torch.nn.LeakyReLU)
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
        
        y = Variable(torch.Tensor(y_obs[:, np.newaxis]), requires_grad=False)
        y_pred = self(X_obs[:, np.newaxis])
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
            #if len(self.model_paths) > self.n_models_to_keep:
            #    self.model_paths.pop(0)

                
  
        
        #increment current epoch
        self.current_epoch +=1
        return loss
    
    def load_saved_model(self,path):
        self.load_state_dict(torch.load(path))

        
    def uncertainty_function(self,X, iters, l2, range_fn=trange,raw_var=False):
        return self.ensemble_uncertainity_estimate(X=X, iters=iters, l2=l2, range_fn=trange)
    
    def weighted_avg_and_std(self,values, weights):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
        """
        average = np.average(values, axis=1, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values-average)**2,axis=1, weights=weights)
        return (average, np.sqrt(variance))

    def ensemble_uncertainity_estimate(self,X, iters, l2=0.005, range_fn=trange):
        outputs = []
        weights = []
        
        for i,path in enumerate(self.model_paths):
            self.load_saved_model(path)
            outputs.append(self(X[:, np.newaxis]).data.numpy())
            weights.append(i)
            
        
        outputs = np.hstack(outputs)
        
        
        
        
        #y_mean = np.average(outputs, axis=1,weights = weights )
        #y_variance = outputs.var(axis=1)
        
        y_mean, y_variance = self.weighted_avg_and_std(outputs, weights)
        #tau = l2 * (1-self.dropout_p) / (2*N*self.decay)
        #y_variance += (1/tau)
        y_std = np.sqrt(y_variance)# + (1/tau)
        return y_mean, y_std
            
           
    
class GPUModel(torch.nn.Module):
    def __init__(self,p=0.05, decay=0.001, non_linearity=torch.nn.LeakyReLU):
        super(GPUModel, self).__init__()
        self.dropout_p = torch.tensor(p, requires_grad=False)
        self.decay = torch.tensor(decay, requires_grad=False)
        self.criterion = torch.nn.MSELoss()

        self.f = torch.nn.Sequential(
            torch.nn.Linear(1,20),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(20,20),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(20, 10),
            non_linearity(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(10,1)
        )
        
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            weight_decay=self.decay) 
        
    def forward(self, X):
        X = Variable(X)
        return self.f(X)
    
    def ensemble_uncertainity_estimate(self,X, iters, l2=0.005, range_fn=range, raw_var=False):
        #input_x = X[:, np.newaxis]
        outputs = np.hstack([self(X).cpu().data.numpy() for i in range_fn(iters)])


        y_mean = outputs.mean(axis=1)
        y_variance = outputs.var(axis=1)
        if not raw_var:
            tau = l2 * (1-self.dropout_p.cpu().data.numpy()) / (2*N*self.decay.cpu().data.numpy())
            y_variance += (1/tau)
        y_std = np.sqrt(y_variance) #+ (1/tau)
        return y_mean, y_std


    
    def uncertainty_function(self,X, iters, l2, range_fn=trange,raw_var=False):
        return self.ensemble_uncertainity_estimate(X=X, iters=iters, l2=l2, range_fn=trange,raw_var=raw_var)
    
    
    
    def fit_ensemble(self, X_abs, y_abs):
        losslist = []
        for model,optimizer in zip(self.models,self.optimizers):
                        
            losslist.append(self.fit_model(model, optimizer, X_abs,y_abs))
    
    
    def fit_model(self, X_obs,y_obs):
        y_pred = self(X_obs)
        self.optimizer.zero_grad()
        loss = self.criterion(y_pred, y_obs)
        loss.backward()
        self.optimizer.step()
        return loss
    
    
class VanillaEnsemble(object):
    def __init__(self,p=0.00, decay=0.001, non_linearity=torch.nn.LeakyReLU, n_models=10, model_list=None,u_iters=100, l2=1, n_std=4, title="",dataset_lenght=None):
        #super(VanillaEnsemble,self).__init__(X_obs,y_obs,X_true,y_true,p=0.00, decay=0.001, non_linearity=torch.nn.LeakyReLU, n_models=10, model_list=None,u_iters=100, l2=1, n_std=4, title="",dataset_lenght=None)
        self.models = [SimpleModel(p,decay,non_linearity) for model in range(n_models)]
        self.optimizers = [torch.optim.Adam(
            model.parameters(),
            weight_decay=model.decay) for model in self.models]
        self.criterion = torch.nn.MSELoss()
        self.dropout_p = p
        self.decay = decay
    

    def ensemble_uncertainity_estimate(self,X, iters, l2=0.005, range_fn=trange):
        outputs = np.hstack([model(X[:, np.newaxis]).data.numpy() for model in self.models])
        y_mean = outputs.mean(axis=1)
        y_variance = outputs.var(axis=1)
        tau = l2 * (1-self.dropout_p) / (2*N*self.decay)
        y_variance += (1/tau)
        y_std = np.sqrt(y_variance)# + (1/tau)
        return y_mean, y_std
    
    
    def uncertainty_function(self,X, iters, l2, range_fn=trange):
        return self.ensemble_uncertainity_estimate(X=X, iters=iters, l2=l2, range_fn=trange)
    
    def fit_ensemble_member(self, model, optimizer, X_obs,y_obs):
        y = Variable(torch.Tensor(y_obs[:, np.newaxis]), requires_grad=False)
        y_pred = model(X_obs[:, np.newaxis])
        optimizer.zero_grad()
        loss = self.criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss
    
    def shuffle(self,X,y):
        arr = np.arange(len(y))
        np.random.shuffle(arr)
        return X[arr], y[arr]

    
    def fit_model(self, X_abs, y_abs):
        losslist = []
        for model,optimizer in zip(self.models,self.optimizers):
                        
            losslist.append(self.fit_ensemble_member(model, optimizer, X_abs,y_abs))
            
        return losslist
    
    
    
class ShuffleEnsemble(VanillaEnsemble):


    def fit_model(self, X_abs, y_abs):
        losslist = []
        for model,optimizer in zip(self.models,self.optimizers):
            
            shuffled_X, shuffled_y = self.shuffle(X_abs,y_abs)
            
            losslist.append(self.fit_ensemble_member(model, optimizer, shuffled_X, shuffled_y))
        return losslist
    
    
    
class BootstrapEnsemble(VanillaEnsemble):
    def __init__(self,p=0.00, decay=0.001, non_linearity=torch.nn.LeakyReLU, n_models=4, model_list=None,u_iters=100, l2=1, n_std=4, title="",dataset_lenght=None,bootstrap_p_positive=0.7):
        super(BootstrapEnsemble, self).__init__(p=0.00, decay=0.001, non_linearity=torch.nn.LeakyReLU, n_models=10, model_list=None,u_iters=100, l2=1, n_std=4, title="",dataset_lenght=None)
        
        self.dataset_lenght = dataset_lenght
        self.bootstrap_dataset_indices = [np.random.choice(a=[True, False], size=dataset_lenght, p=[bootstrap_p_positive, 1-bootstrap_p_positive]) for model in self.models]


    def fit_model(self, X_abs, y_abs):
        losslist = []
        for model,optimizer,indices in zip(self.models,self.optimizers,self.bootstrap_dataset_indices):
            
            X_bootstrapped, y_bootstrapped = X_abs[indices], y_abs[indices]

            
            shuffled_X, shuffled_y = self.shuffle(X_bootstrapped, y_bootstrapped)
            #shuffled_X, shuffled_y = X_bootstrapped, y_bootstrapped

            
            losslist.append(self.fit_ensemble_member(model, optimizer, shuffled_X, shuffled_y))
        return losslist
    
        