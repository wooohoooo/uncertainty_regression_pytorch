import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm, trange
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
    
        