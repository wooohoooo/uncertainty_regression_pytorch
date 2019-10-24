import torch
import numpy as np
from nn_models.base_model import SimpleModel
from tqdm import tqdm, trange

class VanillaEnsemble(object):
    def __init__(self,toy,n_dims_input,p=0.00, decay=0.005, non_linearity=torch.nn.LeakyReLU, n_models=10, model_list=None,u_iters=100, l2=1, n_std=4, title="",dataset_lenght=None):

        self.models = [SimpleModel(toy,n_dims_input,p,decay,non_linearity) for model in range(n_models)]
        self.optimizers = [torch.optim.Adam(
            model.parameters(),
            weight_decay=model.decay) for model in self.models]
        self.criterion = torch.nn.MSELoss()
        self.dropout_p = p
        self.decay = decay
    
    def fit_model(self, X_abs, y_abs):
        losslist = []
        for model,optimizer in zip(self.models,self.optimizers):
                        
            losslist.append(self.fit_ensemble_member(model, optimizer, X_abs,y_abs))
            
        return losslist
    
    def fit_ensemble_member(self, model, optimizer, X_obs,y_obs):
        y = torch.autograd.Variable(torch.Tensor(y_obs), requires_grad=False)
        y_pred = model(X_obs)
        optimizer.zero_grad()
        loss = self.criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss      

    def ensemble_uncertainity_estimate(self,X, iters, l2=0.005, range_fn=trange, all_predictions = True):
        outputs = np.hstack([model(X).data.numpy() for model in self.models])
        y_mean = outputs.mean(axis=1)
        y_std = outputs.std(axis=1)
        if all_predictions:
            return y_mean, y_std, outputs
        return y_mean, y_std
    
    
    def uncertainty_function(self,X, iters, l2, range_fn=trange, all_predictions= False):
        return self.ensemble_uncertainity_estimate(X=X, iters=iters, l2=l2, range_fn=trange, all_predictions=all_predictions)
    

    
    def shuffle(self,X,y,seed=None):
        arr = np.arange(len(y))
        if seed != None:
            np.random.shuffle(arr)
        else:
            np.random.seed(seed)
            np.random.shuffle(arr)
 
        return X[arr], y[arr]

