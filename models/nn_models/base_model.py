import torch

class SimpleModel(torch.nn.Module):
    """base NN model used in ensembles"""
    def __init__(self,toy,n_dims_input,p=0.05, decay=0.001, non_linearity=torch.nn.LeakyReLU):
        super(SimpleModel, self).__init__()
        self.dropout_p = p
        self.decay = decay
        self.criterion = torch.nn.MSELoss()

        if toy:
            self.f = torch.nn.Sequential(
                torch.nn.Linear(n_dims_input,20),
                non_linearity(),
                torch.nn.Linear(20,20),
                non_linearity(),
                torch.nn.Linear(20, 10),
                non_linearity(),
                torch.nn.Linear(10, 10),
                non_linearity(),
                torch.nn.Dropout(p=self.dropout_p),
                torch.nn.Linear(10,1)
            )
        else:
            self.f = torch.nn.Sequential(
                torch.nn.Linear(n_dims_input,100),
                non_linearity(),
                torch.nn.Linear(100,50),
                non_linearity(),            
                torch.nn.Linear(50, 50),
                non_linearity(),
                torch.nn.Dropout(p=self.dropout_p),
                torch.nn.Linear(50, 15),
                non_linearity(),
                torch.nn.Dropout(p=self.dropout_p),
                torch.nn.Linear(15,1)
            )

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            weight_decay=self.decay) 
        
    def forward(self, X):
        X = torch.autograd.Variable(torch.Tensor(X), requires_grad=False)
        return self.f(X)

    
    
    
    def fit_model(self, X_obs,y_obs):
        y = torch.autograd.Variable(torch.Tensor(y_obs), requires_grad=False)
        y_pred = self(X_obs)
        self.optimizer.zero_grad()
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss
