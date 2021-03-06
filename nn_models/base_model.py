import torch





def get_toy_model(n_dims_input, non_linearity,dropout_p):
    """original was 100,100,10; LeakyReLU"""
#     return torch.nn.Sequential(
#                     torch.nn.Linear(n_dims_input,100),
#                     non_linearity(),

#                     torch.nn.Linear(100,100),
#                     torch.nn.Dropout(p=dropout_p),

#                     non_linearity(),

#                     torch.nn.Linear(100,1)
#                 )
    #original
#     return torch.nn.Sequential(
#                     torch.nn.Linear(n_dims_input,20),
#                     non_linearity(),
#                             torch.nn.Dropout(p=dropout_p),

#                     torch.nn.Linear(20,20),
#                     non_linearity(),
#                             torch.nn.Dropout(p=dropout_p),

#                     torch.nn.Linear(20, 10),
#                     non_linearity(),
#                             torch.nn.Dropout(p=dropout_p),

#                     torch.nn.Linear(10, 10),
#                     non_linearity(),
#                     torch.nn.Dropout(p=dropout_p),
#                     torch.nn.Linear(10,1)
#                 )
    
#     #gridsearched
#     #non_linearity = torch.nn.Tanh
    if non_linearity == torch.nn.ReLU:
        return torch.nn.Sequential(
                        torch.nn.Linear(n_dims_input,500),
            
                        non_linearity(),
                                torch.nn.Dropout(p=dropout_p),

                        torch.nn.Linear(500,300),
                        non_linearity(),
                                torch.nn.Dropout(p=dropout_p),

                        torch.nn.Linear(300, 200),
                        non_linearity(),
                                torch.nn.Dropout(p=dropout_p),

                        torch.nn.Linear(200, 10),
                        non_linearity(),
                                torch.nn.Dropout(p=dropout_p),

                        torch.nn.Dropout(p=dropout_p),
                        torch.nn.Linear(10,1)
                    )
#[(500,300),(300,200),(200,10)]

    #non_linearity = torch.nn.Tanh
    return torch.nn.Sequential(
                    torch.nn.Linear(n_dims_input,100),
                            torch.nn.Dropout(p=dropout_p),

                    non_linearity(),

                    torch.nn.Linear(100,100),
                    torch.nn.Dropout(p=dropout_p),

                    non_linearity(),

                    torch.nn.Linear(100, 10),
                            torch.nn.Dropout(p=dropout_p),

                    non_linearity(),


                    torch.nn.Linear(10,1)
                )

def get_kaggle_model(n_dims_input, non_linearity,dropout_p):
    """original was 500,500,15; Tanh"""
    
    #original
#     torch.nn.Sequential(
#                     torch.nn.Linear(n_dims_input,100),
#                     non_linearity(),
#                     torch.nn.Linear(100,50),
#                     non_linearity(),            
#                     torch.nn.Linear(50, 50),
#                     non_linearity(),
#                     torch.nn.Dropout(p=self.dropout_p),
#                     torch.nn.Linear(50, 15),
#                     non_linearity(),
#                     torch.nn.Dropout(p=self.dropout_p),
#                     torch.nn.Linear(15,1)
#                 )
    #non_linearity = torch.nn.Tanh
    return torch.nn.Sequential(
                    torch.nn.Linear(n_dims_input,500),
                    non_linearity(),
                            torch.nn.Dropout(p=dropout_p),

                    torch.nn.Linear(500,500),
                    non_linearity(),                    torch.nn.Dropout(p=dropout_p),

                    torch.nn.Linear(500, 15),
                    non_linearity(),
                    torch.nn.Dropout(p=dropout_p),
                    torch.nn.Linear(15,1)
                )






class SimpleModel(torch.nn.Module):
    """base NN model used in ensembles"""
    def __init__(self,toy,n_dims_input,p=0.00, decay=0.005, non_linearity=torch.nn.LeakyReLU,model_provided = False):
        super(SimpleModel, self).__init__()
        self.dropout_p = p
        self.decay = decay
        self.criterion = torch.nn.MSELoss()
        self.non_linearity = non_linearity
        if not model_provided:
            if toy:
                self.f = get_toy_model(n_dims_input, non_linearity,self.dropout_p)

            else:
                self.f = get_kaggle_model(n_dims_input, non_linearity,self.dropout_p)
        else:
            self.f = model_provided

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            weight_decay=self.decay) 
        
    def forward(self, X):
        #self.f = self.f.train()
        X = torch.autograd.Variable(torch.Tensor(X), requires_grad=False)
        return self.f(X)

    
    
    
    def fit_model(self, X_obs,y_obs):
        #self.f = self.f.eval()
        y = torch.autograd.Variable(torch.Tensor(y_obs), requires_grad=False)
        y_pred = self(X_obs)
        self.optimizer.zero_grad()
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss
