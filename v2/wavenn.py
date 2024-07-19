from torch import nn
import pandas as pd
import copy

# WAVE NN class and convergence class 

# shape of the nn
input = 5
hidden_1 = 5000
hidden_2 = 4000
hidden_3 = 3000
hidden_4 = 2000
hidden_5 = 1000
hidden_6 = 500
output = 1

# neural network class 
class MLP_Regression(nn.Module): 
    def __init__(self):
        super(MLP_Regression, self).__init__()
        self.wave_nn = nn.Sequential(
            nn.Linear(input, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, hidden_3),
            nn.ReLU(),
            nn.Linear(hidden_3, hidden_4),
            nn.ReLU(),
            nn.Linear(hidden_4, hidden_5),
            nn.ReLU(),
            nn.Linear(hidden_5, hidden_6),
            nn.ReLU(),
            nn.Linear(hidden_6, output)
        )
    def forward(self, x):
        return (self.wave_nn(x))


# convergence conditions for training loop
class EarlyStopping():
    def __init__(self, patience=5, min_delta=1e-2, convergence = 0.5, min_loss = 1, restore_best_weights=True):
       
        # controls how many times in a row the model needs to satisfy convergence criteria
        self.patience = patience

        # controls what is the min change in the validation loss value that can be considered an improvment to the model
        self.min_delta = min_delta

        # controls how little the validation loss can vary for it to be considered converged 
        self.convergence = convergence

        # controls what is the max mean squared error loss the model is allowed to have to be considered done
        self.min_loss = min_loss

        # makes sure the best version of the model is the one that is kept
        self.restore_best_weights = restore_best_weights

        # initializing everything else
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, wave_nn, val_loss):
        
        if self.best_loss is None: # training just started, set the best loss to the current validation loss
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(wave_nn)
        elif self.best_loss - val_loss > self.min_delta: # the model has improved, update the best loss, save it as the best model and reset the convergence counter
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = copy.deepcopy(wave_nn)
        elif abs(self.best_loss - val_loss) < self.convergence and val_loss < self.min_loss: # the model didn't improve, is it because it meets convergence criteria?
            self.counter += 1
            if self.counter >= self.patience: # has the model met convergence criteria 5 times?
                self.status = f"stopped on {self.counter}"
                if self.restore_best_weights:
                    wave_nn.load_state_dict(self.best_model.state_dict())
                return True
        self.status = f"{self.counter}/{self.patience}"
        return False


# computing mean and standard deviation of data here too for easy access
bs = pd.read_csv('wave_farm_data_fixed.csv')
training_data = bs.to_numpy()
mean = training_data[:, 0:5].mean(axis=0)
std = training_data[:, 0:5].std(axis=0)