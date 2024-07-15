# wave.nn

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import Akima1DInterpolator

# I have a nvidia gpu how about you
device = "cuda" if torch.cuda.is_available() else "cpu"


# Reading in the training data
bs = pd.read_csv('wave_farm_data_fixed.csv')
training_data = bs.to_numpy()

# Compute the mean and standard deviation of the training data
mean = training_data[:, 0:5].mean(axis=0)
std = training_data[:, 0:5].std(axis=0)

# Normalize the training data
training_data[:, 0:5] = (training_data[:, 0:5] - mean) / std

# Convert the normalized data to a PyTorch tensor
training_data = torch.tensor(training_data, dtype=torch.float32, device=device)
x, y = training_data[:, 0:5], training_data[:, 5]

# reading in torque curve for seeing how wrong the model is (very (not anymore haha)) 
bs = pd.read_csv('curve_1.csv')
test_t = bs.to_numpy()

# Normalize the test data using the training data statistics
test_t[:, 0:5] = (test_t[:, 0:5] - mean) / std

# Convert the normalized test data to a PyTorch tensor
test_t = torch.tensor(test_t, dtype=torch.float32, device=device)
parm, rpm, ta = test_t[:, 0:5], test_t[:, 4], test_t[:, 5]

# Splitting data into training and testing data based on an 80-20 split
train_split = int(0.8 * len(training_data))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

# batch size for training, it's 19 as there are 19 rpm steps in 1 torque curve in this data set
batch_size = 19

# creating training batches
dataset_train = TensorDataset(x_train, y_train)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# creating testing batches
dataset_test = TensorDataset(x_test, y_test)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

# early stopping condition that basically just acts as convergence criteria for the model like in ansys 
class EarlyStopping():
    def __init__(self, patience=5, min_delta=1e-2, convergence = 0.1, min_loss = 0.5, restore_best_weights=True):
       
        # the patience value controls how many times in a row the model needs to satisfy convergence criteria
        self.patience = patience

        # the min_delta value controls what is the min change in the validation loss value that can be considered an improvment to the model
        self.min_delta = min_delta

        # the convergence value controls how little the validation loss can vary for it to be considered converged 
        self.convergence = convergence

        # the min_loss value controls what is the max mean squared error loss the model is allowed to have to be considered done
        self.min_loss = min_loss

        # restore best weights just makes sure the best version of the model is the one that is kept
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
    
        

# shape of the nn, I know it's probably too much and might cause over fitting, but it works for this data
input = 5
hidden_1 = 100
hidden_2 = 250
hidden_3 = 500
hidden_4 = 500
hidden_5 = 250
hidden_6 = 100
output = 1

# creating the nn 
class MLP_Regression(nn.Module): 
    def __init__(self, input, hidden_1, hidden_2, hidden_3, hidden_4, hidden_5, hidden_6, output):
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

# initializing the model
wave_nn = MLP_Regression(input, hidden_1, hidden_2, hidden_3, hidden_4, hidden_5, hidden_6, output)


# moving the model to the gpu of it's available
wave_nn.to(device)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(wave_nn.parameters(), lr=0.0001,betas=(0.9,0.999)) # adam optimizer, lr is learning rate, betas is momentum params

# initialize the early stopping function
es = EarlyStopping()

epochs = 0 
done = False
tloss_history = [] # for plotting testing loss later
vloss_history = [] # for plotting validation loss later

while epochs < 1000 and not done: # either it converges or hits the max epoch limit
    epochs += 1
    steps = list(enumerate(dataloader_train))
    pbar = tqdm.tqdm(steps)

    # Put the model in training mode 
    wave_nn.train()

    for i, (x_batch, y_batch) in pbar:
        # Forward pass
        y_batch_pred = wave_nn(x_batch)
        
        # Compute loss
        loss = loss_fn(y_batch_pred.squeeze(), y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute current loss
        loss_item = loss.item()
        current = (i + 1) * len(x_batch)
        
        # Update progress bar
        if i == len(steps) - 1:
            # Switch to evaluation mode for validation
            wave_nn.eval()
            with torch.inference_mode():
                pred = wave_nn(x_test).flatten()
                vloss = loss_fn(pred, y_test)
                tloss_history.append(loss_item)
                vloss_history.append(vloss.cpu().item())
                if es(wave_nn, vloss): # convergence has been met
                    done = True
                    pbar.set_description(f"epoch: {epochs}, tloss: {loss_item:.4f}, vloss: {vloss:.4f}, early stop: [{es.status}]")
                else:
                    pbar.set_description(f"epoch: {epochs}, tloss: {loss_item:.4f}, vloss: {vloss:.4f}, early stop: [{es.status}]")
        else:
            pbar.set_description(f"epoch: {epochs}, tloss: {loss_item:.4f}")


# saving the model 
torch.save(wave_nn.state_dict(), 'wave_nn.pth')

# getting the model prideicted torque for asuka engine parameters using a much finer rpm step that the model was not trained on
tm =[]
wave_nn.eval()
with torch.inference_mode():
    for i in range(len(rpm)):
     t = wave_nn(parm[i,:])
     tm.append(t.cpu())

# converting from tensors to numpy arrays
rpm = rpm.numpy(force=True)
ta = ta.numpy(force=True)
tm = np.array(tm)

# ploting the model prediction
plt.figure()
plt.plot(rpm,ta, label = 'actual torque')
plt.plot(rpm,tm, label = 'nn predicted torque')
plt.legend()
plt.show()

# ploting the model loss during training 
plt.figure()
plt.plot(tloss_history, label = 'testing loss')
plt.plot(vloss_history, label = 'validation loss')
plt.legend()
plt.show()

# getting the model prideicted torque for asuka engine parameters using a much finer rpm step that the model was not trained on
tm =[]
rpm_plt = []
rpm = np.arange(4000, 13000, 10) 
with torch.inference_mode():
    for i in range(len(rpm)):
        
        if rpm[i]%500 == 0:
            t = wave_nn(parm[i,:])
            tm.append(t.cpu())
            r = rpm[i]
            rpm_plt.append(r)

# converting from tensors to numpy arrays

ta = np.array(ta)
tm = np.array(tm)
tm = Akima1DInterpolator(rpm_plt, tm, method="makima")(rpm)

# ploting the model prediction
plt.figure()
plt.plot(rpm,ta, label = 'actual torque')
plt.plot(rpm,tm, label = 'nn predicted torque')
plt.legend()
plt.show()