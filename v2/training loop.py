import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wavenn import MLP_Regression, EarlyStopping

# I have a nvidia gpu how about you
device = "cuda" if torch.cuda.is_available() else "cpu"


########################################## preping training data ##########################################

# reading in the training data
bs = pd.read_csv('wave_farm_data_fixed.csv')
training_data = bs.to_numpy()

# compute the mean and standard deviation of the training data
mean = training_data[:, 0:5].mean(axis=0)
std = training_data[:, 0:5].std(axis=0)

# normalize the training data
training_data[:, 0:5] = (training_data[:, 0:5] - mean) / std

# convert the normalized data to a PyTorch tensor
training_data = torch.tensor(training_data, dtype=torch.float32, device=device)
x, y = training_data[:, 0:5], training_data[:, 5]

# reading in torque curve for seeing how wrong the model is (very (not anymore haha (nope it can't generalize))) 
bs = pd.read_csv('curve_1.csv')
test_t = bs.to_numpy()

# normalize the test data using the training data statistics
test_t[:, 0:5] = (test_t[:, 0:5] - mean) / std

# convert the normalized test data to a PyTorch tensor
test_t = torch.tensor(test_t, dtype=torch.float32, device=device)
parm, rpm, ta = test_t[:, 0:5], test_t[:, 4], test_t[:, 5]

# splitting data into training and testing data based on an 80-20 split
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

############################################# training the model #############################################


# initializing the model
wave_nn = MLP_Regression()

# moving the model to the gpu of it's available
wave_nn.to(device)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(wave_nn.parameters(), lr=0.001,betas=(0.9,0.999)) # adam optimizer, lr is learning rate, betas is momentum params

# initialize the early stopping function
es = EarlyStopping()

# initializing loop vars/arrs
epochs = 0 
done = False
tloss_history = [] # for plotting testing loss later
vloss_history = [] # for plotting validation loss later

#training loop
while epochs < 1000 and not done: # either it converges or hits the max epoch limit
    epochs += 1
    steps = list(enumerate(dataloader_train))
    pbar = tqdm.tqdm(steps)

    # put the model in training mode 
    wave_nn.train()

    for i, (x_batch, y_batch) in pbar:
        # forward pass
        y_batch_pred = wave_nn(x_batch)
        
        # compute loss
        loss = loss_fn(y_batch_pred.squeeze(), y_batch)
        
        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute current loss
        loss_item = loss.item()
        current = (i + 1) * len(x_batch)
        
        # update progress bar
        if i == len(steps) - 1:
            # switch to evaluation mode for validation
            wave_nn.eval()
            with torch.inference_mode():
                pred = wave_nn(x_test).flatten()
                vloss = loss_fn(pred, y_test)
                tloss_history.append(loss_item)
                vloss_history.append(vloss.cpu().item())
                if es(wave_nn, vloss): # has convergence has been met
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