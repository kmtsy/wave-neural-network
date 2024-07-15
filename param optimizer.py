import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
import csv
from tqdm import tqdm
from ranking_functions import *


# I still have a nvidia gpu how about you
device = "cuda" if torch.cuda.is_available() else "cpu"

# reading in the training data
bs = pd.read_csv('wave_farm_data_fixed.csv')
training_data = bs.to_numpy()

# compute the mean and standard deviation of the training data
mean = training_data[:, 0:5].mean(axis=0)
std = training_data[:, 0:5].std(axis=0)

# yes I am having to remake the entire model class here because if I try to import it from wavenn.py it starts training the model and yes I had to ctr c+v all the normalization code for the same reason
input = 5
hidden_1 = 100
hidden_2 = 250
hidden_3 = 500
hidden_4 = 500
hidden_5 = 250
hidden_6 = 100
output = 1

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

# load pre-trained model weights
model = MLP_Regression(input, hidden_1, hidden_2, hidden_3, hidden_4, hidden_5, hidden_6, output)
model_path = r'D:\LHR\wave.nn\wave_nn.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# moving the model to the gpu of it's available
model.to(device)

# creates a list of rpm values for full factorial sweep depending on range and step size 
rpm = np.arange(4000, 13001, 500)

# initialize full factorial table
factorial_table = []

# load the full factorial table from the CSV file
csv_file = 'factorial_table.csv'
with open(csv_file, 'r', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)  # skip header
    for row in reader:
        factorial_table.append([float(value) for value in row])

# convert factorial_table to a NumPy array
doe = np.array(factorial_table)

# get the dimensions of doe for loop bounds
shape_doe = doe.shape
# finer rpm values for makima
rpm_plt = np.arange(4000, 13001, 50)   

torque_curves = [] # initializing the torque curves tensor 
total_iterations = shape_doe[0] * len(rpm) # total number of cases that will be solved for tqdm


# solve for the cases from the full factorical case study 
with tqdm(total=total_iterations, desc='Solving for {} cases'.format(total_iterations)) as pbar:
    # iterate over engine package parameters 
    for i in range(2000): #doe.shape[0]
        temp = []
        tm = []
        
        # iterate over rpm range, this is the loop that makes the torque curve
        for j in range(len(rpm)):
            inputs = np.array(doe[i,:])  # import torque curve parameters
            inputs_non_normalized = inputs.copy()  # copy of non-normalized inputs for later formatting
            inputs = np.append(inputs, rpm[j])  # append rpm value
            inputs[:] = (inputs[:] - mean) / std  # normalize inputs based on training data normalization
            inputs = torch.tensor(inputs, dtype=torch.float32, device=device) # convert to pytorch tensor
            t = model(inputs)  # predict torque value 
            tm.append(t.item())  # append torque value to tm list
            
            pbar.update(1)  # update progress bar
        
        # smooth predicted torque using Makima interpolator
        tm_smooth = np.array(Akima1DInterpolator(rpm, tm, method="makima")(rpm_plt))
        
        # temp array for storing the completed smoothed torque curve
        params = np.tile(inputs_non_normalized, (len(tm_smooth), 1))  # stack torque curve parameters because formatting 
        temp = np.concatenate((params, rpm_plt[:, np.newaxis], tm_smooth[:, np.newaxis]), axis=1)  # concatenate rpm and torque arrays to parameters

        if i == 0:
            torque_curves = temp.copy()  # this is the first torque curve 
        elif i == 1:
            torque_curves = np.stack((torque_curves, temp), axis=2)  # second has to stacked like this 
        else:
            torque_curves = np.concatenate((torque_curves,temp[..., np.newaxis]), axis=2)  # now subsequent torque curves can be stacked infinitly 

shape = torque_curves.shape
# loop through every torque curve layer and calculate and assign ranking values, yes this creates another array of like 80 million values, yes it is incredibly memory inefficient 
with tqdm(total=shape[2], desc='calculating ranking parameters for {} torque curves'.format(shape[2])) as pbar:
    for i in range(shape[2]):
        test = np.concatenate((torque_curves[:,:,i], get_max_torque(torque_curves[:,:,i]), get_avg_torque(torque_curves[:,:,i]), get_smoothness(torque_curves[:,:,i])), axis=1) # calculate and concatenate ranking parameters
        if i == 0:
            temp_stack = test.copy()  # create an entire new array that contains torques and there ranking parameter values
        elif i == 1:
            temp_stack = np.stack((temp_stack, test), axis=2) # second needs to stacked like this for some reason
        else:
            temp_stack = np.concatenate((temp_stack, test[..., np.newaxis]), axis=2) # concatinate indefinitly 
        pbar.update(1)

# normalize all the ranking parameters based on the max value so all values are between 0 and 1
temp_stack[:,6,:] = temp_stack[1,6,:]/np.max(temp_stack[1,6,:])
temp_stack[:,7,:] = temp_stack[1,7,:]/np.max(temp_stack[1,7,:])
temp_stack[:,8,:] = np.max(temp_stack[1,8,:])/temp_stack[1,8,:]

# looping through every torque curve layer and calculating it's final rank and guess what, we're making a new 100 millon element array! that makes 3 ~100 million arrays now
with tqdm(total=shape[2], desc='calculating final rank for {} torque curves'.format(shape[2])) as pbar:
    for i in range(shape[2]):
        rank = 0.2 * temp_stack[1,6,i] + 0.7 * temp_stack[1,7,i] + 0.1 * temp_stack[1,8,i] # apply ranking parameters weights
        rank_vec = np.full((shape[0], 1), rank)
        temp_ = np.concatenate((temp_stack[:,:,i], rank_vec), axis = 1) # concatenate the ranking value to the torque curve layer

        if i == 0:
            final_torque_curves = temp_.copy()  # this is the first torque curve and creates a NEW array
        elif i == 1:
            final_torque_curves = np.stack((final_torque_curves, temp_), axis=2)  # second has to stacked like this 
        else:
            final_torque_curves = np.concatenate((final_torque_curves, temp_[..., np.newaxis]), axis=2)  # now subsequent torque curves can be stacked infinitly 
        pbar.update(1)

with tqdm(total=shape[2] * np.log10(shape[2]), desc='sorting {} torque curves'.format(shape[2])) as pbar:
    final_torque_curves = quicksort_3d(final_torque_curves, progress_bar=pbar) # sort everything (this took longer than everything else above for some reason)

# plot the top 5 torque curves and their parameters
num_curves = 5
label = [''] * num_curves

plt.figure()

for i in range(num_curves):
    label[i] = (f'sec. header len.= {round(torque_curves[0, 0, i], 3)}in., ' 
                f'header len.= {round(torque_curves[0, 1, i], 3)}mm., ' 
                f'runner len.= {round(torque_curves[0, 2, i], 3)}in., ' 
                f'plenum vol. = {round(torque_curves[0, 3, i], 3)}L')
    
    plt.plot(final_torque_curves[:,4,i], final_torque_curves[:,5,i])

plt.legend(label)
plt.title('Top 5 torque curves')
plt.xlabel('rpm')
plt.ylabel('torque (Nm)')
plt.grid(True)
plt.show()