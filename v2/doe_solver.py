import torch
import numpy as np
from scipy.interpolate import Akima1DInterpolator
from tqdm import tqdm
from wavenn import MLP_Regression, mean, std
from ffact import generate_factorial_table

# I still have a nvidia gpu how about you
device = "cuda" if torch.cuda.is_available() else "cpu"

# load pre-trained model weights
model = MLP_Regression()
model_path = r'D:\LHR\wave.nn\v2\wave_nn.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# moving the model to the gpu of it's available
model.to(device)

# define parameters of the full factorial table  
shlb = 6 # secondary header upper bound 
shub = 10 # secondary header lower bound 
shs = 0.25 # secondar haeder step size
phlb = 355 # primary header lower bound 
phub = 455 # primary header upper bound
phs = 10 # primary header step size 
irlb = 9 # intake runners lower bound
irub = 14 # intake runners upper bound
irs = 0.25 # intake runners step size
pvlb = 2 # plenum volume lower bound 
pvub = 4 # plenum volume upper bound 
pvs = 0.25 # plenum volume step size 

# create full factorial table
factorial_table = generate_factorial_table(shlb, shub, shs, phlb, phub, phs, irlb, irub, irs, pvlb, pvub, pvs)

# convert factorial_table to a NumPy array
doe = np.array(factorial_table)

# get the dimensions of doe for loop bounds
shape_doe = doe.shape

# creates a list of rpm values for full factorial sweep depending on range and step size 
rpm = np.arange(4000, 13001, 500)

# finer rpm values for makima interpolation
rpm_plt = np.arange(4000, 13001, 50)   

total_iterations = shape_doe[0] * len(rpm) # total number of cases that will be solved for tqdm

torque_curves = np.zeros([len(rpm_plt), shape_doe[1]+6, doe.shape[0]]) # initializing the torque curves tensor 


# solve the full factorial doe 
with tqdm(total=total_iterations, desc='Solving for {} cases'.format(total_iterations)) as pbar:
    # iterate over engine package parameters 
    for i in range(doe.shape[0]): #doe.shape[0]
        temp = []
        tm = []
        
        # iterate over rpm range, this is the loop that makes the torque curve
        for j in range(len(rpm)):
            inputs_non_normalized = doe[i,:].copy()  # copy of non-normalized inputs for later formatting
            inputs = np.append(doe[i,:], rpm[j])  # combine engine package parameters and rpm
            inputs[:] = (inputs[:] - mean) / std  # normalize inputs based on training data normalization
            inputs = torch.tensor(inputs, dtype=torch.float32, device=device) # convert to pytorch tensor
            t = model(inputs)  # predict torque value 
            tm.append(t.item())  # append torque value 
            
            pbar.update(1)  # update progress bar
        
        # smooth predicted torque using Makima interpolator
        tm_smooth = np.array(Akima1DInterpolator(rpm, tm, method="makima")(rpm_plt))
        
        params = np.tile(inputs_non_normalized, (len(tm_smooth), 1))  # stack torque curve parameters because formatting 
        temp = np.concatenate((params, rpm_plt[:, np.newaxis], tm_smooth[:, np.newaxis]), axis=1)  # concatenate rpm and torque arrays to parameters to create a completed torque curve

        torque_curves[:,0:6,i] = temp  # populate current layer with the completed torque curve

# save completed torque curve tensor for selecting optimized parameters 
np.save('torque_curves.npy', torque_curves, allow_pickle=True)