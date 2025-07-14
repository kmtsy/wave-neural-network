import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ranking_functions import *

# ------------------------------------------------------------------
# 1. load the data and make sure it is big enough
# ------------------------------------------------------------------
torque_curves = np.load('torque_curves.npy', allow_pickle=True)   # (N_rows, N_cols, N_curves)

REQUIRED_COLS = 10                # 0-3 geometry, 4 rpm, 5 tq, 6-8 metrics, 9 rank
if torque_curves.shape[1] < REQUIRED_COLS:
    missing = REQUIRED_COLS - torque_curves.shape[1]
    # pad with zeros on the second axis (columns)
    torque_curves = np.pad(torque_curves,
                           pad_width=((0, 0), (0, missing), (0, 0)),
                           mode='constant')
# ------------------------------------------------------------------
shape = torque_curves.shape        # (rows, cols, curves)

# ------------------------------------------------------------------
# 2. compute the three ranking parameters
# ------------------------------------------------------------------
with tqdm(total=shape[2],
          desc=f'calculating ranking parameters for {shape[2]} torque curves') as pbar:
    for i in range(shape[2]):
        temp = np.concatenate(
            (get_max_torque(torque_curves[:, :, i]),          # (rows,1)
             get_avg_torque(torque_curves[:, :, i], 7000, 11000),  # (rows,1)
             get_smoothness(torque_curves[:, :, i])),         # (rows,1)
            axis=1)                                           # → (rows,3)

        torque_curves[:, 6:9, i] = temp                       # write into cols 6-8
        pbar.update(1)

# ------------------------------------------------------------------
# 3. normalise the three metrics
# ------------------------------------------------------------------
torque_curves[:, 6, :] = torque_curves[1, 6, :] / np.max(torque_curves[1, 6, :])
torque_curves[:, 7, :] = torque_curves[1, 7, :] / np.max(torque_curves[1, 7, :])
torque_curves[:, 8, :] = torque_curves[1, 8, :] / np.max(torque_curves[1, 8, :])

# ------------------------------------------------------------------
# 4. final rank
# ------------------------------------------------------------------
with tqdm(total=shape[2],
          desc=f'calculating final rank for {shape[2]} torque curves') as pbar:
    for i in range(shape[2]):
        rank_val = (0.05 * torque_curves[1, 6, i] +
                    0.7 * torque_curves[1, 7, i] +
                    0.25 * torque_curves[1, 8, i])
        torque_curves[:, 9, i] = rank_val                     # broadcast through rows
        pbar.update(1)

# ------------------------------------------------------------------
# 5. sort, plot, …  (unchanged)
# ------------------------------------------------------------------
with tqdm(total=shape[2] * np.log10(shape[2]),
          desc=f'sorting {shape[2]} torque curves') as pbar:
    torque_curves = quicksort_3d(torque_curves, progress_bar=pbar)

# plot the top X torque curves and their parameters
num_curves = 100
label = [''] * num_curves

plt.figure()

for i in range(num_curves):
    label[i] = (f'sec. header len.= {round(torque_curves[0, 0, i], 3)}in., ' 
                f'header len.= {round(torque_curves[0, 1, i], 3)}mm., ' 
                f'runner len.= {round(torque_curves[0, 2, i], 3)}in., ' 
                f'plenum vol. = {round(torque_curves[0, 3, i], 3)}L')
    
    plt.plot(torque_curves[:,4,i], torque_curves[:,5,i])

plt.legend(label)
plt.title(f'Top {num_curves} torque curves')
plt.xlabel('rpm')
plt.ylabel('torque (Nm)')
plt.grid(True)
plt.show()