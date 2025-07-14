import numpy as np

def get_max_torque(torque_curve):
    shape = torque_curve.shape
    max_torque = np.max(torque_curve[:, 5])  # finding the max torque of the curve
    max_tourque_vec = np.full((shape[0], 1), max_torque)  # create a column vector of the max torque value

    return max_tourque_vec 

import numpy as np

# ------------------------------------------------------------------
# Column indices used everywhere in this project
RPM_COL     = 4     # torque_curve[:, 4]  -> rpm values
TORQUE_COL  = 5     # torque_curve[:, 5]  -> torque values
# ------------------------------------------------------------------

def get_avg_torque(torque_curve: np.ndarray,
                   low_rpm: float,
                   high_rpm: float) -> np.ndarray:
  
    rpm     = torque_curve[:, RPM_COL]
    torque  = torque_curve[:, TORQUE_COL]

    # Clamp the window to the available rpm range
    low_rpm  = max(low_rpm,  rpm.min())
    high_rpm = min(high_rpm, rpm.max())

    # Points that lie inside the requested rpm window
    mask = (rpm >= low_rpm) & (rpm <= high_rpm)

    # If the window is empty use the whole curve instead
    if mask.any():
        avg_val = torque[mask].mean()
    else:
        avg_val = torque.mean()

    # The main script expects a column vector of the same height
    return np.full((torque_curve.shape[0], 1), avg_val)


def get_smoothness(torque_curve):
    shape = torque_curve.shape    
    second_derivative = np.gradient(np.gradient(torque_curve[:,5]))  # calculate the second derivative of the torque curve 
    squared_second_derivative = np.square(second_derivative)  # calculate the squared second derivative
    smoothness = np.trapz(squared_second_derivative)  # calculate the integral of the squared second derivative
    avg_smoothness = np.sum(smoothness)/shape[0]
    smoothness_vec = np.full((shape[0], 1), avg_smoothness)  # create a column vector of the smoothness value

    return smoothness_vec

def quicksort_3d(arr, progress_bar):
    # base case: if array has less than 2 slices
    if arr.shape[2] < 2:
        return arr
    
    # sort based on final ranking value
    pivot_keys = arr[0, 9, :]
    
    # define the pivot element
    pivot_index = len(pivot_keys) // 2
    pivot_value = pivot_keys[pivot_index]
    
    # partition the slices based on the pivot value
    left_indices = pivot_keys > pivot_value
    right_indices = pivot_keys < pivot_value
    middle_indices = pivot_keys == pivot_value
    
    # Ccate arrays for each partition
    left_slices = arr[:, :, left_indices]
    right_slices = arr[:, :, right_indices]
    middle_slices = arr[:, :, middle_indices]

    # update progress bar
    progress_bar.update(1)
    
    # recursively sort the left and right partitions
    sorted_left = quicksort_3d(left_slices, progress_bar)
    sorted_right = quicksort_3d(right_slices, progress_bar)
    
    # concatenate the results
    sorted_arr = np.concatenate([sorted_left, middle_slices, sorted_right], axis=2)
    
    return sorted_arr