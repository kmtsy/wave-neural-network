import numpy as np
from tqdm import tqdm

def get_max_torque(torque_curve):
    shape = torque_curve.shape
    max_torque = np.max(torque_curve[:, 5])  # finding the max torque of the curve
    max_tourque_vec = np.full((shape[0], 1), max_torque)  # create a column vector of the max torque value

    return max_tourque_vec 

def get_avg_torque(torque_curve):
    shape = torque_curve.shape
    avg_torque = np.sum(torque_curve[:, 5])/shape[0]  # finding the avg torque of the curve
    avg_tourque_vec = np.full((shape[0], 1), avg_torque)  # create a column vector of the avg torque value

    return avg_tourque_vec

def get_smoothness(torque_curve):
    shape = torque_curve.shape    
    second_derivative = np.gradient(np.gradient(torque_curve))  # calculate the second derivative of the torque curve 
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

'''
old trouble shooting 

def workbitch():
    print("fucking finally")
'''