import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def data_read_csv(filepath, sep, comma):
    """ 
    Performs a data-readin of the FTIR spectrum

    Attr:
        filepath: Filepath of the CSV 
        sep: Seperator used in the file
        comma: Comma point used (either . or , )
    
    Returns:
        data_pd: Pandas Dataframe
        data_np: Numpy dataframe
    """

    data = pd.read_csv(filepath, header=None, sep=";")
    data = data.replace(comma, ".", regex=True).astype(float)
    data_np = data.to_numpy() 

    return data, data_np


def derivative_spectrum(y_data):

    """ 
    Calculates the Numerical Derivative using the gradient
    """
    return np.gradient(y_data)

def find_zero_crossing_lin_int(x_data, y_prime, tol = 1e-5):
    """ 
    Finds zero crossing in the given derivative spectrum

    For this we use a linear interpolation approach (i.e sign change yi yi-1 <0)
    """

    zero_crossings = []
    zero_crossings_indices = []
    for i in range(1,len(y_prime)):
        if y_prime[i-1] * y_prime[i] < 0:
            zero_crossing = x_data[i-1] - y_prime[i-1] * (x_data[i] - x_data[i-1])/ (y_prime[i] - y_prime[i-1])
            zero_crossings.append(zero_crossing)
            zero_crossings_indices.append(i)
    return zero_crossings, zero_crossings_indices




def optimize_and_plot(x, y_prime, zero_crossings_indices, idx_zero_crossing, initial_bound, user_bound, bound_type='right', step_size=1):
    """ 
    Performs an optimization of a given area around a zero crossing in the derivative spectrum.

    Attr:
        x: x data points
        y_prime: numerical derivative of the data
        zero_crossing_indices: a list of possible zero crossings in the derivative spectrum
        idx_zero_crossing: the index of the zero crossing we want to look at
        initial_bound: the initial guess for the bound which we want to optimize
        user_bound: a user-defined bound (upper or lower depending on bound_type)
        bound_type: 'right' to optimize the right bound, 'left' to optimize the left bound
        step_size: the step size for the optimization (check with real spectrum)
    """
    def area_difference(left_bound, right_bound, x, y_prime, zero_crossing):
        left_area = np.trapz(y_prime[left_bound:zero_crossing], x[left_bound:zero_crossing])
        right_area = np.trapz(y_prime[zero_crossing:right_bound], x[zero_crossing:right_bound])
        return np.abs(left_area) - np.abs(right_area)

    zero_crossing = zero_crossings_indices[idx_zero_crossing]

    if bound_type == 'right':
        left_bound = initial_bound
        best_bound = initial_bound
        best_area_diff = area_difference(left_bound, initial_bound, x, y_prime, zero_crossing)

        for bound in range(initial_bound, user_bound, step_size):
            current_area_diff = area_difference(left_bound, bound, x, y_prime, zero_crossing)
            if np.abs(current_area_diff) < np.abs(best_area_diff):
                best_area_diff = current_area_diff
                best_bound = bound

        right_bound = best_bound
        left_bound = initial_bound

    elif bound_type == 'left':
        right_bound = initial_bound
        best_bound = initial_bound
        best_area_diff = area_difference(initial_bound, right_bound, x, y_prime, zero_crossing)

        for bound in range(initial_bound, user_bound, -step_size):
            current_area_diff = area_difference(bound, right_bound, x, y_prime, zero_crossing)
            if np.abs(current_area_diff) < np.abs(best_area_diff):
                best_area_diff = current_area_diff
                best_bound = bound

        left_bound = best_bound
        right_bound = initial_bound

    else:
        raise ValueError("Invalid bound_type. Use 'right' or 'left'.")

    # Plotting the derivative spectrum and marking the zero crossings
    plt.plot(x, y_prime, label="Derivative Spectrum")
    plt.scatter(x[zero_crossing], y_prime[zero_crossing], color="red", label="Zero Crossing")

    # Marking the areas with fill_between
    plt.fill_between(x[left_bound:zero_crossing], y_prime[left_bound:zero_crossing], color='blue', alpha=0.3, label='Left Area')
    plt.fill_between(x[zero_crossing:right_bound], y_prime[zero_crossing:right_bound], color='green', alpha=0.3, label='Right Area')

    plt.legend()
    plt.show()

    return left_bound, right_bound

def plot_spectra(x,y,title):
    """
    Plots the whole FTIR Spectrum given x, and y data
    """

    plt.plot(x,y, label="FTIR Spectrum", color = "black", linewidth = 0.6)
    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber (cm${-1}$)")
    plt.ylabel("Absorbance / --")
    plt.title(title)
    plt.show()