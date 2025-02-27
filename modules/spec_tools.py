import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d


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


def plot_spectral_window(x,y,spectral_window,title):
    """ 
    Plots a given spectral window for the IR data

    Attr:
        x: x-data
        y: y-data
        spectral_window: a tuple with the spectral window
        title: a string with the title
    
    Returns:
        A Plot of the Spectral Window with the selected data
    """
    # Select the data using the spectral window
    mask = np.logical_and(x >= spectral_window[1], x <= spectral_window[0])
    x_window = x[mask]
    y_window = y[mask]

    plt.plot(x_window, y_window, label="Spectral Window", color ="black",linewidth = 0.6)
    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber / (cm${-1}$)")
    plt.ylabel("Absorbance")
    plt.title(title)
    plt.show()
    return x_window, y_window


def peak_picking_spectral_window(x_window,y_window,height=None,distance=None,prominence=None):
    """
    Applies Peak Picking to a given spectral window

    Attr:
        x_window: x-data of spectral window
        y_window: y-data of spectral window
        height: Required height of Peaks, optional
        distance: Required horizontal distance between peaks
        prominence: Required prominence of peaks
    """

    peaks, properties = find_peaks(y_window, height=height, distance=distance, prominence=prominence)

    plt.figure(figsize=(10,6))
    plt.plot(x_window,y_window, label="Spectal Window", color = "black",linewidth = 0.6)
    plt.plot(x_window[peaks],y_window[peaks], "x", label="Peaks", color ="Red")
    plt.gca().invert_xaxis()
    plt.title("Peak Picking Spectral Window")
    plt.xlabel("Wavenumber / (cm${-1}$)")
    plt.ylabel("Absorbance")
    plt.legend()

    for peak in peaks:
        plt.annotate(f"{x_window[peak]:.0f}", (x_window[peak],y_window[peak]),textcoords="offset points", xytext=(0,5), ha="center")
    plt.show()

    return peaks,properties


def finite_differences(x,y, order=1):
    """ 
    Gradient or first derivative is computed using central differences as the interior points or first, second order accurate one sides
    differences at the boundary
    """
    dydx = y
    for _ in range(order):
        dydx = np.gradient(dydx,x)
    return dydx

def plot_derivative_spectrum(x_window, y_window, method="finite", max_order=2):
    """ 
    Calculates and plots the derivative spectrum using different methods for derivative computation.

    Methods:
        "finite": computation by finite differences (central differences)
    """
    fig, axes = plt.subplots(max_order + 1, 1, figsize=(10, 6 * (max_order + 1)))

    # Plot the normal spectrum
    axes[0].plot(x_window, y_window, label="Spectral Window", color="black", linewidth=0.6)
    axes[0].invert_xaxis()
    axes[0].set_title("Normal Spectrum")
    axes[0].set_xlabel("Wavenumber / (cm${-1}$)")
    axes[0].set_ylabel("Absorbance")
    axes[0].legend()

    # Plot the derivative spectra iteratively
    for order in range(1, max_order + 1):
        if method == "finite":
            y_prime_window = finite_differences(x_window, y_window, order=order)
        
        axes[order].plot(x_window, y_prime_window, label=f"Derivative Spectrum (Order {order})", color="red", linewidth=0.6)
        axes[order].invert_xaxis()
        axes[order].set_title(f"Derivative Spectrum (Order {order})")
        axes[order].set_xlabel("Wavenumber / (cm${-1}$)")
        axes[order].set_ylabel(f"{order} Derivative of Absorbance")
        axes[order].legend()

    plt.tight_layout()
    plt.show()



def fit_baseline_asls_aspls(x_window,y_window, lam, tol, max_iter):
    """ 
    Fits the Baseline using  the Asymmetrically reweighted penalized least squares smoothing

    Also fits Baseline using the Asymmetrically Least Squares smoothing and compares the convergence
    
    Attr:
        x_window = selected x data
        y_window = selected y data
        lam = 
        tol = 
        max_iter
    """
    baseline_fitter= Baseline(x_window)

    fit1,params_1 = baseline_fitter.asls(y_window, lam=lam, tol=tol, max_iter=max_iter)
    fit2,params_2 = baseline_fitter.aspls(y_window, lam=lam, tol=tol, max_iter=max_iter)


    plt.plot(x_window,y_window, label ="Spectral Window", color ="black", linewidth = 0.6)
    plt.plot(x_window,fit1, label="asls baseline")
    plt.plot(x_window,fit2, label="aspls baseline")
    plt.legend()
    plt.xlabel("Wavenumber / (cm$^{-1}$)")
    plt.ylabel("Absorbance")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.show()

    # Next of show the Fits
    plt.figure()
    plt.plot(np.arange(1, len(params_1["tol_history"]) + 1), params_1["tol_history"], label="asls")
    plt.plot(np.arange(1, len(params_2["tol_history"]) + 1), params_2["tol_history"], label="aspls")
    plt.axhline(tol, ls=":", color="k", label="tolerance")
    plt.legend()
    plt.show()

    baseline_corrected_spectrum_asls = y_window - fit2
    baseline_corrected_spectrum_aspls = y_window - fit2
    return baseline_corrected_spectrum_asls, baseline_corrected_spectrum_aspls



def fit_baseline_poly(x_window,y_window,p):
    """
    Fits the Baseline using Modpoly method

    Attr:
        x_window = selected x data
        y_window = selected y data
        p = degree of polyomial
    """

    baseline_fitter = Baseline(x_window)

    fit, params = baseline_fitter.modpoly(y_window,poly_order=p)
    
    plt.figure(figsize=(10,6))
    plt.plot(x_window,y_window, label="Spectral Window", color = "black", linewidth = 0.6)
    plt.gca().invert_xaxis()
    plt.plot(x_window, fit, label=f"Polynomial Order {p}", color="blue", ls="--")
    plt.legend()
    plt.xlabel("Wavenumber / (cm$^{-1}$)")
    plt.ylabel("Intensity")

    baseline_correted_spectrum_modpoly = y_window - fit
    return baseline_correted_spectrum_modpoly

def fit_baseline_morph(x_window,y_window, offset):
    """ 
    Performs a baseline fit by using a morphology based method,
    this algorithm also approximates the half_window by using the FWHM of the largest peak

    Attr:
        x_window = selected x data
        y_window = selected y data
        offset = offset to addjust half_window size
    """
    baseline_fitter = Baseline(x_window)

    peak_idx = np.argmax(y_window)
    peak_value = y_window[peak_idx]

    half_max = peak_value/2

    interpolator = interp1d(x_window,y_window-half_max, kind="linear", fill_value="extrapolate")
    roots = np.where(np.diff(np.sign(interpolator(x_window))))[0]

    if len(roots) >= 2:
        fwhm = np.abs(x_window[roots[-1]] - x_window[roots[0]])
        x_left = x_window[roots[-1]]
        x_right = x_window[roots[0]]
    
    
    fit, params = baseline_fitter.mor(y_window,half_window=round(fwhm)+offset)




    plt.plot(x_window,y_window, color = "black", linewidth = 0.6, label="Spectral Window")
    plt.axhline(half_max, color = "red", ls = "--", label="Half Max Biggest Peak")
    plt.gca().invert_xaxis()
    plt.plot([x_left,x_right],[half_max,half_max], "go", label="FWHM Points")
    plt.plot(x_window, fit, label="Mor Baseline")
    plt.annotate(f"FWHM = {fwhm:.2f}", xy=((x_left + x_right)/2, half_max), xytext=(0,10), textcoords="offset points", ha="center")
    plt.legend()

    baseline_corrected_spectrum_morph = y_window - fit
    return baseline_corrected_spectrum_morph
