from pybaselines import Baseline as Baseline_fit
from pybaselines import utils

import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def generate_random_gaussian_peaks(num_peaks, x_range,noise_level=0.1):
    """  
    Function to generate random gaussian peaks for testing purposes of different baseline correction methods

    Parameters
    ----------
    num_peaks : int
        Number of peaks to generate.
    x_range : tuple
        Range of x values to generate the peaks in.
    noise_level : float
        Standard deviation of the noise to be added to the peaks.
    """
    x = np.linspace(x_range[0], x_range[1], 1000)

    gaussians = [utils.gaussian(x=x, height=np.random.uniform(4, 20), center=np.random.uniform(x_range[0],x_range[1]),sigma=np.random.uniform(1,2)) for _ in range(num_peaks)]
    y = np.sum(gaussians, axis=0) + np.random.normal(0, 0.1, x.shape)

    # Add noise to the y values
    y += np.random.normal(0, noise_level, y.shape)

    return x, y

def generate_random_gaussian_with_shifted_baseline(num_peaks, x_range, noise_level=0.1):
    """ 
    Function that generates a random gaussian with a shifted baseline for testing purposes of different baseline correction methods
    """

    x = np.linspace(x_range[0], x_range[1], 1000)
    gaussians = [utils.gaussian(x=x, height=np.random.uniform(4, 20), center=np.random.uniform(x_range[0],x_range[1]),sigma=np.random.uniform(1,2)) for _ in range(num_peaks)]
    y = np.sum(gaussians, axis=0) + np.random.normal(0, 0.1, x.shape)
    baseline = 5 + 10 * np.exp(-x/60)
    y += baseline

    # Add noise to the y values
    y += np.random.normal(0, noise_level, y.shape)

    return x,y 


class Baseline:
    """
    Baseline Class that containts different methods for baseline correction of a given spectrum
    """

    def __init__(self, spectrum):
        """
        Parameters
        ----------
        spectrum : Spectrum
            Spectrum object to be used for baseline correction.
        """
        self.spectrum = spectrum
        self.baseline = None
        self.corrected_spectrum = None


    @staticmethod
    def gaussian_function(x, height, center, sigma):
        """
        Gaussian function to be used for fitting
        """
        return height * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    
    def fit_gaussian(self, x, y):
        """
        Fits a gaussian to the data using the scipy curve_fit function
        """

        parameters, covariane = curve_fit(self.gaussian_function, x, y)

        return parameters

    
    def apply_rolling_average_filter(self, window_size=5):
        """ 
        Uses scipy's uniform_filter1d to apply a rolling average filter to the data

        Parameters
        ----------
        window_size : int
            Size of the window to be used for the rolling average filter.  
        """
        data = self.spectrum.data
        data = pd.DataFrame(data[:,[0,1]],columns=["x","y"])

        # Apply the filter to y values
        smooth_y = uniform_filter1d(data["y"], size=window_size)

        # Plot Before and After
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(data["x"], data["y"], label="Original Data")
        ax[0].set_title("Original Data")
        ax[1].plot(data["x"], smooth_y, label="Filtered Data")
        ax[1].set_title("Filtered Data")
        ax[1].legend()
        ax[0].legend()
        plt.show()

        # Replace the data in the spectrum with the filtered data
        self.spectrum.data[:,1] = smooth_y

    def apply_gaussian_filter(self,sigma=1,order=0):
        """
        Applies a gaussian filter to the data using scipy's gaussian_filter1d function

        Parameters
        ----------
        sigma: int
            Standard deviation of the gaussian filter to be used.
        order: int
            Order of the gaussian filter to be used. 0 = Gaussian, 1 = Derivative of Gaussian, 2 = Second Derivative of Gaussian (for convolution)
        """
        data = self.spectrum.data
        data = pd.DataFrame(data[:,[0,1]],columns=["x","y"])

        # Apply the filter to y values
        smooth_y = gaussian_filter1d(data["y"], sigma=sigma, order=order)

        # Plot Before and After
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(data["x"], data["y"], label="Original Data")
        ax[0].set_title("Original Data")
        ax[1].plot(data["x"], smooth_y, label="Filtered Data")
        ax[1].set_title("Filtered Data")
        ax[1].legend()
        ax[0].legend()
        plt.show()

        # Replace the data in the spectrum with the filtered data
        self.spectrum.data[:,1] = smooth_y

    def apply_savgol_filter(self,window_length=5,polyorder=2):
        """ 
        Applies a Savitzky-Golay filter to the data using scipy's savgol_filter function
        
        Parameters
        ----------
        window_length: int
            Length of the filter window. Must be a positive odd integer.
        polyorder: int
            Order of the polynomial used to fit the samples. Must be less than window_length.
        """
        data = self.spectrum.data
        data = pd.DataFrame(data[:,[0,1]],columns=["x","y"])

        # Apply the filter to y values
        smooth_y = savgol_filter(data["y"], window_length=window_length, polyorder=polyorder)

        # Plot Before and After
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(data["x"], data["y"], label="Original Data")
        ax[0].set_title("Original Data")
        ax[1].plot(data["x"], smooth_y, label="Filtered Data")
        ax[1].set_title("Filtered Data")
        ax[1].legend()
        ax[0].legend()
        plt.show()

        # Replace the data in the spectrum with the filtered data
        self.spectrum.data[:,1] = smooth_y



    # Polynomial Baseline Methods
    def selective_masking_poly_fit(self,poly_order=3, peak_height=0.5, peak_distance=1,peak_width_guess=5, FWHM_scaling_factor=1.5):
        """ 
        Fits a polynomial baseline to the spectrum using selective masking to take out the peaks

        Algorithm:

        1. Pick the peaks using **scipy_signal.find_peaks** function
        2. Fit a gaussian to the peaks to estimate the peak width
        3. Add a error margin to the peak width
        4. Create a mask of the data 
        5. Fit a polynomial to the masked data

        Parameters
        ----------
        poly_order : int
            Order of the polynomial to be fitted to the data.
        peak_height : float
            Minimum height of the peaks to be detected.
        peak_distance : int
            Minimum distance between the peaks to be detected.
        peak_width_guess : int
            Initial guess for the peak width to be used in the gaussian fit.
        """

        data = self.spectrum.data # get the data from the spectrum
        # Search for peaks in the data

        data = pd.DataFrame(data[:,[0,1]],columns=["x","y"])


        
        
        # TODO: Add more options for peak detection here
        peaks, _ = find_peaks(data["y"], height=peak_height, distance=peak_distance)

        # Fit a gaussian to each peak to estimate the peak width

        peak_widths = []
        for peak in peaks:
            # Fit a gaussian to the peak
            try:
                params = self.fit_gaussian(data["x"][peak-peak_width_guess:peak+peak_width_guess], data["y"][peak-peak_width_guess:peak+peak_width_guess])

                #plt.plot(data["x"][peak-peak_width_guess:peak+peak_width_guess], data["y"][peak-peak_width_guess:peak+peak_width_guess])

                # Estimate FWHM 2*sqrt(2)*np.log(2)*params[2] # sigma
                FWHM = 2 * np.sqrt(2) * np.log(2) * params[2]
                # Add scaling factor to the FWHM
                FWHM = FWHM * FWHM_scaling_factor
                peak_widths.append(FWHM)
            except:
                pass
            

        # Generate the x values for masking

        non_peaks = ()
        for i in range(len(peak_widths)):
            mask = (data["x"] > data["x"][peaks[i]] - peak_widths[i]) & (data["x"] < data["x"][peaks[i]] + peak_widths[i])
            non_peaks = np.append(non_peaks, data["y"][mask])
        
        x_mask = data["x"][~data["y"].isin(non_peaks)]
        y_mask = data["y"][~data["y"].isin(non_peaks)]

        # Fit a polynomial to the masked data
        _, params = Baseline_fit(x_mask).poly(y_mask, poly_order=poly_order,return_coef=True)

        baseline = np.polynomial.Polynomial(params["coef"])(data["x"])

        # Make Four Plots, one showing the original data + peaks one showing the original + mask
        # Next one showing the original + baseline and last the corrected data
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].plot(data["x"], data["y"], label="Original Data")
        ax[0, 0].scatter(data["x"][peaks], data["y"][peaks], color="red", label="Peaks")
        ax[0, 0].set_title("Original Data + Peaks")
        ax[0, 0].legend()
        ax[0, 1].plot(data["x"], data["y"], label="Original Data")
        ax[0, 1].scatter(data["x"][peaks], data["y"][peaks], color="red", label="Peaks")
        ax[0, 1].scatter(x_mask, y_mask, color="green", label="Mask")
        ax[0, 1].set_title("Original Data + Mask")
        ax[0, 1].legend()
        ax[1, 0].plot(data["x"], data["y"], label="Original Data")
        ax[1, 0].plot(data["x"], baseline, color="orange", label="Baseline")
        ax[1, 0].set_title("Original Data + Baseline")
        ax[1, 0].legend()
        ax[1,1].plot(data["x"], data["y"] - baseline, label="Corrected Data")
        ax[1,1].set_title("Corrected Data")
        ax[1,1].legend()
        plt.show()


    

    




