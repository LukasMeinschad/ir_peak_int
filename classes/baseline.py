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

# Use Kneed to find the knee point in the data
from kneed import KneeLocator
from scipy.sparse import diags, eye
from scipy.sparse.linalg import inv




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

        # But the baseline in the object
        self.baseline = baseline

    def modpoly_fit(self,poly_order=2, tol=1e-3, max_iter=250, mask_initial_peaks=False, return_coef=True):
        """ 
        Fits a polynomial baseline using Modified Polynomial Fitting (MODPOLY)

        Parameters:
        ----------
        poly_order : int
            Order of the polynomial to be fitted to the data.
        tol : float
            Tolerance for the convergence of the algorithm.
        max_iter : int
            Maximum number of iterations for the algorithm.
        mask_initial_peaks : bool
            If True, the initial peaks will be masked before fitting the polynomial. This works by comparing the residuals of the data to the baseline.
        return_coef : bool
            If True, the coefficients of the polynomial will be returned.
        """

        data = self.spectrum.data
        data = pd.DataFrame(data[:,[0,1]],columns=["x","y"])

        x = data["x"]
        y = data["y"]

        print(y)
        
        _, params = Baseline_fit(x).modpoly(y,poly_order=poly_order, tol=tol, max_iter=max_iter, mask_initial_peaks=mask_initial_peaks, return_coef=True)
        baseline = np.polynomial.Polynomial(params["coef"])(x)

        # Two subplots first showing original data + baseline and second showing the corrected data
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(data["x"], data["y"], label="Original Data")
        ax[0].plot(data["x"], baseline, color="orange", label="modpoly")
        ax[0].set_title("Original Data + Baseline")
        ax[0].legend()
        ax[1].plot(data["x"], data["y"] - baseline, label="Corrected Data")
        ax[1].set_title("Corrected Data")
        ax[1].legend()
        plt.show()

        # But the baseline in the object
        self.baseline = baseline

    def loess_polynomial_fit(self,fraction=0.2, scale=3,poly_order=1,tol=1e-3,use_threshold=False, return_coef=True):
        """
        Fits a polynomial baseline using LOESS (Local Regression)

        Parameters:
        ----------
        fraction : float
            Fraction of N data points to include for the fitting on each point 
        scale : int
        A scale factor applied to the weighted residueals to control the robustness of the fit. Default is 3.
        poly_order: int
            Order of the polynomial to be fitted to the data.
        tol : float
            Tolerance for the convergence of the algorithm.
        use_threshhold : bool
            If True, threshholding is used to perform iterative fitting similar to MODPOLY approach
        """
        data = self.spectrum.data
        data = pd.DataFrame(data[:,[0,1]],columns=["x","y"])
        x = data["x"]
        y = data["y"]
        # Apply LOESS fitting
        baseline, params= Baseline_fit(x).loess(y, fraction=fraction, scale=scale, poly_order=poly_order, tol=tol, use_threshold=use_threshold, return_coef=return_coef)

        # Two subplots first showing original data + baseline and second showing the corrected data
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(data["x"], data["y"], label="Original Data")
        ax[0].plot(data["x"], baseline, color="orange", label="LOESS")
        ax[0].set_title("Original Data + Baseline")
        ax[0].legend()
        ax[1].plot(data["x"], data["y"] - baseline, label="Corrected Data")
        ax[1].set_title("Corrected Data")
        ax[1].legend()
        plt.show()
        # But the baseline in the object
        self.baseline = baseline

    def asls_fit(self, lam=1e6, p=1e-2, tol=1e-3):
        """
        Uses penalized least squares (Whittaker smoothing) to fit a baseline to the data

        Parameters
        ----------
        lam : float
            Smoothing parameter. The larger the value, the smoother the baseline.
        p : float
            Penalazing weight factor. The larger the value, the more penalizing is done.
        tol : float
            Tolerance for the convergence of the algorithm.
        """
        data = self.spectrum.data
        data = pd.DataFrame(data[:,[0,1]],columns=["x","y"])
        x = data["x"]
        y = data["y"]

        # Apply ASLS fitting
        baseline, params = Baseline_fit(x).asls(y, lam=lam, p=p, tol=tol)

        # Two subplots first showing original data + baseline and second showing the corrected data
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(data["x"], data["y"], label="Original Data")
        ax[0].plot(data["x"], baseline, color="orange", label="ASLS")
        ax[0].set_title("Original Data + Baseline")
        ax[0].legend()
        ax[1].plot(data["x"], data["y"] - baseline, label="Corrected Data")
        ax[1].set_title("Corrected Data")
        ax[1].legend()
        plt.show()

        # But the baseline in the object
        self.baseline = baseline

    def airpls(self, lam=1e6, diff_order=2, max_iter=50, tol=1e-3):
        """ 
        Uses adaptive iteratively reweighted penalized least squares (AIRPLS) to fit a baseline to the data

        Parameters
        ----------
        lam : float
            Smoothing parameter. The smaller lambda the better the fit to the data.
        diff_order : int
            Order of the derivative operator, normally 2
        max_iter : int
            Maximum number of iterations for the algorithm.
        tol : float
            Tolerance for the convergence of the algorithm.
        """
        data = self.spectrum.data
        data = pd.DataFrame(data[:,[0,1]],columns=["x","y"])
        x = data["x"]
        y = data["y"]

        # Apply AIRPLS fitting
        baseline, params = Baseline_fit(x).airpls(y, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol)
        
        # Two subplots first showing original data + baseline and second showing the corrected data
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(data["x"], data["y"], label="Original Data")
        ax[0].plot(data["x"], baseline, color="orange", label="AIRPLS")
        ax[0].set_title("Original Data + Baseline")
        ax[0].legend()
        ax[1].plot(data["x"], data["y"] - baseline, label="Corrected Data")
        ax[1].set_title("Corrected Data")
        ax[1].legend()
        plt.show()
        # But the baseline in the object
        self.baseline = baseline

    def validate_lambda(self,lam_range=(-2,6), num_points=50, method="asls"):
        """ 
        Explores the effect of the lambda parameter on the baseline correction. 

        For this we use the Residual sum of squares (RSS) to measure the discrepancy between the data and the baseline.
        """

        data = self.spectrum.data
        data = pd.DataFrame(data[:,[0,1]],columns=["x","y"])
        x = data["x"]
        y = data["y"]

        # Generate Lambda values in log space
        lambdas = np.logspace(lam_range[0], lam_range[1], num_points)

        # For each lambda value we fit the baseline and then estimate the RSS
        smoothed_signals = []
        residuals = []

        for lam in lambdas:
            if method == "asls":
                baseline,params = Baseline_fit(x).asls(y, lam=lam)
                smoothed_signals.append(baseline)
                residuals.append(np.sum((y - baseline) ** 2))
            if method == "airpls":
                baseline,params = Baseline_fit(x).airpls(y, lam=lam)
                smoothed_signals.append(baseline)
                residuals.append(np.sum((y - baseline) ** 2))

        # Use the kneed package to find the knee point in the data
        kn = KneeLocator(lambdas, residuals, curve="convex", direction="increasing")

        print("...RSS values for the given lambda range...")

        

        plt.figure(figsize=(10, 5))
        plt.semilogx(lambdas, residuals) # Converts the x axis to log scale
        plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], colors='r', linestyles='--', label='Knee Point')
        plt.xlabel("Lambda (Log Scale)")
        plt.ylabel("Residual Sum of Squares (RSS)")
        plt.title("Effect of Lambda on Baseline Correction")
        plt.grid()
        plt.legend()
        plt.show()

        print("--------------- Optimal Lambda Value ----------------")
        print(f"Optimal Lambda Value: {kn.knee}")
        print("-----------------------------------------------------")

        baseline_opt, params_opt = Baseline_fit(x).asls(y, lam=kn.knee)
        # Plot the original data and the baseline
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(data["x"], data["y"], label="Original Data")
        ax[0].plot(data["x"], baseline_opt, color="orange", label="Baseline")
        ax[0].set_title("Original Data + Baseline")
        ax[0].legend()
        ax[1].plot(data["x"], data["y"] - baseline_opt, label="Corrected Data")
        ax[1].set_title("Corrected Data")
        ax[1].legend()
        plt.show()


        
        # Return a new baseline oject with the smoothed baseline_opt as y data
        data["y"] = baseline_opt

        # Combine x,y
        smoothed_signals = np.array([data["x"],baseline_opt]).T

        return smoothed_signals, kn.knee

    def loocv_lambda(self, lam_range=(-2,6), num_points=50, method="asls"):
        """ 
        Find an optimal lambda value using Leave-One-Out Cross Validation (LOOCV)
        This method is used to estimate the performance of the model on unseen data by leaving one data point out and fitting the model on the rest of the data.
        """

        data = self.spectrum.data
        data = pd.DataFrame(data[:,[0,1]],columns=["x","y"])
        x = data["x"]
        y = data["y"]
        n = len(y)
        loocv_scores = []
        rmse_scores = []
        lambdas = np.logspace(lam_range[0], lam_range[1], num_points)
        print("...Performing LOOCV for the given Lambda range...")
        for lam in lambdas:
            all_squared_residuals = []
            for i in range(n):
                # remove the i-th data point
                x_train = np.delete(x, i)
                y_train = np.delete(y, i)
                # The test data is exactly the i-th data point
                x_test = x[i]
                y_test = y[i]
                # Fit the model on the training data
                if method == "asls":
                    baseline, params = Baseline_fit(x_train).asls(y_train, lam=lam)

                    y_pred = np.interp(x_test, x_train, baseline)
                    # Calculate the squared residual
                    squared_residual = (y_test - y_pred) ** 2
                    all_squared_residuals.append(squared_residual)
                if method == "airpls":
                    baseline, params = Baseline_fit(x_train).airpls(y_train, lam=lam)
                    y_pred = np.interp(x_test, x_train, baseline)
                    # Calculate the squared residual
                    squared_residual = (y_test - y_pred) ** 2
                    all_squared_residuals.append(squared_residual)
                    
            # Sum all the squared residuals and calculate the mean
            mean_squared_residual = np.mean(all_squared_residuals)
            loocv_scores.append(mean_squared_residual)
            rmse_scores.append(np.sqrt(mean_squared_residual))
        
        # Use the kneed package to find the knee point in the data
        kn = KneeLocator(lambdas, loocv_scores, curve="convex", direction="increasing")

        # Plot LOOCV scores and RMSE in two subplots
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].semilogx(lambdas, loocv_scores)
        ax[0].vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], colors='r', linestyles='--', label='Knee Point')
        ax[0].set_xlabel("Lambda (Log Scale)")
        ax[0].set_ylabel("LOOCV Score")
        ax[0].set_title("Effect of Lambda on LOOCV Score") 
        ax[0].grid()
        ax[1].semilogx(lambdas, rmse_scores)
        ax[1].set_xlabel("Lambda (Log Scale)")
        ax[1].set_ylabel("RMSE")
        ax[1].set_title("Effect of Lambda on RMSE")
        ax[1].grid()
        plt.legend()
        plt.show()

        print("--------------- Optimal Lambda Value ----------------")
        print(f"Optimal Lambda Value: {kn.knee}")
        print("-----------------------------------------------------")

        # Compute baseline using the optimal lambda value
        baseline_opt, params_opt = Baseline_fit(x).asls(y, lam=kn.knee)
        # Plot the original data and the baseline
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(data["x"], data["y"], label="Original Data")
        ax[0].plot(data["x"], baseline_opt, color="orange", label="Baseline")
        ax[0].set_title("Original Data + Baseline")
        ax[0].legend()
        ax[1].plot(data["x"], data["y"] - baseline_opt, label="Corrected Data") 
        ax[1].set_title("Corrected Data")
        ax[1].legend()
        plt.show()
        data["y"] = baseline_opt

        # Combine x,y
        smoothed_signals = np.array([data["x"],baseline_opt]).T

        return smoothed_signals, kn.knee
    


    
    def plot_different_lambdas(self,lam_values = [1e-2,1e0,1e2,1e4], method="asls"):
        """
        Plots the effect of different lambda values on the baseline correction

        Parameters
        ----------
        lam_values : list
            List of lambda values to be used for the baseline correction.
        method : str
            Method to be used for the baseline correction. Default is "asls".
        """
        data = self.spectrum.data
        data = pd.DataFrame(data[:,[0,1]],columns=["x","y"])
        x = data["x"]
        y = data["y"]

        baselines = []

        for lam in lam_values:
            baseline, params = Baseline_fit(x).asls(y, lam=lam)
            baselines.append(baseline)
        # Plot the original data and the baselines
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].plot(data["x"], data["y"], label="Original Data")
        ax[0].set_title("Original Data")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")
        ax[1].set_title("Corrected Data")
        ax[1].plot(data["x"], data["y"], label="Original Data")
        for i, lam in enumerate(lam_values):
            ax[0].plot(data["x"], baselines[i], label=f"Baseline (lambda={lam})")
            ax[1].plot(data["x"], data["y"] - baselines[i], label=f"Corrected Data (lambda={lam})")
        ax[0].legend()
        ax[1].legend()
        plt.show()

    def irsqr_fitting(self,lam=100, quantile=0.05, num_knots=100,spline_degree=3,max_iter=100):
        """ 
        Uses iteratively reweighted spline quantile regression. Is a form of penalized spline fitting

        Parameters
        ----------
        lam : float
            Smoothing parameter. The larger the value, the smoother the baseline.
        quantile : float
            Quantile to be used for the fitting. Default is 0.05.
        num_knots : int
            Number of knots for the spline fitting. Default is 100.
        spline_degree : int
            Degree of the spline to be used for the fitting. Default is 3, so cubic spline.
        max_iter : int
            Maximum number of iterations for the algorithm. Default is 100.
        """    
    
        data = self.spectrum.data
        data = pd.DataFrame(data[:,[0,1]],columns=["x","y"])
        x = data["x"]
        y = data["y"]

        # Apply IRSQR fitting
        baseline, params = Baseline_fit(x).irsqr(y, lam=lam, quantile=quantile, num_knots=num_knots, spline_degree=spline_degree, max_iter=max_iter)

        # Two subplots first showing original data + baseline and second showing the corrected data
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(data["x"], data["y"], label="Original Data")
        ax[0].plot(data["x"], baseline, color="orange", label="IRSQR")
        ax[0].set_title("Original Data + Baseline")
        ax[0].legend()
        ax[1].plot(data["x"], data["y"] - baseline, label="Corrected Data")
        ax[1].set_title("Corrected Data")
        ax[1].legend()
        plt.show()

        # But the baseline in the object
        self.baseline = baseline

    




