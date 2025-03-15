import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import altair as alt
from pybaselines import Baseline as Baseline_fit


class Spectrum:
    def __init__(self,name,data):

        """ 
        Initializes the spectrum object

        Parameters:
            name (str): Name of the Compound
            data (np.array): The data of the spectrum as NumPy array
        """
        self.name = name
        self.data = data

    

    @classmethod
    def from_csv(cls,filepath,sep,comma,header=None):
        """ 
        Reads in the data from a CSV file

        Parameters:
            filepath (str): Filepath of the CSV
            sep (str): Seperator used in the file
            comma (str): Comma point used (either . or , )
            header (int): None if no header is present, else the header determines how much rows are skipped
        """
        data = pd.read_csv(filepath, header=header, sep=sep)
        data = data.replace(comma, ".", regex=True).astype(float)
        data_np = data.to_numpy()
        return cls(filepath,data_np)
    
    
    def derivative(self,data):
        """ 
        Calculates the numerical deriative using central differences

        Returns:
            np.array: The Numerical Derivative [x,y']
        """
        return np.gradient(self.data[:,1],self.data[:,0])

     
    def interactive_integration(self):
        """ 
        Allows for interactive integration where the user is able to set the boundaries of integration
        using the altair package
        """

        alt.data_transformers.disable_max_rows()

        data = pd.DataFrame(self.data[:,0:2], columns=["x","y"]) 

        interval_selection = alt.selection_interval(encodings=["x"],name="interval")

        base_chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title="Wave Number / cm$^{-1}$", sort="descending").axis(format="0.0f"),
            y=alt.Y("y", title="Intensity"),
            color=alt.value("black")
        ).properties(
            title="Interactive Integration",
            width = 800,
            height = 400
        ).add_selection(
            interval_selection
        )

        integral = alt.Chart(data).mark_text(align="left",dx=10,dy=10, fontSize=14).encode(
            x=alt.value(10),
            y=alt.value(10),
            text=alt.condition(interval_selection,alt.datum(alt.Expr("sum(datum.y)*(max(datum.x)-min(datum.x)/ count(datum.x)")),
            alt.value("Select an Interval to Calculate the Integral")
            )
        ).transform_filter(
            interval_selection
        ).transform_aggregate(
            sum_y="sum(y)",
            count_x="count(x)",
            min_x="min(x)",
            max_x="max(x)"
        ).transform_calculate(
            integral="datum.sum_y * (datum.max_x - datum.min_x) / datum.count_x"
        )

        chart = base_chart + integral

        return chart

    def plot_spectrum(self,title=None):
        """ 
        Plots the spectrum using Altair as a package
        
        Parameters:
            self: The Spectrum object
            title: The title of the plot
            cols: Further specify which colums should be used
        """
        alt.data_transformers.disable_max_rows()
        if title:
            title = "Spectrum of " + self.name
        else:
            title = title

        data = pd.DataFrame(self.data[:,0:2], columns=["x","y"])
        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title="Wave Number / cm$^{-1}$", sort="descending").axis(format="0.0f"),
            y=alt.Y("y", title="Intensity"),
            color=alt.value("black")
        ).properties(
            title=title,
            width = 800,
            height = 400
        )

        # Create Selection
        selection = alt.selection_interval(bind="scales")

        # Add selection to chart
        chart = chart.add_selection(selection)

        return chart

    def plot_derivative(self,title=None):
        """ 
        Plots the Derivative Spectrum of the Given Spectrum
        """
        alt.data_transformers.disable_max_rows()
        if title:
            title = "Derivative Spectrum of " + self.name
        else:
            title = title

        derivative = self.derivative(self.data[:,0:2])
        data = np.zeros((len(self.data),2))
        data[:,0] = self.data[:,0]
        data[:,1] = derivative
        data = pd.DataFrame(data, columns=["x","y"])

        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title="Wave Number / cm$^{-1}$", sort="descending").axis(format="0.0f"),
            y=alt.Y("y", title="First Derivative of Intensity"),
            color=alt.value("red")
        ).properties(
            title=title,
            width = 800,
            height = 400
        )

        # Create Selection
        selection = alt.selection_interval(bind="scales")

        # Add selection to chart
        chart = chart.add_selection(selection)

        return chart
    
    def plot_spectral_window(self,x_min,x_max,title=None):
        """
        Plots a given spectral window of the Spectrum object

        Attributes:
            x_min: Minimum Wavenumber
            x_max: Maximum Wavenumber
            title: Title of the plot
        """
        data = self.data
        mask = (data[:,0] >= x_min) & (data[:,0] <= x_max)
        data = data[mask]
        
        # Select column one
        data = pd.DataFrame(data[:,0:2], columns=["x","y"])
        alt.data_transformers.disable_max_rows()
        if title:
            title = "Spectral Window of " + self.name
        else:
            title = title

        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title="Wave Number / cm$^{-1}$", sort="descending").axis(format="0.0f"),
            y=alt.Y("y", title="Intensity"),
            color=alt.value("black")
        ).properties(
            title=title,
            width = 800,
            height = 400
        )

        # Create Selection

        selection = alt.selection_interval(bind="scales")

        # Add selection to chart

        chart = chart.add_selection(selection)

        return chart
    
    def plot_spectral_window_peak_picking(self,x_min,x_max,title=None,threshold=0.1):
        """
        Performs peakpicking on a spectrum in a given spectral window using the Scipy find_peaks function

        Attributes:
            x_min: Minimum Wavenumber
            x_max: Maximum Wavenumber
            title: Title of the plot
            threshold: Threshold for peak picking
        """

        data = self.data
        mask = (data[:,0] >= x_min) & (data[:,0] <= x_max)
        data = data[mask]

        # Select column one
        data = pd.DataFrame(data[:,0:2], columns=["x","y"])
        alt.data_transformers.disable_max_rows()
        if title:
            title = "Spectral Window of " + self.name
        else:
            title = title

        # Find the peaks

        peaks, _ = find_peaks(data["y"], height=threshold)

        # Create the chart
        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title="Wave Number / cm$^{-1}$", sort="descending").axis(format="0.0f"),
            y=alt.Y("y", title="Intensity"),
            color=alt.value("black")
        ).properties(
            title=title,
            width = 800,
            height = 400
        )

        # Create the scatter plot for the peaks
        data_peaks = data.iloc[peaks]
        peaks = alt.Chart(data_peaks).mark_point(color="red").encode(
            x="x",
            y="y"
        )

        annotations = alt.Chart(data_peaks).mark_text(align="left", baseline="middle", dx=7, dy=-7, fontSize=14).encode(
            x="x",
            y="y",
            text=alt.Text("x:Q",format=".2f")
        )

        # Create Selection

        selection = alt.selection_interval(bind="scales")

        # Add selection to chart

        chart = chart+ annotations + peaks.add_selection(selection)

        return chart
    
    # Function to split a spectral window in a own object

    def split_spectral_window(self,window):
        """
        Function that splits the spectrum object at the given x_min,x_max values.
        The obtained spectrum can then be used for futher processing

        Attributes:
            window = A list of two values [x_min,x_max]
        """

        data = self.data
        mask = (data[:,0] >= window[0]) & (data[:,0] <= window[1])
        data = data[mask]

        return Spectrum(self.name + " " + str(window),data) 

    
    def calculate_integral(self,x_min,x_max, plotting_tol=50):
        """ 
        Calculates the Integral of the Spectrum in a given Spectral Window and 
        visualized the area under the curve in the plot
        """



        data = self.data

    
        mask_integral = (data[:,0] >= x_min) & (data[:,0] <= x_max)
        mask_plot = (data[:,0] >= x_min - plotting_tol) & (data[:,0] <= x_max + plotting_tol)

        data_int = data[mask_integral]
        data_plot = data[mask_plot]



        

        # Calculate the Integral
        # Trapzoidal Rule
        integral = np.abs(np.trapz(data_int[:,1],data_int[:,0]))

        # Plot the Spectrum

        data_plot = pd.DataFrame(data_plot[:,0:2], columns=["x","y"])
        chart = alt.Chart(data_plot).mark_area().encode(
            x=alt.X("x", title="Wave Number / cm$^{-1}$", sort="descending").axis(format="0.0f"),
            y=alt.Y("y", title="Intensity"),
            color=alt.value("black")
        ).properties(
            title="Spectral Window",
            width = 800,
            height = 400
        )

        # Highlight the Area in the Spectrum
        data_int = pd.DataFrame(data_int[:,0:2], columns=["x","y"])
        area = alt.Chart(data_int).mark_area(line={"color": "darkgreen"},color=alt.Gradient(
            gradient = "linear",
            stops = [alt.GradientStop(color="white", offset=0),
                     alt.GradientStop(color="darkgreen", offset=1)],
                     x1=1,
                     x2=1,
                     y1=1,
                     y2=0
        )).encode(
            x="x",
            y="y"
        ).transform_filter(
            alt.FieldRangePredicate(field="x", range=[x_min,x_max])
        )

        annotation = alt.Chart(pd.DataFrame({
            "x":[x_min + (x_max - x_min)/2],
            "y": [data_int["y"].max()],
            "text": [f"Integral: {integral:.2f}"]
        })).mark_text(align="center", baseline="bottom", fontSize=14).encode(                                 
            x="x:Q",
            y="y:Q",
            text="text:N"
        )


        chart = chart + area + annotation


        # Create Selection

        selection = alt.selection_interval(bind="scales")

        # Add selection to chart

        chart = chart.add_selection(selection)
        
        return chart, integral

    def plot_multiple_spectral_windows(self,ls_of_windows,title="None", subtitles=None):
        """ 
        Function to plot multiple spectral windows

        Attr:
            ls_of_windows: List of tuples with the spectral windows
            title: Title of the plot
            subtitles: List of subtitles for the plots
        """

        num_windows = len(ls_of_windows)
        rows = (num_windows + 1) // 2

        charts = []
        for i in range(num_windows):
            x_min, x_max = ls_of_windows[i]
            mask = (self.data[:,0] >= x_min) & (self.data[:,0] <= x_max)
            data_window = self.data[mask]

            # convert to pandas dataframe
            data_df = pd.DataFrame(data_window[:,0:2], columns=["x","y"])

            chart = alt.Chart(data_df).mark_line().encode(
                x=alt.X("x", title="Wave Number / cm$^{-1}$", sort="descending").axis(format="0.0f"),
                y=alt.Y("y", title="Intensity"),
                color=alt.value("black")
            ).properties(
                title=subtitles[i] if subtitles else f"Spectral Window {i+1}",
                width = 200,
                height = 200
            ).add_selection(
                alt.selection_interval(bind="scales")
            )
            charts.append(chart)
        
        # Combine Charts into a grid

        grid = alt.vconcat(*[alt.hconcat(*charts[i:i+2]) for i in range(0,num_windows,2)]).properties(
            title=title
        )

        return grid
    



    def integrate_multiple_spectral_windows(self,ls_of_windows, title="None", subtitles=None):
        """ 
        Function to integrate multiple spectral windows

        Attr:
            ls_of_windows: List of tuples with the spectral windows
        """

        num_windows = len(ls_of_windows)
        rows  = (num_windows + 1) // 2

        integrals = []
        charts = []

        for i in range(num_windows):
            x_min, x_max = ls_of_windows[i]
            mask = (self.data[:,0] >= x_min) & (self.data[:,0] <= x_max)
            data_window = self.data[mask]

            # convert to pandas dataframe
            data_df = pd.DataFrame(data_window[:,0:2], columns=["x","y"])

            # Calculate the Integral
            # Trapzoidal Rule
            integral = np.abs(np.trapz(data_window[:,1],data_window[:,0]))
            integrals.append(integral)

            # Plot the Spectrum

            data_plot = pd.DataFrame(data_window[:,0:2], columns=["x","y"])

            chart = alt.Chart(data_plot).mark_area().encode(
                x=alt.X("x", title="Wave Number / cm$^{-1}$", sort="descending").axis(format="0.0f"),
                y=alt.Y("y", title="Intensity"),
                color=alt.value("black")
            ).properties(
                title=subtitles[i] if subtitles else f"Spectral Window {i+1}",
                width = 200,
                height = 200
            )

            # Highlight the Area in the Spectrum

            area = alt.Chart(data_plot).mark_area(line={"color": "darkgreen"},color=alt.Gradient(
                gradient = "linear",
                stops = [alt.GradientStop(color="white", offset=0),
                        alt.GradientStop(color="darkgreen", offset=1)],
                        x1=1,
                        x2=1,
                        y1=1,
                        y2=0
            )).encode(
                x="x",
                y="y"
            ).transform_filter(
                alt.FieldRangePredicate(field="x", range=[x_min,x_max])
            )

            annotation = alt.Chart(pd.DataFrame({
                "x":[x_min + (x_max - x_min)/2],
                "y": [data_plot["y"].max()],
                "text": [f"Integral: {integral:.2f}"]
            })).mark_text(align="center", baseline="bottom", fontSize=12).encode(                                 
                x="x:Q",
                y="y:Q",
                text="text:N"
            )

            # Add Selection

            selection = alt.selection_interval(bind="scales")

            chart = chart + area + annotation
            
            # Add Selection to chart
            chart = chart.add_selection(selection)


            charts.append(chart)


        # Combine Charts into a grid

        grid = alt.vconcat(*[alt.hconcat(*charts[i:i+2]) for i in range(0,num_windows,2)]).properties(
            title=title if title else f"Integrals of {self.name}"
        )

        return grid, integrals
    

class Baseline:
    """
    A baseline class which contains different methods for baseline correction
    """

    def __init__(self,spectrum):
        """
        Initializes the Baseline class.

        Parameters:
            spectrum (object) a Spectrum object
        """

        self.spectrum = spectrum
        self.baseline = None


    def fit_baseline_asls(self,lam,tol,max_iter):
        """
        Calculates the Baseline Fitting using Asls and Aspls method
        """

        y = self.spectrum.data[:,1]

        baseline_fitter = Baseline_fit(self.spectrum.data[:,0])

        # Fit the Baseline

        fit,params = baseline_fitter.asls(y, lam=lam, tol=tol, max_iter=max_iter)

        # Modify Baseline Object

        self.baseline = fit

        return fit,params
    
    def fit_baseline_iasls(self,lam,tol,max_iter):
        """
        Fits the baseline using improved asymmetric least squares method

        Parameters:
            lam (float): Lambda value for the method smoothing value
            tol (float): Tolerance value
            max_iter (int): Maximum number of iterations
        """

        y = self.spectrum.data[:,1]

        baseline_fitter = Baseline_fit(self.spectrum.data[:,0])

        fit,params = baseline_fitter.iasls(y, lam=lam, tol=tol, max_iter=max_iter)

        self.baseline = fit

        return fit,params

    
    def fit_baseline_poly(self,p):
        """
        Fits a polynomial baseline to the spectrum. Uses least squares method to fit the polynomial to the data

        Parameters:
            p (int): The degree of the polynomial
        """

        y = self.spectrum.data[:,1]
        baseline_fitter = Baseline_fit(self.spectrum.data[:,0])

        fit,params = baseline_fitter.poly(y,poly_order=p)

        self.baseline = fit

        return fit,params

    
    def substract_baseline(self):
        """ 
        Substracts the current baseline from the spectrum

        Returns:
            Spectrum: object of the spectrum class

        """

        y = self.spectrum.data[:,1]
        y_corrected = y - self.baseline

        return Spectrum(self.spectrum.name + " Corrected", np.column_stack((self.spectrum.data[:,0],y_corrected)))


    def plot_baseline_and_spectrum(self,title=None):
        """ 
        Method to plot a given baseline
        """

        data = self.spectrum.data

        # convert to pandas dataframe
        data = pd.DataFrame(data[:,0:2], columns=["x","y"])

        # Create the chart

        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title="Wave Number / cm$^{-1}$", sort="descending").axis(format="0.0f"),
            y=alt.Y("y", title="Intensity"),
            color=alt.value("black")
        ).properties(
            title=title,
            width = 800,
            height = 400
        )

        data_baseline = pd.DataFrame({
            "x": data["x"],
            "y": self.baseline
        })
        
        # Plot Baseline

        baseline = alt.Chart(data_baseline).mark_line(color="red").encode(
            x="x",
            y="y"
        )

        combined = alt.layer(
            chart.encode(color=alt.value("black")).properties(title="Spectrum"),
            baseline.encode(color=alt.value("red")).properties(title="Baseline")
        ).resolve_scale(
            color="independent"
            ).properties(
            title=title,
            width = 800,
            height = 400
        )

        selection = alt.selection_interval(bind="scales")


        combined = combined.add_selection(selection)

        return combined
    





def data_read_csv(filepath, sep, comma, header=None):
    """ 
    Performs a data-readin of the FTIR spectrum

    Attr:
        filepath: Filepath of the CSV 
        sep: Seperator used in the file
        comma: Comma point used (either . or , )
        header: None if no header is present, else the header determines how much rows are skipped
    
    Returns:
        data_pd: Pandas Dataframe
        data_np: Numpy dataframe
    """

    data = pd.read_csv(filepath, header=header, sep=";")
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
