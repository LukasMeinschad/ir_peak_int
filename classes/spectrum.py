import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import altair as alt

from classes import annotation




def convert_subscript_superscript(text, is_superscript=True):
    """ 
    Helper Function to convert subscripts and superscripts
    """
    normal_chars = "0123456789"
    superscript_chars = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    subscript_chars = "₀₁₂₃₄₅₆₇₈₉"

    if is_superscript:
        mapping = str.maketrans(normal_chars,superscript_chars)
    else:
        mapping = str.maketrans(normal_chars, subscript_chars)
    converted_text = text.translate(mapping)
    return converted_text



def ratios_of_integrals(integrals,reference):
    """
    Calculates the ratios of a list of integrals to a given reference

    The reference should be given as a index
    """

    ratios = []
    for integral in integrals:
        ratio = integral/integrals[reference]
        ratios.append(ratio)
    return ratios

def remove_numeration_atoms(ls):
    """
    Removes the enumeration of a list of atoms

    ["H1","H2"] -> ["H","H"]
    """
    new_list = []
    for atom in ls:
        if atom[-1].isdigit():
            new_list.append(atom[:-1])
        else:
            new_list.append(atom)
    return np.array(new_list)

class Spectra:
    def __init__(self,ls_of_spectra):
        """
        Initalizes a Spectra Object:

        This object is a list of multiple members of the spectrum class.
        The Spectra Class Contains Methods to overlap and compare spectra with each other
        """

        # check instances of all list oject

        if all(isinstance(x, Spectrum) for x in ls_of_spectra):
            print("All list items of type Spectrum, Initializing Spectra Object ...")
            self.ls_of_spectra = ls_of_spectra
        else:
            raise Exception("Spectra Object can only be initialized as list of spectrum objects")
        
    def normalize_spectra(self):
        """
        Normalizes the intensity of all spectra to unity
        this happens by dividing through the maximum value
        """

        normalized_spectra = []
        for spectrum in self.ls_of_spectra:
            data = spectrum.data
            y_min = data[:,1].min()
            y_max = data[:,1].max()
            data[:,1] = (data[:,1] - y_min)/(y_max - y_min)
            spectrum.data = data
            normalized_spectra.append(spectrum)
        
        # set new list of spectra
        self.ls_of_spectra = normalized_spectra
    
    def create_ridgeline_plot(self,step=20,title="Ridgeline Plot of Spectra"):
        """
        """

        all_data = []

        min_x = float("inf")
        max_lower_bound = float("inf")


        for i, spectrum in enumerate(self.ls_of_spectra):
            df = pd.DataFrame(spectrum.data[:,0:2],columns=["x","y"])
            df["Spectrum"] = spectrum.name
            df["Order"] = i
            all_data.append(df)

            # find lowest maximum
            max_lower_bound = min(max_lower_bound, df["x"].max())
            # find minimum x value
            min_x = min(min_x, df["x"].min())

        
        combined_data = pd.concat(all_data)

        chart = alt.Chart(combined_data, width=1000, height=step).mark_area(
            interpolate="monotone",
            fillOpacity=0.8,
            stroke="lightgray",
            strokeWidth =0.5
        ).encode(
            alt.X("x:Q",
                  title= "Wave Number / cm⁻¹", 
                  sort="descending",
                  scale=alt.Scale(domain=[min_x,max_lower_bound]),
                  axis = alt.Axis(format="0.0f",grid=False)),
            alt.Y("y:Q", 
                  axis=None,
                  scale=alt.Scale(domain=[0,1.2])),
            alt.Fill("Order:O",legend=None, scale=alt.Scale(scheme="category10")),
        ).facet(
            row = alt.Row("Spectrum:N", title=None, header=alt.Header(labelAlign="left"))
        ).properties(
            title=title,
            bounds = "flush"
        ).configure_facet(
            spacing=0
        ).configure_view(
            stroke=None
        ).configure_title(
            anchor="start",
        )

        # Add selection to chart
        selection = alt.selection_interval(bind="scales")
        chart = chart.add_selection(selection)
        return chart
    
class Spectrum:
    def __init__(self,name,data):

        """
        Initializes the Spectrum object

        The spectrum object is initialized with a name and a data array. This object 
        includes basic functions for plotting and analyzing the data

        Attributes:
            name (str): Name of the spectrum
            data (np.array): Data of the spectrum
            annotations (dict): Dictionary of annotations

        """
        self.name = name
        self.data = data
        self.annotations = []
        self.plot_configuration = {
            "x_label" : "Wavenumber / cm⁻¹",
            "y_label" : "Intensity",
            "axis_format" : "0.0f",
            "interactive" : True,
            "width" : 800,
            "height" : 400,
            "color" : "black",
            "line_width" : 1,
            "categorical_scheme" : "category10",
            "color_reverse" : False,
            "ppi" : 600,
            }
        

    @classmethod
    def from_orca_spectrum(cls,filepath):
        """ 
        Reads in the IR Data created by Orca MAPSC command
        
        """
        data = pd.read_csv(filepath, delim_whitespace=True)
        data_np = data.to_numpy()
        return cls(filepath,data_np)
    
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


    def normalize_spectra(self):
        """"
        Normalizes the y data in the spectrum object to unity, this is needed for comparison of spectra
        """

        data = self.data
        # find maximum
        max_y = self.data[:,1].max()
        min_y = self.data[:,1].min()
        data[:,1] = (data[:,1] -min_y) / (max_y - min_y)
        self.data = data

    def convert_transmittance_to_absorbance(self):
        """
        Changes the y data from transmittance to absorbance

        A = 2-log(%T)
        """
        data = self.data
        data[:,1] = 2-np.log(data[:,1])
        self.data = data


    def plot_spectrum(self,title=None,legend_title="Legend",save=False):
        """ 
        Plots the spectrum using Altair as a package
        
        Parameters:
            self: The Spectrum object
            title: The title of the plot
            legend_title: Title of the Legend in the Plot
        """
        
        alt.data_transformers.disable_max_rows()
        
        if not title:
            title = "Spectrum of " + self.name

        data = pd.DataFrame(self.data[:,0:2], columns=["x","y"])
        
        data["legend"]  = self.name

        # Adjust x axis scale

        min_x = data["x"].min() - 10
        max_x = data["x"].max() + 10

        # Adjust y axis scale

        min_y  = data["y"].min()
        max_y = data["y"].max()

        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", 
                    title=self.plot_configuration["x_label"], 
                    sort="descending",
                    axis = alt.Axis(format="0.0f"),
                    scale=alt.Scale(domain=[min_x,max_x])),
            
            y=alt.Y("y", 
                    title=self.plot_configuration["y_label"],
                    scale=alt.Scale(domain=[min_y,max_y])),
            
            color = alt.Color("legend:N",legend=alt.Legend(title="Spectrum Name")).scale(scheme=self.plot_configuration["categorical_scheme"],reverse=self.plot_configuration["color_reverse"]),
            #color=alt.value("black"),
        ).properties(
            title=title,
            width = self.plot_configuration["width"],
            height = self.plot_configuration["height"]
        )

        # Create Selection

        if self.plot_configuration["interactive"]==True:
            # Create Selection
            selection = alt.selection_interval(bind="scales")
            # Add selection to chart
            chart = chart.add_selection(selection)


        if save:
            # Save the chart as a PNG file
            chart.save("spectrum.png",ppi= self.plot_configuration["ppi"])


        return chart





    def import_vci_prediction(self,molpro_out_filepath):
        """
        Imports the VCI prediction from a molpto input file
        
        Parameters:
            molpro_out_filepath (str): Filepath of the molpro output file
        """

        with open(molpro_out_filepath,"r") as file:
            lines = file.readlines()

            # Find start of VCI Calculation

            Filter_Start = "Results of VCI calculation" 
            Filter_End = "VCI/ZPVE vibrationally averaged"

            # Build extraction switch
            start = False

            # Extract the VCI Calculation

            VCI_Results_total = []
            for line in lines:
                if Filter_Start in line:
                    start = True
                    continue
                if Filter_End in line:
                    start = False
                if start:
                    VCI_Results_total.append(line)

            # Extract the Fundamentals

            Filter_Start = "Fundamentals"
            Filter_End = "Overtones" 


            # Next extraction switch
            start = False         
            VCI_Results_fundamentals = []

            for line in VCI_Results_total:
                if Filter_Start in line:
                    start = True
                    continue
                if Filter_End in line:
                    start = False
                if start:
                    VCI_Results_fundamentals.append(line)

            # Remove all blank lines
            VCI_Results_fundamentals = [line for line in VCI_Results_fundamentals if line.strip()]

            # Remove Header Line

            VCI_Results_fundamentals = VCI_Results_fundamentals[1:] 

            # We only extract assigned modes these can be found by searching for ^ symbol

            VCI_Results_fundamentals = [line for line in VCI_Results_fundamentals if "^" in line]

            fundamental_annotation_dict = {}

            for line in VCI_Results_fundamentals:
                line = line.split()
                # for each line entry in dictionary
                # First element is the mode
                mode = line[0]

                # Transform mode into wavefunction representation
                # |\nu_4^1> and so on
               
                # Fundamentals are easy always one symbol
                split_mode = mode.split("^")

                # convert first letter to subscript
                subscript = convert_subscript_superscript(split_mode[0], is_superscript=False)
                superscript = convert_subscript_superscript(split_mode[1])

                # Build String
                mode_string = "|" + "\u03bd" + subscript + superscript + ">"



                mulliken = line[1]
                E_abs = line[2]
                VCI_freq = line[3]
                IR_intensity = line[4]
                annotation = {
                    "Symmetry":mulliken, 
                    "E_abs":E_abs, 
                    "VCI_freq":VCI_freq, 
                    "IR_intensity":IR_intensity
                }
                
                # Add to fundamentals dict
                fundamental_annotation_dict[mode_string] = annotation

            # Make Annotation Dictionary

            self.annotations["fundamentals"] = fundamental_annotation_dict


            # Next the overtones

            Filter_Start = "Overtones"
            Filter_End = "Combination Bands"
            start = False
            VCI_Results_overtones = []

            for line in VCI_Results_total:
                if Filter_Start in line:
                    start = True
                    continue
                if Filter_End in line:
                    start = False
                if start:
                    VCI_Results_overtones.append(line)
            
            # Remove Blank lines and header
            VCI_Results_overtones = [line for line in VCI_Results_overtones if line.strip()]
            VCI_Results_overtones = VCI_Results_overtones[1:]
            # We only extract assigned modes these can be found by searching for ^ symbol
            VCI_Results_overtones = [line for line in VCI_Results_overtones if "^" in line]
            # Exclute lines that start with "Multi"
            VCI_Results_overtones = [line for line in VCI_Results_overtones if "Multi" not in line] 

            overtone_annotation_dict = {}

            for line in VCI_Results_overtones: 
                line = line.split()
                mode = line[0]

                split_mode = mode.split("^")

                # convert first letter to subscript
                subscript = convert_subscript_superscript(split_mode[0], is_superscript=False)
                superscript = convert_subscript_superscript(split_mode[1])

                # Build String

                mode_string = "|" + "\u03bd" + subscript + superscript + ">"


                mulliken = line[1]
                E_abs = line[2]
                VCI_freq = line[3]
                IR_intensity = line[4]
                annotation = {
                    "Symmetry":mulliken, 
                    "E_abs":E_abs, 
                    "VCI_freq":VCI_freq, 
                    "IR_intensity":IR_intensity
                }
                overtone_annotation_dict[mode_string] = annotation
        
            self.annotations["overtones"] = overtone_annotation_dict

            # Combination bands

            Filter_Start = "Combination bands"
            Filter_End = "VCI/ZPVE vibrationally averaged"
            start = False
            VCI_Results_combinations = []

            for line in VCI_Results_total:
                if Filter_Start in line:
                    start = True
                    continue
                if Filter_End in line:
                    start = False
                if start:
                    VCI_Results_combinations.append(line)

            # Remove Blank lines and header
            VCI_Results_combinations = [line for line in VCI_Results_combinations if line.strip()]
            VCI_Results_combinations = VCI_Results_combinations[1:]
            # We only extract assigned modes these can be found by searching for ^ symbol
            VCI_Results_combinations = [line for line in VCI_Results_combinations if "^" in line]
 
            combination_annotation_dict = {}

            for line in VCI_Results_combinations:
                # Note here we have always the Combinations
                line = line.split()
                comb = str(line[0]) + "-" + str(line[1])
                
                # split comb string

                split_comb = comb.split("-")

                wf = []
                for s in split_comb:
                    s_split = s.split("^")

                    # convert first letter to subscript
                    subscript = convert_subscript_superscript(s_split[0], is_superscript=False)
                    superscript = convert_subscript_superscript(s_split[1])
                    # build intermediate
                    wf.append("\u03bd" + subscript + superscript)
                
                # Build String
                comb_string = "|" + wf[0] + wf[1] + ">"

                mulliken = line[2]
                E_abs = line[3]
                VCI_freq = line[4]
                IR_intensity = line[5]
                annotation = {
                    "Symmetry":mulliken, 
                    "E_abs":E_abs, 
                    "VCI_freq":VCI_freq, 
                    "IR_intensity":IR_intensity
                }
                combination_annotation_dict[comb_string] = annotation

            self.annotations["combinations"] = combination_annotation_dict



        
    
    def derivative(self,data):
        """ 
        Calculates the numerical deriative using central differences

        Returns:
            np.array: The Numerical Derivative [x,y']
        """
        return np.gradient(self.data[:,1],self.data[:,0])
    
    def higher_order_derivative(self,order=2):
        """
        Calculates higher order derivatives using central differences
        """
        # build up array of derivatives
        # first column = x
        # second column = y
        # third column = y'
        # ...
        # shape of array is (len(self.data[:0   ]),order+1)
        data = np.zeros((len(self.data),order+1))
        data[:,0] = self.data[:,0]
        data[:,1] = self.data[:,1]
        for i in range(2,order+1):
            data[:,i] = np.gradient(data[:,i-1],data[:,0])
        return data 
    

    def derivative_spectrum(self):
        """  
        Calculates the derivative of spectrum and gives it back as a new Spectrum object

        Returns:
            Spectrum: The derivative spectrum
        """

        derivative = self.derivative(self.data)
        return Spectrum(self.name + " Derivative", np.column_stack((self.data[:,0],derivative)))

    def plot_derivative(self,title=None, legend_title="Legend", save=False):
        """ 
        Plots the Derivative Spectrum of the Given Spectrum
        """
        alt.data_transformers.disable_max_rows()
        if not title:
            title = "Derivative Spectrum of " + self.name

        derivative = self.derivative(self.data[:,0:2])
        data = np.zeros((len(self.data),2))
        data[:,0] = self.data[:,0]
        data[:,1] = derivative
        data = pd.DataFrame(data, columns=["x","y"])
        data["Legend"]  = legend_title


        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title=self.plot_configuration["x_label"] , sort="descending").axis(format="0.0f"),
            y=alt.Y("y", title=self.plot_configuration["y_label"]),
            color = alt.Color("legend:N",legend=alt.Legend(title="Spectrum Name")).scale(scheme=self.plot_configuration["categorical_scheme"],reverse=self.plot_configuration["color_reverse"]),
        ).properties(
            title=title,
            width = self.plot_configuration["width"],
            height = self.plot_configuration["height"]
        )

        if self.plot_configuration["interactive"]==True:
            selection = alt.selection_interval(bind="scales")
            # Add selection to chart
            chart = chart.add_selection(selection)

        if save:
            # Save the chart as a PNG file
            chart.save("spectrum_derivative.png",ppi= self.plot_configuration["ppi"])

        return chart

    def plot_spectral_window(self,x_min,x_max,title=None,legend_title="Legend",save=False):
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

        if not title:
            title = "Spectral Window of " + self.name

        data["Legend"]  = legend_title


        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title=self.plot_configuration["x_label"], sort="descending").axis(format="0.0f"),
            y=alt.Y("y", title=self.plot_configuration["y_label"]),
            color=alt.Color("legend:N",legend=alt.Legend(title=legend_title)).scale(scheme=self.plot_configuration["categorical_scheme"],reverse=self.plot_configuration["color_reverse"]),
        ).properties(
            title=title,
            width = self.plot_configuration["width"],
            height = self.plot_configuration["height"]
        )

        # Create Selection

        if self.plot_configuration["interactive"]==True:
            # Create Selection
            selection = alt.selection_interval(bind="scales")
            # Add selection to chart
            chart = chart.add_selection(selection)


        # Add selection to chart

        chart = chart.add_selection(selection)

        if save:
            # Save the chart as a PNG file
            chart.save("spectrum_spectral_window.png",ppi= self.plot_configuration["ppi"])


        return chart

    def plot_higher_derivatives_spectral_window(self,x_min,x_max,order=2,title=None,legend_title="Legend",save=False):
        """
        Plots Higher Order Derivatives of a given spectral window
        """

        # TODO FIX THIS BROKEN AS FUNCTION
        derivatives = self.higher_order_derivative(order=order)
        # mask
        mask = (derivatives[:,0] >= x_min) & (derivatives[:,0] <= x_max)
        derivatives = derivatives[mask]
        # Labels for columns
        labels = ["x"] + [f"y^({i})" for i in range(1,order+1)]
        derivatives = pd.DataFrame(derivatives, columns=labels)

        # Set the title
        if not title:
            title = "Higher Derivative Spectrum of " + self.name

        alt.data_transformers.disable_max_rows()

        # Create the chart
        # make subplot for each derivative stack horizontally 

        charts = []
        for i in range(1,order+1):
            # This is ugly but otherwise it doesnt work
            data = derivatives.iloc[:,[0,i]]

            chart = alt.Chart(data).mark_line().encode(
                x=alt.X("x", title=self.plot_configuration["x_label"], sort="descending").axis(format="0.0f"),
                y=alt.Y(f"y^({i}):Q", title=f"y^({i})"),
                color=alt.value("black")
            ).properties(
                title=f"{title} Derivative of Order {i}",
                width = self.plot_configuration["width"] * order,
                height = self.plot_configuration["height"] * order
            )

        
        combined_chart = alt.hconcat(*charts).resolve_scale(
            y='independent'
        )
        # Create Selection
        if self.plot_configuration["interactive"]==True:
            # Create Selection
            selection = alt.selection_interval(bind="scales")
            # Add selection to chart
            combined_chart = combined_chart.add_selection(selection)
        # Add selection to chart
        combined_chart = combined_chart.add_selection(selection)
        if save:
            # Save the chart as a PNG file
            combined_chart.save("spectrum_higher_derivative.png",ppi= self.plot_configuration["ppi"])
        return combined_chart

        



    def interactive_integration(self, title="Interactive Peak Integration"):
        """ 
        Allows for interactive integration where the user is able to set the boundaries of integration
        using the altair package
        """

        alt.data_transformers.disable_max_rows()


        data = pd.DataFrame(self.data[:,0:2], columns=["x","y"])

        # adjust axis scale
        min_x = data["x"].min() - 10
        max_x = data["x"].max() + 10

        base = alt.Chart(data).mark_line().encode(
            x=alt.X("x", 
                    title=self.plot_configuration["x_label"], 
                    sort="descending",
                    axis = alt.Axis(format="0.0f"),
                    scale=alt.Scale(domain=[min_x,max_x])),
            y=alt.Y("y",
                    title=self.plot_configuration["y_label"]),

            color=alt.value("black")
        ).properties(
            title = title,
            width = self.plot_configuration["width"],
            height = self.plot_configuration["height"]
        )

        # Add a brush selection for x-axis

        brush = alt.selection_interval(encodings=["x"], resolve="intersect")

        # Create the background chart with brush

        background = base.add_params(brush)

        # Create the selected chart area

        selected = base.transform_filter(brush).mark_area( 
            color = "blue",
            opacity = 0.5
        )

        # Calculate the integral under the curve for selected interval

        integral_text = alt.Chart(data).mark_text(
            align="center",
            baseline="middle",
            fontSize=14,
            color = "black",
            dx=10,
            dy=10
        ).encode(
            x = alt.value(400),
            y = alt.value(50)
        ).transform_filter(
            brush
        ).transform_aggregate(
            integral="sum(y)"
        ).encode(

            text=alt.Text("integral:Q", format=".4f")
        )


        # combine the charts

        chart = (background + selected + integral_text).properties(
            width = 800,
            height = 400
        )

        return chart



    def find_maxima(self, threshold=0.1, distance=1):
        """
        Finds the Indices of local maxima and minima using scipy find peaks

        Parameters:
            treshold: The treshold for peak picking
            distance: The minimum distance between peaks 
        """

        # Find normal peaks
        peaks, _ = find_peaks(self.data[:,1], height=threshold, distance=distance)
        
        # Find also negative peaks

        negative_peaks, _ = find_peaks(-self.data[:,1], height=threshold, distance=distance)
        peaks = np.concatenate([peaks,negative_peaks])
        return peaks
    
    

    def add_annotations(self,annotations):
        """
        Adds a given list of annotations to the spectrum object

        Attributes:
            annotations: A list of annotation objects
        """ 

        self.annotations.extend(annotations)

    def plot_spectrum_annotations_vertical(self,title=None):
        """ 
        Plots a given spectrum with annotations as dashed vertical lines
        """
        data = pd.DataFrame(self.data[:,0:2], columns=["x","y"])
        alt.data_transformers.disable_max_rows()
        if not title:
            title = "Spectrum of " + self.name
        data["Legend"]  = self.name
        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title=self.plot_configuration["x_label"], sort="descending").axis(format="0.0f"),
            y=alt.Y("y", title=self.plot_configuration["y_label"]),
            color=alt.Color("legend:N",legend=alt.Legend(title="Spectrum Name")).scale(scheme=self.plot_configuration["categorical_scheme"],reverse=self.plot_configuration["color_reverse"]),
        ).properties(
            title=title,
            width = self.plot_configuration["width"],
            height = self.plot_configuration["height"]
        )

        # Create Selection
        if self.plot_configuration["interactive"]==True:
            # Create Selection
            selection = alt.selection_interval(bind="scales")
            # Add selection to chart
            chart = chart.add_selection(selection)
        # Add selection to chart
        chart = chart.add_selection(selection)

        # Add the annotations

        

        annotations = []
        for annotation in self.annotations:

            #search data to find nearest x point 
            x_search = annotation.x_position

            # Find the nearest x value in the data
            nearest_x = min(data["x"], key=lambda x: abs(x - x_search))

        

            # Create the vertical line
            line = alt.Chart(data).mark_rule(color="red",strokeDash=[5,5]).encode(
                x=alt.X("x:Q"),
                y=alt.Y("y:Q"),
                color=alt.value(annotation.color)
            ).transform_filter(
                alt.datum.x == nearest_x
            )

            # Create the text annotation
            text = alt.Chart(data).mark_text(
                align="left",
                baseline="middle",
                dx = 10,
                dy = -20,
            ).encode(
                x=alt.X("x:Q"),
                y=alt.Y("y:Q"),
                text=alt.value(f"{annotation.description}"),
            ).transform_filter(
                alt.datum.x == nearest_x
            )


            chart = chart + line + text
        
        return chart
    



    def plot_spectrum_with_vci_annotations(self,title="None",mode=0):
        """
        Plots the spectrum with the given VCI annotations

        Mode: 
            0 - just fundamentals
            1 - fundamentals + overtones
            2 - fundamentals + overtones + combination bands
        """

        alt.data_transformers.disable_max_rows()

        data = pd.DataFrame(self.data[:,0:2], columns=["x","y"])

        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title="Wave Number / cm⁻¹", sort="descending").axis(format="0.0f"),
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


        # Add the annotations
        annotations = []
        annotations_data_fundamentals = []
        annoations_data_overtones = []
        annotations_data_combinations = []
        annotations_data_fundamentals = pd.DataFrame([(key, value["VCI_freq"], value["IR_intensity"], value["Symmetry"]) for key, value in self.annotations["fundamentals"].items()], columns=["mode","VCI_freq","IR_intensity","Symmetry"])

        annotations_data_fundamentals["text"] = annotations_data_fundamentals.apply(
            lambda row: f"Mode: {row['mode']}, Freq: {row['VCI_freq']}, \n Sym: {row['Symmetry']}", axis=1
        )

        # if mode == 1 import overtones
        if mode == 1:
            annotations_data_overtones = pd.DataFrame([(key, value["VCI_freq"], value["IR_intensity"], value["Symmetry"]) for key, value in self.annotations["overtones"].items()], columns=["mode","VCI_freq","IR_intensity","Symmetry"])
            annotations_data_overtones["text"] = annotations_data_overtones.apply(
                lambda row: f"Mode: {row['mode']}, Freq: {row['VCI_freq']}, \n Sym: {row['Symmetry']}", axis=1
            )
        
        if mode == 2:
            annotations_data_overtones = pd.DataFrame([(key, value["VCI_freq"], value["IR_intensity"], value["Symmetry"]) for key, value in self.annotations["overtones"].items()], columns=["mode","VCI_freq","IR_intensity","Symmetry"])
            annotations_data_overtones["text"] = annotations_data_overtones.apply(
                lambda row: f"Mode: {row['mode']}, Freq: {row['VCI_freq']}, \n Sym: {row['Symmetry']}", axis=1
            )
            annotations_data_combinations = pd.DataFrame([(key, value["VCI_freq"], value["IR_intensity"], value["Symmetry"]) for key, value in self.annotations["combinations"].items()], columns=["mode","VCI_freq","IR_intensity","Symmetry"])
            annotations_data_combinations["text"] = annotations_data_combinations.apply(
                lambda row: f"Mode: {row['mode']}, Freq: {row['VCI_freq']}, \n Sym: {row['Symmetry']}", axis=1
            )

    

        
        lines_fund = alt.Chart(annotations_data_fundamentals).mark_rule(color="red",strokeDash=[5,5]).encode(
            x="VCI_freq:Q"
        )

        if mode == 1:
            lines_overtones = alt.Chart(annotations_data_overtones).mark_rule(color="blue",strokeDash=[5,5]).encode(
                x="VCI_freq:Q"
            )
            lines = lines_fund + lines_overtones

        if mode == 2:
            lines_overtones = alt.Chart(annotations_data_overtones).mark_rule(color="blue",strokeDash=[5,5]).encode(
                x="VCI_freq:Q"
            )

            lines_combinations = alt.Chart(annotations_data_combinations).mark_rule(color="green",strokeDash=[5,5]).encode(
                x="VCI_freq:Q"
            )
            lines = lines_fund + lines_overtones + lines_combinations

        
        text_fund = alt.Chart(annotations_data_fundamentals).mark_text(
            align="left",
            baseline="middle",
            dx = 10,
            dy = -20,

        ).encode(
            x="VCI_freq:Q", 
            text=alt.Text("text:N")
        )

        if mode == 1:
            text_overtones = alt.Chart(annotations_data_overtones).mark_text(
                align="left",
                baseline="middle",
                dx = 5,
                dy = -5,
            ).encode(
                x="VCI_freq:Q", 
                text=alt.Text("text:N")
            )

            text = text_fund + text_overtones
        
        if mode == 2:
            text_overtones = alt.Chart(annotations_data_overtones).mark_text(
                align="left",
                baseline="middle",
                dx = 20,
                dy = -35,
            ).encode(
                x="VCI_freq:Q", 
                text=alt.Text("text:N")
            )

            text_combinations = alt.Chart(annotations_data_combinations).mark_text(
                align="left",
                baseline="middle",
                dx = 5,
                dy = -5,
            ).encode(
                x="VCI_freq:Q", 
                text=alt.Text("text:N")
            )

            text = text_fund + text_overtones + text_combinations


        combined = chart + lines + text

        return combined


    def plot_spectrum_with_peaks(self,peaks,title=None):
        """ 
        Plots the spectrum and marks a given list of peaks

        Parameters:
            peaks: List of peaks
            title: Title of the plot
        """

        alt.data_transformers.disable_max_rows()

        if not title:
            title = "Spectrum of " + self.name
 

        data = pd.DataFrame(self.data[:,0:2], columns=["x","y"])
        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title="Wave Number / cm⁻¹", sort="descending").axis(format="0.0f"),
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

        # Create the scatter plot for the peaks

        data_peaks = data.iloc[peaks]

        peaks_plot = alt.Chart(data_peaks).mark_point(color="red").encode(
            x="x",
            y="y"
        )

        # combine chart

        chart = chart + peaks_plot

        return chart

     





    


    def plot_spectrum_peak_picking(self,title=None,threshold=0.01,distance=1, width=1, vertical_shift= -10):
        """
        Performs peakpicking on a total given spectrum object using the Scipy find_peaks function

        Attributes:
            title: Title of the plot
            threshold: Threshold for peak picking
            distance: Minimum horizontal distance between two peaks
            width: Minimum width of the peaks        
        """

        alt.data_transformers.disable_max_rows()

        if not title:
            title = "Spectrum of " + self.name

        data = pd.DataFrame(self.data[:,0:2], columns=["x","y"])

        # Find the peaks

        peaks, _ = find_peaks(data["y"], height=threshold,distance=distance,width=width)


        # Create the chart

        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title="Wave Number / cm⁻¹", sort="descending").axis(format="0.0f"),
            y=alt.Y("y", title="Intensity"),
            color=alt.value("black")
        ).properties(
            title=title,
            width = 800,
            height = 400
        )

        # Create the scatter plot for the peaks

        data_peaks = data.iloc[peaks]

        peaks_chart = alt.Chart(data_peaks).mark_rule(color="red",strokeDash=[5,5]).encode(
            x="x",
            y="y"
        )

        annotations = alt.Chart(data_peaks).mark_text(
            align = "center",
            baseline = "bottom",
            dx = vertical_shift,
            dy = -10,
            angle=90,
            fontSize=10,
            color="black"
        ).encode(
            x = "x",
            y = "y",
            text=alt.Text("x:Q",format=".2f")
        ).transform_calculate(
            text = "datum.x +  ' cm⁻¹'"
        ).encode(
            text="text:N"
        )

        # Create Selection

        selection = alt.selection_interval(bind="scales")

        # Add selection to chart

        chart = chart.add_selection(selection)

        # Combine chart
        
        chart = chart + peaks_chart + annotations

        return chart,peaks


    
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

        if not title:
            title = "Spectral Window of " + self.name

        

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

        return chart,peaks
    
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
                x=alt.X("x", title="Wave Number / cm⁻¹", sort="descending").axis(format="0.0f"),
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
                "text": [f"Integral: {integral:.4f}"]
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