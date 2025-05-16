import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import altair as alt
from pybaselines import Baseline as Baseline_fit
import py3Dmol


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms




# Importing the necessary packages for the deconvolution
from scipy.optimize import curve_fit

from scipy.signal import savgol_filter


# General Functions

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


class Molecule:
    def __init__(self,name,coords,atoms):
        """ 
        Initializes an object of the molecule class
        """

        self.name = name
        self.coords = coords
        self.atoms = atoms
        self.mol = None

    def atom_coords_to_mol(self,bond_thresholds=None):
        """ 
        Creates a RDKit molecule from lsits of atoms and coordinates

        Args:
            atoms (list): List of atoms
            coords (list): List of coordinates

        Returns:
            RDKit molecule object
        """

        mol = Chem.RWMol()

        # Add atoms to the molecule

        atom_list = remove_numeration_atoms(self.atoms)
       
        for atom in atom_list:
            atom = Chem.Atom(atom)
            mol.AddAtom(atom)
 

        # Add coordinates to the molecule
        conf = Chem.Conformer(len(atom_list))
        for i, (x,y,z) in enumerate(self.coords):
            conf.SetAtomPosition(i, (x,y,z))

        
        # Default bond threshholds for common pairs
        if bond_thresholds is None:
            bond_thresholds = {
                ("C","C"): 1.7,
                ("C","O"): 1.5,
                ("C","N"): 1.5,
                ("C","H"): 1.1,
                ("O","O"): 1.5,
                ("O","N"): 1.5,
                ("N","N"): 1.5,
                ("H","H"): 1.0  
            }
        # Add bonds to the molecule
        for i in range(mol.GetNumAtoms()):
            for j in range(i+1, mol.GetNumAtoms()):
                # calculate pairwise distance
                dist = rdMolTransforms.GetBondLength(conf,i,j)
                print(dist)
                sym_i = mol.GetAtomWithIdx(i).GetSymbol()
                sym_j = mol.GetAtomWithIdx(j).GetSymbol()

                # Get Threshold
                threshold = bond_thresholds.get((sym_i,sym_j)) or \
                    bond_thresholds.get((sym_j,sym_i))
                
                if dist >= threshold:
                    mol.AddBond(i,j,Chem.rdchem.BondType.SINGLE)



        # Convert to regular molecule
        rdmol = mol.GetMol()
        rdmol.AddConformer(conf)

    

                               

        self.mol = rdmol

    @staticmethod
    def visualize_plane_of_symmetry(viewer, origin=(0,0,0), normal=(1,0,0), size=10, color="gray"):

        # Create Plane Geometry
        x,y,z = origin
        nx,ny,nz = normal

        plane_geom = {
            "color" : color,
            "opacity": 0.5,
            "vertices": [
                [x - size, y - size,z],
                [x + size, y-size, z],
                [x + size, y + size,z],
                [x - size, y + size,z]
            ],
            "faces": [[0,1,2,3]]
        }

        viewer.addModel(str(plane_geom), "json")
        viewer.setStyle({"model": -1}, {"line": {}})
    
    
    @classmethod
    def visualize_mol_3d(cls, mol, size=(400,400), style="stick"):
        
        pdb_block = Chem.MolToPDBBlock(mol)
        print(pdb_block)

        viewer = py3Dmol.view(width=size[0], height=size[1])

        viewer.addModel(pdb_block, "pdb")
        viewer.setStyle({"stick": {"radius":0.1}, "sphere": {"radius": 0.3}})
    
        viewer.zoomTo()
        viewer.spin(1)

        return viewer.show()

    @classmethod
    def import_molecule_molpro(cls,filepath):
        """
        Imports a molecule from a molpro output file
        """

        with open(filepath,"r") as file:
            lines = file.readlines()


            # Find start of geometry

            Filter_Start = "Atomic Coordinates"

            Filter_End = "Gradient norm at"

            start = False

            # Extract the geometry
            geometry_total = []

            for line in lines:
                if Filter_Start in line:
                    start = True
                    continue
                if Filter_End in line:
                    start = False
                if start:
                    geometry_total.append(line)
            
            # Remove all blank lines
            geometry_total = [line for line in geometry_total if line.strip()]

            # First line then is the header
            geometry_total = geometry_total[1:]
            

            # Now we can extract the coordinates
            coords = []
            atoms = []
            for line in geometry_total:
                line = line.split()
                atoms.append(line[1])
                coords.append([float(line[3]),float(line[4]),float(line[5])])

            coords = np.array(coords)
            atoms = np.array(atoms)

            # Create the molecule object
            return cls(filepath,coords,atoms) 

     

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

        # Add annotation dictionary
        self.annotations = {}

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

    def derivative_spectrum(self):
        """  
        Calculates the derivative of spectrum and gives it back as a new Spectrum object

        Returns:
            Spectrum: The derivative spectrum
        """

        derivative = self.derivative(self.data)
        return Spectrum(self.name + " Derivative", np.column_stack((self.data[:,0],derivative)))


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
    
    @classmethod
    def initialize_annotation_dict(cls,mode="chemist_not"):
        """
        Initializes a empty notation dictionary which can be filled by the user

        mode = chemist_not (chemist's notation based on the principal motion pattern)

        mode = spectro (spectroscopist's notation based on the molecular point group)

        mode = physicist (notation carrying quantum mechanical information)
        """

        chemist_notation = {
           "stretching": "\u03bd",
           "bending": "\u03b4",
           "rocking": "\u03c1",
           "wagging": "\u03a9",
           "twisting": "\u03c4" 
        }

        mode = input("Please enter the mode of the annotation dictionary: chemist_not, spectro, physicist")
        if mode == "chemist_not":
            vib_type = input("Please enter the type of vibration: \n "  +
                                "stretching, bending, rocking, wagging, twisting")
            if vib_type in chemist_notation.keys():
                vib_type = chemist_notation[vib_type]
                annotation = {
                    "vib_type": vib_type,
                    "freq": None,
                    "group": None,
                    "description": None
                }
            else:
                print("Invalid Vibration Type")
                return None
            

        return annotation

    def clear_annotations(self):
        """
        Clears the current annotations from the object
        """

        self.annotations = {}
        print("Current annotations cleared")

    def add_annotation(self,annotation, num=1):
        """
        Adds one of the three standardized annotations to the spectrum object
        
        Parameters:


        """
        
        # Check if annotations are already present
        if not self.annotations:
            self.annotations = {}


        self.annotations[num] = annotation
        print("Annotation added to spectrum object")

    def plot_annotations_user(self):
        """
        Plots all current user defined annotations into the spectrum
        """

        alt.data_transformers.disable_max_rows()

        data = pd.DataFrame(self.data[:,0:2], columns=["x","y"])

        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title="Wave Number / cm⁻¹", sort="descending").axis(format="0.0f"),
            y=alt.Y("y", title="Intensity"),
            color=alt.value("black")
        ).properties(
            title="Spectrum of " + self.name,
            width = 800,
            height = 400
        )

        # Create Selection
        selection = alt.selection_interval(bind="scales")

        # Add selection to chart
        chart = chart.add_selection(selection)

        # Make vertical lines for each annotation

        annotations = []
        for key, value in self.annotations.items():
            if value["freq"]:
                annotations.append(alt.Chart(pd.DataFrame({"x":[value["freq"]]})).mark_rule(color="red",strokeDash=[5,5]).encode(
                    x="x"
                ))
        
        # Add vib_type and group as text

        annotations_text = []
        for key, value in self.annotations.items():
            if value["freq"]:
                annotations_text.append(alt.Chart(pd.DataFrame({"x":[value["freq"]], "vib": [value["vib_type"] + value["group"]]})).mark_text(align="left", baseline="middle", dx=7, dy=-7, fontSize=14).encode(
                    x="x",
                    # Add the vib_type and group as text
                    text=alt.Text("vib:N")
                ))


        chart = chart + alt.layer(*annotations) + alt.layer(*annotations_text)

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

     
    def interactive_integration(self):
        """ 
        Allows for interactive integration where the user is able to set the boundaries of integration
        using the altair package
        """

        alt.data_transformers.disable_max_rows()

        data = pd.DataFrame(self.data[:,0:2], columns=["x","y"])

        base = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title="Wave Number / cm⁻¹", sort="descending").axis(format="0.0f"),
            y=alt.Y("y", title="Intensity"),
            color=alt.value("black")
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


    def plot_spectrum(self,title=None,interactive=True,color="black",legend_title="Legend"):
        """ 
        Plots the spectrum using Altair as a package
        
        Parameters:
            self: The Spectrum object
            title: The title of the plot
            interactive: If True, plot is interactive
            color: Gives the Color of the Data in the Plot
            legend_title: Title of the Legend in the Plot
        """
        
        alt.data_transformers.disable_max_rows()
        
        if not title:
            title = "Spectrum of " + self.name

        



        data = pd.DataFrame(self.data[:,0:2], columns=["x","y"])
        
        data["Legend"]  = self.name

        # Adjust axis scale

        min_x = data["x"].min() - 10
        max_x = data["x"].max() + 10

        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", 
                    title="Wave Number / cm⁻¹", 
                    sort="descending",
                    axis = alt.Axis(format="0.0f"),
                    scale=alt.Scale(domain=[min_x,max_x])),
            
            y=alt.Y("y", title="Intensity"),
            
            color = alt.Color("legend:N",legend=alt.Legend(title="Spectrum Name"))
            #color=alt.value("black"),
        ).properties(
            title=title,
            width = 800,
            height = 400
        )

        # Create Selection

        if interactive==True:
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
            x=alt.X("x", title="Wave Number / cm⁻¹", sort="descending").axis(format="0.0f"),
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

        if not title:
            title = "Spectral Window of " + self.name



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

        if not title:
            title = "Baselinefitting " + self.spectrum.name

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
    
    def savgol_filter(self, poly_order = 3, window_size= 10, offset = 0.05):
        """
        Applies a Savitzky-Golay filter to the spectrum.

        Uses the Respective Scipy Method

        Some key observations to note:

            + Small Window Size Low Degree = Smooths data but doesnt capture overall trend
            + Small Window Size High Degree = Captures trend but may overfits data
            + Large Window Size Low Degree = Provides stable smoothing but may smooth out important features
            + Large Window Size, High Degree = May introduce artifacts
        """


        data = self.spectrum.data

        y_data = data[:,1]

        # Apply Savitzky-Golay filter

        y_smooth = savgol_filter(y_data,window_length=window_size, polyorder=poly_order)

        plt.plot(data[:,0],y_data, label="Original Data")
        plt.plot(data[:,0],y_smooth, label="Savitzky-Golay Filtered Data")
        plt.title("Savitzky-Golay Filtering")
        plt.xlabel("Wavenumber / cm⁻¹")
        plt.ylabel("Intensity")
        plt.legend()
        plt.show()

        # Return new Object with smoothed data

        Spectrum_smoothed = Spectrum(self.spectrum.name + " Smoothed", np.column_stack((data[:,0],y_smooth)))

        return Baseline(Spectrum_smoothed)

    def rolling_average_smoothing(self, window_size=2, mode="full"):
        """ 
        Computes a rolling average and smoothes the spectrum
        """


        data = pd.DataFrame(self.spectrum.data[:,0:2], columns=["x","y"])

        data["y_smoothed"] = data["y"].rolling(window=window_size, min_periods=1, center=True).mean()

        # Drop Nanas
        data = data.dropna()

        x_smoothed = data["x"].values
        y_smoothed = data["y_smoothed"].values

        plt.plot(data["x"],data["y"], label="Original Data")
        plt.plot(x_smoothed,y_smoothed, label="Rolling Average Smoothed Data")
        plt.title("Rolling Average Smoothing")
        plt.xlabel("Wavenumber / cm⁻¹")
        plt.ylabel("Intensity")
        plt.legend()
        plt.show()

        # Return Object with Smoothed Data

        Spectrum_smoothed = Spectrum(self.spectrum.name + " Smoothed", np.column_stack((x_smoothed,y_smoothed)))

        return Baseline(Spectrum_smoothed)

    def return_spectrum(self):
        """
        Returns the spectrum object
        """
        return self.spectrum






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


class Deconvolution:
    """
    Class for different deconvolution methods for peaks
    """

    def __init__(self,spectrum,peaks):
        """
        Initializes the Deconvolution class

        Attr:
            spectrum: Spectrum object
            peaks: Peaks of the spectrum given as list
            fits: a list of fits
        """
        self.spectrum = spectrum
        self.peaks = peaks 
        self.fits = None


    # Functions

    @classmethod

    def gaussian(cls,x,amplitude,mean,stddev):
        """
        Gaussian Function
        """
        return amplitude * np.exp(-((x-mean)**2)/(2*stddev**2))
    
    @classmethod

    def poly_gaussian(cls,x,max_height,x_mean,stdev,m):
        """
        Fitting Function for a Polynomially Modified Gaussian (PMG)
        
        https://doi.org/10.1016/j.aca.2012.10.035

        Attributes:
            x: x data
            max_height: maximum height of the peak
            mean: mean of the peak
            stdex_v: standard deviation of the peak
            m: polynomial order
        """

        return max_height * np.exp(-0.5 * ((x-x_mean)**2)/((stdev + m * (x - x_mean))**2))
    

    @classmethod
    def barplot_integrals(cls, list_of_annotations):
        """
        Given a list of annotations makes a bar plot of the integrals
        """

        x = [annotation["x"] for annotation in list_of_annotations]
        y = [annotation["y"] for annotation in list_of_annotations]
        integrals = [float(annotation["text"].split(":")[1]) for annotation in list_of_annotations]

        data = pd.DataFrame(
            {
                "x": x,
                "y": y,
                "integrals": integrals
            }
        )

        chart = alt.Chart(data).mark_bar().encode(
            x = alt.X("x:O", title="Wavenumber", axis=alt.Axis(labelAngle=-45)),
            y = alt.Y("integrals:Q", title="Integral"),
        ).properties(
            title="Peak Integrals",
            width = 800,
            height = 400
        )

        text = chart.mark_text(align="center", baseline="middle", dy=-5).encode(
            text="integrals:Q"
        ).encode(
            text=alt.Text("integrals:Q", format=".3f")
        )


        return chart + text


    @classmethod
    def heatmap_integrals(cls,list_of_annotations):
        """  
        Takes a list of annotations as an input and displays all the rations as a heatmap
        """

        # Extrac x,y values and group as tuples

        x = [annotation["x"] for annotation in list_of_annotations]
        y = [annotation["y"] for annotation in list_of_annotations]
        integrals = [float(annotation["text"].split(":")[1]) for annotation in list_of_annotations]
    

        ratios = []
        for i in range(len(integrals)):
            for j in range(len(integrals)):
                ratio = integrals[i]/integrals[j]
                ratios.append((x[i],x[j],ratio))

        ratios_df = pd.DataFrame(
            ratios,
            columns = ["Peak 1","Peak 2","Ratio"]
        )

        # Setup color scale

        heatmap = alt.Chart(ratios_df).mark_rect().encode(
            x = alt.X("Peak 1:O", title="Peak 1 Wavenumber"),
            y = alt.Y("Peak 2:O", title="Peak 2 Wavenumber"),
            color = alt.Color("Ratio:Q", scale=alt.Scale(scheme="greenblue"), legend=alt.Legend(title="Ratio")),
            tooltip = ["Peak 1","Peak 2","Ratio"]
        ).properties(
            title="Peak Ratios",
            width = 600,
            height = 400
        )

        text = heatmap.mark_text(baseline="middle").encode(
            text=alt.Text("Ratio:Q", format=".2f"),
            color=alt.condition(
                alt.datum.Ratio == 1,
                alt.value("red"),
                alt.value("black")
            )
        )

        heatmap = heatmap + text


        return heatmap


        

    def fit_gaussian(self,peak, broadness=10):
        """
        Fits a Gaussian to a given peak
        """

        x = self.spectrum.data[:,0]
        y = self.spectrum.data[:,1]

        # peak is given as index add a broadness to the peak
        # add the interval of [-broadness,broadness] to the peak

        peak_interval = np.arange(peak-broadness,peak+broadness)

        # Select interval from data

        x_peak = x[peak_interval]

        y_peak = y[peak_interval]

        # Fit the Gaussian

        popt, _ = curve_fit(self.gaussian,x_peak,y_peak,p0=[y[peak],x[peak],1])

        return popt
    
    def fit_poly_gaussian(self,peak, broadness=10, m=1):
        """
        Fits the Polynomially Modified Gaussian to a given peak

        Attributes:
            peak: Peak index
            broadness: Broadness of the peak
            m: Model Parameter for skewness
        """

        x = self.spectrum.data[:,0]
        y = self.spectrum.data[:,1]

        # peak is given as index and add broadness as interval

        peak_interval = np.arange(peak-broadness,peak+broadness)


        # Select interval from data

        x_peak = x[peak_interval]
        y_peak = y[peak_interval]

        # Fit the Polynomially Modified Gaussian

        popt, _ = curve_fit(self.poly_gaussian,x_peak,y_peak,p0=[y[peak],x[peak],1,m])

        return popt
    
    def plot_poly_gaussian_fit(self,peak, broadness=10, m=1):
        """
        Helper Function that plots the polynomially modified gaussian fit
        """

        x = self.spectrum.data[:,0]
        y = self.spectrum.data[:,1]

        popt = self.fit_poly_gaussian(peak,broadness,m)

        fit = self.poly_gaussian(x,*popt)

        plt.plot(x,y, label="Spectrum", color="black")
        plt.plot(x,fit, label="Poly Gaussian Fit", color="red")
        plt.gca().invert_xaxis()
        plt.legend()
        plt.show()




    def plot_gaussian_fit(self,peak, broadness=10):
        """
        Helper Function that plots the gaussian fit
        """

        x = self.spectrum.data[:,0]
        y = self.spectrum.data[:,1]

        popt = self.fit_gaussian(peak,broadness)

        fit = self.gaussian(x,*popt)

        plt.plot(x,y, label="Spectrum", color="black")
        plt.plot(x,fit, label="Gaussian Fit", color="red")
        plt.gca().invert_xaxis()
        plt.legend()
        plt.show()

    # Function to fit multiple gaussians

    def fit_multiple_gaussians_and_plot(self, broadness=10):
        """
        General Function to Fit all the Peaks of the Deconvolution Object with Gaussians
        """

        x = self.spectrum.data[:,0]
        y = self.spectrum.data[:,1]

        # Create the Fits
        fits = []
        fit_labels = []

        for i, peak in enumerate(self.peaks):
            popt = self.fit_gaussian(peak, broadness)
            fit = self.gaussian(x, *popt)
            fits.append(fit)
            fit_labels.append(f"Fit {i+1}")

        # Now make an Altair chart and plot all the fits
        data = pd.DataFrame({"x": x, "y": y})

        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title="Wave Number / cm$^{-1}$", sort="descending").axis(format="0.0f"),
            y=alt.Y("y", title="Intensity"),
            color=alt.value("black")
        ).properties(
            title="Spectrum",
            width=800,
            height=400
        )

        # Create Selection
        selection = alt.selection_interval(bind="scales")

        # Add selection to chart
        chart = chart.add_selection(selection)

        # Create a color for each fit
        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "grey", "yellow", "cyan"]

        # Create a DataFrame for the fits
        fits_data = pd.DataFrame({"x": np.tile(x, len(fits)), "y": np.concatenate(fits), "fit": np.repeat(fit_labels, len(x))})

        # Create the chart for the fits
        chart_fits = alt.Chart(fits_data).mark_line().encode(
            x="x",
            y="y",
            color=alt.Color("fit:N", scale=alt.Scale(range=colors), legend=alt.Legend(title="Fits"))
        )

        # Combine the charts
        chart = chart + chart_fits

        # Add fits to object

        self.fits = fits

        return chart
    
    def fit_multiple_poly_gaussians_and_plot(self, broadness=10, ls_of_m=[1]):
        """
        General Function to Fit all the Peaks of the Deconvolution Object using Polynomially Modified Gaussians

        Attr:
            broadness: Broadness of the peaks
            list_of_m: List of model parameters for the skewness

        Note That the Model Parameters M can be used for assymetric peaks
        """

        x = self.spectrum.data[:,0]
        y = self.spectrum.data[:,1]

        # Create the Fits

        fits = []

        fit_labels = []

        for i, peak in enumerate(self.peaks):
            for m in ls_of_m:
                popt = self.fit_poly_gaussian(peak,broadness,m)
                fit = self.poly_gaussian(x,*popt)
                fits.append(fit)
                fit_labels.append(f"Fit {i+1} m={m}")

        # Now make an Altair chart and plot all the fits

        data = pd.DataFrame({"x": x, "y": y})

        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", title="Wave Number / cm$^{-1}$", sort="descending").axis(format="0.0f"),
            y=alt.Y("y", title="Intensity"),
            color=alt.value("black")
        ).properties(
            title="Spectrum",
            width=800,
            height=400
        )

        # Create Selection

        selection = alt.selection_interval(bind="scales")

        # Add selection to chart

        chart = chart.add_selection(selection)

        # Create a color for each fit

        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "grey", "yellow", "cyan"]

        # Create a DataFrame for the fits

        fits_data = pd.DataFrame({"x": np.tile(x, len(fits)), "y": np.concatenate(fits), "fit": np.repeat(fit_labels, len(x) )})

        # Create the chart for the fits

        chart_fits = alt.Chart(fits_data).mark_line().encode(
            x="x",
            y="y",
            color=alt.Color("fit:N", scale=alt.Scale(range=colors), legend=alt.Legend(title="Fits"))
        )

        # Combine the charts

        chart = chart + chart_fits

        # Add fits to object

        self.fits = fits

        return chart



    def calculate_integral_of_fits(self):
        """
        Calculates the Integral of the Fits
        """


        integrals = []
        annotations = []

        for i, fit in enumerate(self.fits):
            integral = np.abs(np.trapz(fit,self.spectrum.data[:,0]))
            integrals.append(integral)

            # Annotate the Peak

            peak_x = self.spectrum.data[self.peaks[i],0]
            peak_y = self.spectrum.data[self.peaks[i],1]

            # Also add the Name of the Spectrum

            annotation = {
                "x": str(peak_x) + " cm-1 "  + self.spectrum.name,
                "y": str(peak_y) + " cm-1 " + self.spectrum.name,
                "text": f"Integral: {integral:.5f}"
            }

            annotations.append(annotation)

        # Create the DataFrame for the annotations

        return integrals, annotations


    @staticmethod
    def deconvolution_function(mode="Gaussian", FWHM=1.0):
        """
        Makes a deconvolution function of a certain type.
        """
        if mode == "Gaussian":
            sigma = FWHM / (2 * np.sqrt(2 * np.log(2))) 
            def gaussian(x):
                b = 1/(sigma * np.sqrt(2*np.pi))*np.exp(-0.5*((x-np.mean(x))/sigma)**2)
                N = len(b)
                b_shifted = np.roll(b, N//2)
                return b_shifted
            
            return gaussian

    # This shit doesnt work
    @classmethod
    def mixed_gaussian_lorentzian(cls,x,beta, m=0.5, M=0.5):
        """
        Initializes a mixed gaussian lorentzian function for fitting the peaks
        
        Attributes:
            x: x data 
            beta: full width at half maximum
            m: mixing parameter m=0 pure lorentzian, m=1 pure gaussian
            M: sampling interval
        """

        M_samp = np.mean(x) + M

        # Initialize Gaussian
        gaussian_prefactor = (2*m*np.sqrt(np.log(2)))/(beta*np.sqrt(np.pi))
        gaussian_exp = np.exp(-4*np.log(2)*((x-M_samp)/(beta))**2)
        gaussian_part = gaussian_prefactor * gaussian_exp 

        # Initialíze Lorentian
        lorentzian_part = (2*(1-m)/(beta*np.pi*(1+4*((x-M_samp)/beta)**2)))

        return gaussian_part + lorentzian_part

    def fit_mixed_gaussian_lorentzian(self,peak,broadness=10,m=0.5,M=0.5):
        """
        Fits a mixed gaussian lorentzian model using using curve_fit and a predefined
        mixture
        """

        x = self.spectrum.data[:,0]
        y = self.spectrum.data[:,1]

        peak_interval = np.arange(peak-broadness,peak+broadness)

        x_peak = x[peak_interval]
        y_peak = y[peak_interval]


        # Fit all parameters
        popt, _ = curve_fit(self.mixed_gaussian_lorentzian,x_peak,y_peak,p0=[1,m,M]) 

        plt.plot(x,y, label="Spectrum", color="black")
        plt.plot(x,self.mixed_gaussian_lorentzian(x,*popt), label="Mixed Gaussian Lorentzian Fit", color="red")
        plt.gca().invert_xaxis()
        plt.legend()
        plt.show()
        return popt

