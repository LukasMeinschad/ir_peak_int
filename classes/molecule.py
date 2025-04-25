import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
from ipywidgets import widgets, interact, fixed
from IPython.display import display 


# I know I know ...
def section(fle, begin, end):
    """ 
    Yields lines between begin and end markers in a file.

    Args:
        fle (str): The file path.
        begin (str): The beginning marker.
        end (str): The ending marker.
    """
    with open(fle) as f:
        for line in f:
            if begin in line:  
                for line in f:
                    # Found end so top
                    if end in line: 
                        return
                    # Yield the line
                    yield line.rstrip()

# 
def section_vibration(fle, begin, end, vibration):
    """ 
    Yields the lines between begin and end markes with an additional vibration index
    """
    with open(fle) as f:
        for line in f:
            if begin in line and str(vibration) in line:  
                for line in f:
                    # Found end so top
                    if end in line: 
                        return
                    # Yield the line
                    yield line.rstrip()



def parse_molden(file_path):
    """
    Extracts frequencies, coordinates, and normal modes from Molden File
    
    Args:
        filename: Path to Molden file
    
    Returns:
        Tuple of (frequencies, coordinates, normal_modes)
    """    
    # Get all lines between [FREQ] and [FR-COORDS]
    all_frequencies = list(section(file_path, "[FREQ]", "[FR-COORD]"))
    # store all frequencies with indices
    all_frequencies = [(float(freq),i) for i,freq in enumerate(all_frequencies)]

    # Get all coordinates between [FR-COORD] and [FR-NORM-COORD]
    coords = list(section(file_path, "[FR-COORD]", "[FR-NORM-COORD]"))

    # Get normal modes for each vibration

    normal_modes = []
    for freq in range(len(all_frequencies)):
        if freq+1 != len(all_frequencies): 
            # Get modes between vibration X and vibration X+1
            normal_modes.append(list(section_vibration(file_path, "Vibration", "Vibration", freq+1))) 

        else:
            # For last mode get all lines until the end of the file
            # Dont ask me pls
            normal_modes.append(list(section_vibration(file_path, "Vibration", "                   ", freq+1)))
  
    return all_frequencies, coords, normal_modes




def draw_normal_mode(mode=0, coords=None, normal_modes=None):
    """
    Visualizes a normal mode with an animation
    """

    # converion factor bohr / angstrom
    bohr_to_angstrom = 0.529177249

    # Create XYZ format string with coordinates and displacements
    xyz = f"{len(coords)}\n\n"
    for i in range(len(coords)):
        # Get the Base coordinates
        # Never ask me about this
        atom_coords = [float(m) for m in coords[i][8:].split('       ')]
        
        mode_coords = [float(m) for m in normal_modes[mode][i][8:].split('       ')]

        # Format line: atom baseX,baseY,baseZ, dispX,dispY,dispZ
        xyz += (f"{coords[i][0:4]}"
                f" {atom_coords[0]*bohr_to_angstrom} "
                f"{atom_coords[1]*bohr_to_angstrom} "
                f"{atom_coords[2]*bohr_to_angstrom} "
                f"{mode_coords[0]*bohr_to_angstrom} "
                f"{mode_coords[1]*bohr_to_angstrom} "
                f"{mode_coords[2]*bohr_to_angstrom}\n"
                )

    # set up 3D view
    view = py3Dmol.view(width=800, height=400)
    # Add model with vibration an parameters
    view.addModel(xyz, "xyz", {"vibrate": {"frames":20, "amplitude":1.5}})
    # Style the atoms
    view.setStyle({"sphere": {"scale": 0.3}, "stick": {"radius": 0.2}})

    # Set background color
    view.setBackgroundColor("white")
    view.animate({"loop": "backAndForth"})
    view.show()

def show_normal_modes(filename):
    """
    Creates an Interactive widget to explore the normal modes
    """
    # Parse Molden File
    all_frequencies, coords, normal_modes = parse_molden(filename)
    
    # Create Interative widget
    interact(draw_normal_mode,
             coords=fixed(coords),
             normal_modes=fixed(normal_modes),
             mode = widgets.Dropdown(
                 options = all_frequencies,
                 value = 0,
                 description = "Normal Mode",
                 style = {"description_width": "initial"},
             ))


class Atom:
    """
    Class representing an atom in a molecule
    """
    def __init__(self, symbol, coords):
        """
        Initializes an atom object

        Args:
            symbol (str): The symbol of the atom (e.g., 'H', 'O', 'C').
            coords (tuple): The coordinates of the atom in 3D space.
        """ 
        self.symbol = symbol
        self.coords = coords


class Molecule:
    def __init__(self,name,atoms):
        """
        Initializes a molecule object
        """
        self.name = name
        self.atoms = atoms
        self.mol2D = None
        self.mol3D = None
        self.frequencies = None
        self.normal_modes = None
   

    @classmethod
    def molden_to_molecule(cls,filepath, name="MyMolecule"):
        """ 
        Converts a Molden file into a molecule object
        """     

        all_frequencies, coords, normal_modes = parse_molden(filepath)

        atoms = []
        
        for line in coords:
            

            line = line.strip().split()
            atom_symbol = line[0]
            x,y,z = float(line[1]), float(line[2]), float(line[3])
            atom = Atom(atom_symbol, (x,y,z))
            atoms.append(atom)
        
        normal_modes_dict = {}
        for i, normal_mode in enumerate(normal_modes):
            normal_modes_dict[i] = []
            for line in normal_mode:
                line = line.strip().split()
                x,y,z = float(line[0]), float(line[1]), float(line[2])
                
                normal_modes_dict[i].append((x,y,z))
        
        # make molecule object
        Molecule = cls(name=name,atoms=atoms)
        Molecule.frequencies = all_frequencies
        Molecule.normal_modes = normal_modes_dict

        return Molecule

    def draw_normal_mode(self, mode=0):
        """
        Visualizes a normal mode with an animation
        """
        # converion factor bohr / angstrom
        bohr_to_angstrom = 0.529177249

        # Create XYZ format string with coordinates and displacements
        xyz = f"{len(self.atoms)}\n\n"
        for i in range(len(self.atoms)):
            # Get the base coordinateds

            atom_coords = self.atoms[i].coords
            mode_coords = self.normal_modes[mode][i]
            # Format line: atom baseX,baseY,baseZ, dispX,dispY,dispZ
            xyz += (f"{self.atoms[i].symbol} "
                    f"{atom_coords[0]*bohr_to_angstrom} "
                    f"{atom_coords[1]*bohr_to_angstrom} "
                    f"{atom_coords[2]*bohr_to_angstrom} "
                    f"{mode_coords[0]*bohr_to_angstrom} "
                    f"{mode_coords[1]*bohr_to_angstrom} "
                    f"{mode_coords[2]*bohr_to_angstrom}\n"
                    )
        # set up 3D view
        view = py3Dmol.view(width=800, height=400)
        # Add model with vibration an parameters
        view.addModel(xyz, "xyz", {"vibrate": {"frames":20, "amplitude":1.5}})
        # Style the atoms
        view.setStyle({"sphere": {"scale": 0.3}, "stick": {"radius": 0.2}})
        # Set background color
        view.setBackgroundColor("white")
        view.animate({"loop": "backAndForth"})
        view.show()

    def visualize_normal_modes(self):
        """
        Visualizes the normal modes of the molecule
        """

        # Error if no normal modes
        if self.frequencies is None:
            raise ValueError("No normal modes found for the molecule.")
        

        # Create Interative widget
        interact(self.draw_normal_mode,
                 mode = widgets.Dropdown(
                     options = self.frequencies,
                     value = 0,
                     description = "Normal Mode",
                     style = {"description_width": "initial"},
                 ))
        




    @classmethod
    # IMPLEMENT THIS
    def molpro_to_molecule(cls,file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

            extracting = False
            start = "Atomic Coordinates"
            end = "Gradient"

            extracted_data = []
            for line in lines:
                # turn on extraction switch
                if start in line:
                    extracting = True
                # turn off extraction switch
                if end in line:
                    extracting = False
                    break
                # If extraction switch on --> extract
                if extracting:
                    extracted_data.append(line)
            



    @classmethod
    def mol3d_to_molecule(cls,file_path):
        """
        Converts a 3Dmol file into a molecule object
        """
    
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            extracting = False
            
            start = "V20"

            end = "END"

            extracted_data = []

            for line in lines:
                # turn on extraction switch
                if start in line:
                    extracting = True
                # turn off extraction switch
                if end in line:
                    extracting = False
                    break
                # If extraction switch on --> extract
                if extracting:
                    extracted_data.append(line)
            
            num_atoms = int(extracted_data[0].split()[0])
            num_bonds = int(extracted_data[0].split()[1])
            extracted_data = extracted_data[1:]

            # Now Extract Atom Block atom_start = 0 -> num_atoms
            atoms = []
            coords = []
            for line in extracted_data[:num_atoms]:
                values = line.strip().split()
                x = float(values[0])
                y = float(values[1])
                z = float(values[2])
                atom_symbol = values[3]
                atoms.append(atom_symbol)
                coords.append([x, y, z])
            
            # Return coordinates atoms + mol3d
            Molecule = cls(name="3Dmol",coords=coords,atoms=atoms)
            
            with open(file_path, 'r') as f:
                file_contend = ""
                line = f.readline()
                while line:
                    file_contend += line
                    line = f.readline()
            Molecule.mol3D = file_contend
            Molecule.mol2D = None
            return Molecule

    def visualize_molecule_mol3D(self):
        """
        Visualizes the molecule using py3Dmol
        """
        view = py3Dmol.view(width=800, height=400)

        if self.mol3D is None:
            raise ValueError("No 3D structure found for the molecule.")
        else:
            mol3d = self.mol3D
            mol = Chem.MolFromMolBlock(mol3d)
            view.addModel(mol3d, 'mol')
            view.setStyle({'stick': {}})
            view.setBackgroundColor('white')
            view.zoomTo()
            view.show()
        

           
                    

        



    @classmethod
    def mol3d_to_molecule_nist_compound(cls,compound):
        """ 
        Converts a NIST compound object to a molecule objec
        """

        name = compound.name
        Molecule = cls(name=name,coords=None,atoms=None)

        if compound.mol3D is None:
            raise ValueError("No 3D structure found for the compound.")
        else:
            mol3d = compound.mol3D
            Molecule.mol3D = mol3d
        if compound.mol2D is None:
            raise ValueError("No 2D structure found for the compound.")
        else:
            mol2d = compound.mol2D
            Molecule.mol2D = mol2d

        return Molecule

