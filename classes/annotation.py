


class Annotation:
    def __init__(self, name: str, annotation_scheme="chemist", x_position = 0, y_position=0, description = None, color = None):
        """
        Initializes an annotation object with the given name and position.

        Parameters
        ----------
        name : str
            The name of the annotation.
        annotation_scheme : str
            Selects the given annotation scheme. Options are "chemist", "physicist", "spectroscopist"
        x_position : int
            the x position of the annotation.
        y_position : int
            the y position of the annotation.
        description : str
            Description of the annotation.
        """
        self.name = name
        self.annotation_scheme = annotation_scheme
        self.x_position = x_position
        self.y_position = y_position
        self.description = description
        self.color = color

    def __repr__(self):
        return f"Annotation(name={self.name}, annotation_scheme={self.annotation_scheme}, x_position={self.x_position}, y_position={self.y_position}, description={self.description})"
    def __str__(self):
        return f"Annotation: {self.name} \n \
                Annotation Scheme: {self.annotation_scheme} \n \
                X Position: {self.x_position} \n \
                Y Position: {self.y_position} \n \
                Description: {self.description}"


    chemist_notation = {
           "stretching": "\u03bd",
           "bending": "\u03b4",
           "rocking": "\u03c1",
           "wagging": "\u03a9",
           "twisting": "\u03c4" 
        }
    
    def add_chemist_notation(self, mode, type, group = None, x_position=0, symmetry_info = None):
        """
        Adds a chemist notation
        """


        self.x_position = x_position
        self.y_position = 0

        if symmetry_info is None:
            self.description = str(mode) + " " + str(self.chemist_notation[type]) + str(group) +  " " + str(x_position)
        else:
            # maybe add convert_to_subscript_superscript here sometimes
            self.description = str(mode) + " " + str(self.chemist_notation[type]) + self.convert_to_subscript_superscript(symmetry_info,is_superscript=False) + " " + str(group) +  " " + str(x_position)

    @staticmethod
    def convert_to_subscript_superscript(text, is_superscript=True):
        superscript_mapping = {
                        "as": "ᵃˢ",
                        "ip": "ⁱᵖ",
                        "s": "ˢ"
        }
        subscript_mapping = {
                        "as": "ₐₛ",
                        "ip": "ᵢₚ",
                        "s": "ₛ"
        }

        mapping = superscript_mapping if is_superscript else subscript_mapping
        return mapping.get(text, text)  # Fallback: return original text if no match is found


    @classmethod
    def import_harmonic_frequencies_molpro(cls,filepath):
        """
        Imports the harmonic frequencies from a molpro output file
        
        Parameters
        ----------
        filepath : str
            The path to the molpro output file.
        Returns
        -------
        annotations_low : list
            A list of Annotation objects for low vibrations.
        annotations_vibrations : list
            A list of Annotation objects for vibrations.
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

            # Make a switch for filtering

            extracting = False
            extracted_lines = []
            for line in lines:
                if "Low Vibration" in line:
                    extracting = True
                if extracting:
                    if "FREQUENCIES *" in line:
                        extracting = False
                    else:
                        extracted_lines.append(line)
            
            # We have Low Vibration Modes and Vibration Modes

            # split the lines
            extracted_lines = [line.split() for line in extracted_lines]
            
            # find indeces with Vibration

            vibration_indices = []
            for i, line in enumerate(extracted_lines):
                if "Vibration" in line:
                    vibration_indices.append(i)
            
            # Low Vibrations = first index + 1 : second index
            # Vibration = second index + 1 : end
            low_vibrations = extracted_lines[vibration_indices[0] + 1 : vibration_indices[1]]
            vibrations = extracted_lines[vibration_indices[1] + 1 : ] 
            
            # Remove first line with units
            low_vibrations = low_vibrations[1:]
            vibrations = vibrations[1:]
            # Remove empty lists
            low_vibrations = [line for line in low_vibrations if line]
            vibrations = [line for line in vibrations if line]

            # Now first value is mode, second value is frequency

            annotations_low = []
            for line in low_vibrations:
                mode = line[0]
                x_position = line[1]
                annotations_low.append(cls(mode, x_position=x_position))

            annotations_vibrations = []
            for line in vibrations:
                mode = line[0]
                x_position = float(line[1])
                color = "red" # TODO: maybe add color scheme here
                # Add a description
                description = "Harm. Freq" + "\n" + str(line[1]) + " cm-1"
    
                annotations_vibrations.append(cls(mode, x_position=x_position, description=description, color=color))
            
            return annotations_low, annotations_vibrations
            



    
