
class Annotation:
    def __init__(self, name: str, annotation_scheme="chemist", x_position = 0, y_position=0, description = None):
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

        self.description = str(mode) + " " + str(self.chemist_notation[type]) + str(group) +  " " + str(x_position)


    
