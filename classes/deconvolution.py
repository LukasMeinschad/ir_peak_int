import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Deconvolution:
    """
    General class where different deconvolution methods are implemented.
    """
    def __init__(self,spectrum):
        """ 
        Initializes a deconvolution object by using a spectrun object

        Parameters
        ----------
        spectrum : Spectrum
            A spectrum object containing the data to be deconvoluted.
        """
        self.spectrum = spectrum
        ss