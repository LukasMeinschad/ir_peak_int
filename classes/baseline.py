from pybaselines import Baseline as Baseline_fit
import numpy as np
import altair as alt

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

