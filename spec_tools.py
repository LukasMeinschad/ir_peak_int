# import packages
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nistchempy as nist



# Module Import
from classes import spectrum
from classes import molecule




def nist_compound_search(identifier, search_type="name"):
    """
    Searches for a compound in the NIST database using the provided identfier.

    Paramters
    ----------

    identifier : str
        Identifier of the compond
    search_type : str
        Type of search to perform options are "name", "inchi", "cas", "formula"
    """

    s = nist.run_search(identifier,search_type)

    return s

def nist_get_compound(compound_id):
    """
    Retrieves a compound from the NIST database using the provided compound ID.
    
    Parameters:
    ----------
    compound_id : str
        The ID of the compound to retrieve.
    """
    compound = nist.get_compound(compound_id)
    return compound

def nist_get_spectra(compound):
    """
    Retrieves all spectra for a given compound from the NIST database

    Parameters:
    ----------
    compound : NistCompound
        The compound object for which to retrieve spectra.
    """

    compound.get_all_spectra()

    # Return spectrum
    return compound

def parse_jdx_compound(compound):
    """
    Parses the JDX file of a given compound and returns a dataframe
    
    Parameters:
    ----------
    compound
    ----------
    """
    for line in compound.jdx_text:
        filter_start = False
        filter_end = False

        # turn on extraction switch
        if line.startswith("##XYDATA"):
            filter_start = True
        # turn off extraction switch
        if line.startswith("##END"):
            filter_end = True
            break
        # if extraction switch is on, extract data
        if filter_start and not filter_end:
            extracted_data.append(line)
    # remove first line
    extracted_data = extracted_data[1:]
    # remove last line
    extracted_data = extracted_data[:-1]

    # split each line into a list of values
    extracted_data = [line.split() for line in extracted_data]

    # initialize numpy array
    data_raw = np.array(extracted_data, dtype=float)

    return data_raw

def parse_jdx_file(file_path):
    """ 
    parses a JDX file and returns a np.array of the data
    """

    # Open the file
    with open(file_path, "r") as file:
        # Read in all lines
        lines = file.readlines()

        # XYDATA = (X++(Y..))
        # Each line starts with an 𝑥
        # value, and is followed by as many 𝑦
        # values as can fit within the 80 character per line limit. Subsequent 𝑥
        # values are incremented according to the 𝑥
        # resolution and the number of 𝑦
        # values that fit on the previous line (which in turn depends upon the compression scheme).

        # filter start
        filter_start = False
        # filter end
        filter_end = False
        
        extracted_data = []
        for line in lines:
            # turn on extraction switch
            if line.startswith("##XYDATA"):
                filter_start = True
            # turn off extraction switch
            if line.startswith("##END"):
                filter_end = True
                break
            # if extraction switch is on, extract data
            if filter_start and not filter_end:
                extracted_data.append(line)

        # remove first line
        extracted_data = extracted_data[1:]
        # remove last line
        extracted_data = extracted_data[:-1]

        # split each line into a list of values
        extracted_data = [line.split() for line in extracted_data]

        # initialize numpy array

        data_raw = np.array(extracted_data, dtype=float)

        return data_raw


def get_mol_file(compound):
    """
    Retrieves the mol file from the NIST database for a given compound"""

    compound.get_molfiles()
    # Return mol file
    return compound
