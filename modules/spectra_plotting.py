"""
This module contains a variety of functions for plotting IR spectras
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_spectra(x,y,title):
    """
    Plots the whole FTIR Spectrum given x, and y data
    """

    plt.plot(x,y, label="FTIR Spectrum", color = "black", linewidth = 0.6)
    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber (cm${-1}$)")
    plt.ylabel("Absorbance / --")
    plt.title(title)
    plt.show()


def overlay_spectra(data_np, title="Overlayed Spectra"):
    
    """
    Overlays the Spectra of a given dataset
    """
    fig, ax = plt.subplots()
    for i in range(1,data_np.shape[1]):
        ax.plot(data_np[:,0],data_np[:,i],label="Measurement "+str(i))
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Wavelength (cm-1)")
    ax.set_ylabel("Absorbance")      
    plt.show()




def plot_spectral_window(x,y,spectral_window,title):
    """ 
    Plots a given spectral window for the IR data

    Attr:
        x: x-data
        y: y-data
        spectral_window: a tuple with the spectral window
        title: a string with the title
    
    Returns:
        A Plot of the Spectral Window with the selected data
    """
    # Select the data using the spectral window
    mask = np.logical_and(x >= spectral_window[1], x <= spectral_window[0])
    x_window = x[mask]
    y_window = y[mask]

    plt.plot(x_window, y_window, label="Spectral Window", color ="black",linewidth = 0.6)
    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber / (cm${-1}$)")
    plt.ylabel("Absorbance")
    plt.title(title)
    plt.show()
    return x_window, y_window


def peak_picking_spectral_window(x_window,y_window,height=None,distance=None,prominence=None):
    """
    Applies Peak Picking to a given spectral window

    Attr:
        x_window: x-data of spectral window
        y_window: y-data of spectral window
        height: Required height of Peaks, optional
        distance: Required horizontal distance between peaks
        prominence: Required prominence of peaks
    """

    peaks, properties = find_peaks(y_window, height=height, distance=distance, prominence=prominence)

    plt.figure(figsize=(10,6))
    plt.plot(x_window,y_window, label="Spectal Window", color = "black",linewidth = 0.6)
    plt.plot(x_window[peaks],y_window[peaks], "x", label="Peaks", color ="Red")
    plt.gca().invert_xaxis()
    plt.title("Peak Picking Spectral Window")
    plt.xlabel("Wavenumber / (cm${-1}$)")
    plt.ylabel("Absorbance")
    plt.legend()

    for peak in peaks:
        plt.annotate(f"{x_window[peak]:.0f}", (x_window[peak],y_window[peak]),textcoords="offset points", xytext=(0,5), ha="center")
    plt.show()

    return peaks,properties