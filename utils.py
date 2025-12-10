from matplotlib import pyplot as plt
from ipywidgets import interact
import numpy as np
import sys


def log(msg: str):
    print(msg, flush=True)


def log_error(msg: str):
    print(msg, file=sys.stderr, flush=True)


def explore_3D_array(arr: np.ndarray, cmap: str = 'gray'):
    """
    Given a 3D array with shape (Z,X,Y) This function will create an interactive
    widget to check out all the 2D arrays with shape (X,Y) inside the 3D array.
    The purpose of this function to visual inspect the 2D arrays in the image.

    Args:
        arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
        cmap : Which color map use to plot the slices in matplotlib.pyplot
    """
    def fn(SLICE):
        plt.close('all')  # Close previous figures
        plt.figure(figsize=(7,7))
        plt.imshow(arr[SLICE, :, :], cmap=cmap)
        plt.title(f'Slice {SLICE}/{arr.shape[0]-1}')

    interact(fn, SLICE=(0, arr.shape[0]-1))
