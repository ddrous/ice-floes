"""
This module defines classes and functions for the collision and displacement of ice floes along with their 2 nodes.
"""

import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from IPython.display import Image
from ipywidgets import interact

import imageio, io
import os
import PIL.Image as PILImage



