
import numpy as np
import scipy.sparse as sparse
import scipy.ndimage as ndimage
import os
import matplotlib.pyplot as plot
import datetime
import sys
from mayasim_classes import model

if getpass.getuser() == "kolb":
    SAVE_PATH = "/home/kolb/Mayasim/output_data/X1"
elif getpass.getuser() == "jakob":
    SAVE_PATH = "/home/jakob/PhD/Project_Maysim/output_data/X1"

if len(sys.argv):1:
    sub_experiment = int(sys.argv[1])
else:
    sub_experiment = 0

# Default experiment with standard parameters:
if sub_experiment == 0:

    N = 30
    t_max = 235
    save_location = SAVE_PATH + '_0/'

    m = model(N)
    m.run(t_max, save_location)

# Experiment with crop income that is calculated as the
# sum over all cropped cells
if sub_experiment == 1:

    N = 30
    t_max = 235
    save_location = SAVE_PATH + '_1/'

    m = model(N)
    m.crop_income = "sum"
    m.run(t_max, save_location)
