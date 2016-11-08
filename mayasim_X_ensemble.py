from __future__ import print_function

import cPickle as cp
import getpass
import itertools as it
import numpy as np
import sys
from subprocess import call

import pandas as pd

from mayasim_model.model import model
from mayasim_model.model_parameters import parameters
from pymofa import experiment_handling as eh


def RUN_FUNC(r_bca, population_control, N, crop_income_mode, steps, filename):
    """
    Set up the Model for different parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values and parameters, as well as the time
    development of aggregated variables for each run.

    Parameters:
    -----------
    r_bca : float > 0
        the pre factor for income from agriculture
    population_control : boolean
        determines whether the population grows
        unbounded or if population growth decreases
        with income per capita and population density.
    N : int > 0
        initial number of settlements on the map
    crop_income_mode : string
        defines the mode of crop income calculation.
        possible values are 'sum' and 'mean'
    filename: string
        path to save the results to.
    """

    # initialize the Model

    m = model(N)
    m.population_control = True
    m.crop_income_mode = crop_income_mode
    m.r_bca = r_bca
    m.output_level = 'trajectory'

    # store initial conditions and parameters

    res = {}
    res["initials"] = pd.DataFrame({"Settlement X Possitions":
                                    m.settlement_positions[0],
                                    "Settlement Y Possitions":
                                    m.settlement_positions[1],
                                    "Population": m.population})

    res["parameters"] = pd.Series({key:
                                   getattr(m, key)
                                   for key in dir(parameters)
                                   if not key.startswith('__')
                                   and not callable(key)})

    # run Model

    m.run(steps, filename)

    # retrieve results

    trajectory = m.trajectory
    headers = trajectory.pop(0)
    res["trajectory"] = pd.DataFrame(trajectory, columns=headers)

    with open(filename, 'wb') as dumpfile:
        cp.dump(res, dumpfile)
    try:
        tmp = np.load(filename)
    except IOError:
        print ("writing results failde for " + filename)
    return 0


def compute(SAVE_PATH_RAW):

    eh.compute(RUN_FUNC, PARAM_COMBS, SAMPLE_SIZE, SAVE_PATH_RAW)


def resave(SAVE_PATH_RAW, SAVE_PATH_RES, sample_size=None):
    """
    dictionary of lambda functions to calculate the
    average and errors for the trajectories from all
    runs given in the list of file names (fnames)
    that is handled internally by resave_data.

    Parameters:
    -----------
    SAVE_PATH_RAW : string
        path to the raw output from runs
    SAVE_PATH_RES : string
        path to save the results to
    sample_size : int
        the number of runs computed for one
        combination of parameer e.g. the size
        of the ensemble for statistical analysis
    """
    EVA = {"<mean_trajectories>":
           lambda fnames: pd.concat([np.load(f)["trajectory"]
                                     for f in fnames]).groupby(level=0).mean(),
           "<sigma_trajectories>":
           lambda fnames: pd.concat([np.load(f)["trajectory"]
                                     for f in fnames]).groupby(level=0).std()
           }

    eh.resave_data(SAVE_PATH_RAW,
                   PARAM_COMBS,
                   INDEX,
                   EVA,
                   NAME,
                   sample_size,
                   save_path=SAVE_PATH_RES)


# get subexperiment from comand line
if len(sys.argv) > 1:
    sub_experiment = int(sys.argv[1])
else:
    sub_experiment = 0

if getpass.getuser() == "kolb":
    SAVE_PATH_RAW = "/p/tmp/kolb/Mayasim/output_data/X_ensemble"
    SAVE_PATH_RES = "/home/kolb/Mayasim/output_data/X_ensemble"
elif getpass.getuser() == "jakob":
    SAVE_PATH = "/home/jakob/PhD/Project_MayaSim/Python/output_data/X_ensemble/"

if sub_experiment == 0:

    SAVE_PATH_RAW = SAVE_PATH_RAW + "raw_data/"
    SAVE_PATH_RES = SAVE_PATH_RES + "results/"

    N, crop_income_mode, steps = [30], ['sum'], [350]

    r_bcas = [0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    pcs = [True, False]

    PARAM_COMBS = list(it.product(r_bcas, pcs, N, crop_income_mode, steps))

    NAME = "mayasim_ensemble_testing"
    INDEX = {0: "r_bca", 1: "population_control"}
    SAMPLE_SIZE = 10

    compute(SAVE_PATH_RAW)
    resave(SAVE_PATH_RAW, SAVE_PATH_RES, SAMPLE_SIZE)
    call(["python", "visuals/mayasim_visuals.py", SAVE_PATH_RAW,
          SAVE_PATH_RES, `t_max`])


if sub_experiment == 1:

    SAVE_PATH_RAW = SAVE_PATH_RAW + "raw_data/"
    SAVE_PATH_RES = SAVE_PATH_RES + "results/"

    N, crop_income_mode, steps = [30], ['sum'], [3]

    r_bcas = [0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    pcs = [True, False]

    PARAM_COMBS = list(it.product(r_bcas, pcs, N, crop_income_mode, steps))

    NAME = "mayasim_ensemble_testing"
    INDEX = {0: "r_bca", 1: "population_control"}
    SAMPLE_SIZE = 3

    resave(SAVE_PATH_RAW, SAVE_PATH_RES, SAMPLE_SIZE)
    call(["python", "visuals/mayasim_visuals.py", SAVE_PATH_RAW,
          SAVE_PATH_RES, `t_max`])
