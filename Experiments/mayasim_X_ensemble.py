from __future__ import print_function

try:
    import cPickle as cP
except ImportError:
    import pickle as cP
import getpass
import itertools as it
import numpy as np
import sys
from subprocess import call

import pandas as pd

from pymofa.experiment_handling import experiment_handling as handle
from mayasim.model.ModelCore import ModelCore as Model
from mayasim.model.ModelParameters import ModelParameters as Parameters

global test


def run_func(r_bca=0.25, population_control=False, n=30,
             rf_crop_income_mode='sum', rf_steps=350, filename='./'):
    """
    Set up the Model for different Parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values and Parameters, as well as the time
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

    m = Model(n, output_data_location=filename, debug=test)
    m.population_control = population_control
    m.crop_income_mode = rf_crop_income_mode
    m.r_bca = r_bca
    m.output_level = 'trajectory'

    # store initial conditions and Parameters

    res = {}
    res["initials"] = pd.DataFrame({"Settlement X Possitions":
                                    m.settlement_positions[0],
                                    "Settlement Y Possitions":
                                    m.settlement_positions[1],
                                    "Population": m.population})

    res["Parameters"] = pd.Series({key:
                                   getattr(m, key)
                                   for key in dir(Parameters)
                                   if not key.startswith('__')
                                   and not callable(key)})

    # run Model
    if test:
        rf_steps = 5
    m.run(rf_steps)

    # retrieve results

    trajectory = m.trajectory
    headers = trajectory.pop(0)
    res["trajectory"] = pd.DataFrame(trajectory, columns=headers)

    with open(filename, 'wb') as dumpfile:
        cP.dump(res, dumpfile)
    try:
        tmp = np.load(filename)
    except IOError:
        print ("writing results failde for " + filename)
    return 0



# get subexperiment from comand line
if len(sys.argv) > 1:
    test = int(sys.argv[1])
else:
    test = 1

test_folder = ['', 'test_output/'][test]

if getpass.getuser() == "kolb":
    save_path_raw = "/p/tmp/kolb/Mayasim/output_data/{}X_ensemble".format(test_folder)
    save_path_res = "/home/kolb/Mayasim/output_data/{}X_ensemble".format(test_folder)
elif getpass.getuser() == "jakob":
    save_path_raw = "/home/jakob/Project_MayaSim/output_data/{}X_ensemble/".format(test_folder)
    save_path_res = save_path_raw

estimators = {"<mean_trajectories>":
                  lambda fnames: pd.concat([np.load(f)["trajectory"]
                                            for f in fnames]).groupby(level=0).mean(),
              "<sigma_trajectories>":
                  lambda fnames: pd.concat([np.load(f)["trajectory"]
                                            for f in fnames]).groupby(level=0).std()
              }


if test == 0:

    save_path_raw = save_path_raw + "raw_data/"
    save_path_res = save_path_res + "results/"

    N, crop_income_mode, steps = [30], ['sum'], [350]

    r_bcas = [0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    pcs = [True, False]

    parameter_combinations = list(it.product(r_bcas, pcs, N, crop_income_mode, steps))

    name = "mayasim_ensemble_testing"
    index = {0: "r_bca", 1: "population_control"}
    sample_size = 10

    h = handle(sample_size=sample_size,
               parameter_combinations=parameter_combinations,
               index=index,
               path_raw=save_path_raw,
               path_res=save_path_res)
    h.compute(run_func=run_func)
    h.resave(eva=estimators, name=name)
    call(["python", "visuals/mayasim_visuals.py", save_path_raw,
          save_path_res, repr(steps[0])])


if test == 1:

    save_path_raw = save_path_raw + "raw_data/"
    save_path_res = save_path_res + "results/"

    N, crop_income_mode, steps = [300], ['sum'], [10]

    r_bcas = [0.08, 0.14, 0.2]
    pcs = [True, False]

    parameter_combinations = list(it.product(r_bcas, pcs, [test]))

    name = "mayasim_ensemble_testing"
    index = {0: "r_bca", 1: "population_control"}
    sample_size = 2

    h = handle(sample_size=sample_size,
               parameter_combinations=parameter_combinations,
               index=index,
               path_raw=save_path_raw,
               path_res=save_path_res)
    h.compute(run_func=run_func)
    h.resave(eva=estimators, name=name)
