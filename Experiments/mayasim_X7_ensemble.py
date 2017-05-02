from __future__ import print_function
try:
    import cPickle as cp
except ImportError:
    import pickle as cp
import getpass
import itertools as it
import numpy as np
import sys
import pandas as pd

from pymofa.experiment_handling import experiment_handling as eh
from mayasim.model.ModelCore import ModelCore as Model


def RUN_FUNC(r_bca=0.2, r_eco=0.0002, population_control=False,
             N=30, crop_income_mode='sum',
             kill_cropless=True, steps=350, filename='./'):
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

    m = Model(N)
    m.population_control = population_control
    m.crop_income_mode = crop_income_mode
    m.r_bca_sum = r_bca
    m.r_es_sum = r_eco
    m.kill_cities_without_crops = kill_cropless
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
        m.run(3)
    else:
        m.run(steps)

    # retrieve results

    res["trajectory"] = m.get_trajectory()

    with open(filename, 'wb') as dumpfile:
        cp.dump(res, dumpfile)
    try:
        tmp = np.load(filename)
    except IOError:
        print ("writing results failde for " + filename)
    return 0


EVA = {"<mean_trajectories>":
       lambda fnames: pd.concat([np.load(f)["trajectory"]
                                 for f in fnames]).groupby(level=0).mean(),
       "<sigma_trajectories>":
       lambda fnames: pd.concat([np.load(f)["trajectory"]
                                 for f in fnames]).groupby(level=0).std()
       }


# get subexperiment from comand line
if len(sys.argv) > 1:
    sub_experiment = int(sys.argv[1])
else:
    sub_experiment = 0

if getpass.getuser() == "kolb":
    SAVE_PATH_RAW = "/p/tmp/kolb/Mayasim/output_data/X_ensemble"
    SAVE_PATH_RES = "/home/kolb/Mayasim/output_data/X_ensemble"
elif getpass.getuser() == "jakob":
    SAVE_PATH_RAW = \
        "/home/jakob/PhD/Project_MayaSim/Python/output_data/X_ensemble/"
    SAVE_PATH_RES = SAVE_PATH_RAW

SAVE_PATH_RAW = SAVE_PATH_RAW + "raw_data/"
SAVE_PATH_RES = SAVE_PATH_RES + "results/"

INDEX = {0: "r_bca", 1: "r_eco", 2: "kill_cropless"}
if sub_experiment == 0:
    r_bcas = [0.1, 0.15, 0.2, 0.25, 0.3]
    r_ecos = [0.0001, 0.00015, 0.0002, 0.00025]
    kill_cropless = [True, False]
    test=False
if sub_experiment == 1:
    r_bcas = [0.1, 0.15]
    r_ecos = [0.0001, 0.00015]
    kill_cropless = [True, False]
    test=True

PARAM_COMBS = list(it.product(r_bcas, r_ecos, kill_cropless))

NAME = "mayasim_ensemble_testing"

SAMPLE_SIZE = 10 if not test else 2

handle = eh(sample_size=SAMPLE_SIZE,
            parameter_combinations=PARAM_COMBS,
            index=INDEX,
            path_raw=SAVE_PATH_RAW,
            path_res=SAVE_PATH_RES,
            use_kwargs=True)

handle.compute(run_func=RUN_FUNC)
handle.resave(eva=EVA, name=NAME)

