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


def run_func(r_bca=0.25, r_es=0.002, crop_income_mode='sum', n=30,
             rf_steps=350, filename='./'):
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
    r_es: float > 0
        pre factor for income form ecosystem services
    crop_income_mode : string
        defines the mode of crop income calculation.
        possible values are 'sum' and 'mean'
    n: int > 0
        initial number of settlements on the map
    steps: int
        number of steps to integrate the model for one run
    filename: string
        path to save the results to.
    """

    # initialize the Model

    m = Model(n=n, output_data_location=filename, debug=test)

    m.crop_income_mode = crop_income_mode

    if crop_income_mode == 'sum':
        m.r_bca_sum = r_bca
        m.r_es_sum = r_es
    elif crop_income_mode == 'mean':
        m.r_bca_mean = r_bca
        m.r_es_mean = r_es
    else:
        print('no valid income mode provided')

    # store initial conditions and Parameters

    res = {"initials": pd.DataFrame({"Settlement X Possitions":
                                         m.settlement_positions[0],
                                     "Settlement Y Possitions":
                                         m.settlement_positions[1],
                                     "Population":
                                         m.population}),
           "Parameters": pd.Series({key:
                                    getattr(m, key)
                                    for key in dir(Parameters)
                                    if not key.startswith('__')
                                    and not callable(key)})
           }

    # run Model
    if test:
        rf_steps = 5
    m.run(rf_steps)

    # retrieve results

    res["trajectory"] = m.get_trajectory()

    with open(filename, 'wb') as dumpfile:
        cP.dump(res, dumpfile)
    try:
        tmp = np.load(filename)
    except IOError:
        print ("writing results failde for " + filename)
    return 0


def run_experiment(argv):
    """
    Parse input from command line and initialize and start experiment accordingly.
    Also execute post processing and plotting of results.
    
    Parameters
    ----------
    argv: list
        input arguments from command line
    """

    global test
    if len(argv) > 1:
        test = int(argv[1])
    else:
        test = 1
    if len(argv) > 2:
        sub_experiment = int(argv[2])
    else:
        sub_experiment = 0

    test_folder = ['', 'test_output/'][test]
    exp_folder = ['income_mean', 'income_sum'][sub_experiment]

    if getpass.getuser() == "kolb":
        save_path_raw = "/p/tmp/kolb/Mayasim/output_data/{}X8_{}".format(test_folder, exp_folder)
        save_path_res = "/home/kolb/Mayasim/output_data/{}X8_{}".format(test_folder, exp_folder)
    elif getpass.getuser() == "jakob":
        save_path_raw = "/home/jakob/Project_MayaSim/output_data/{}raw/X8_{}/".format(test_folder, exp_folder)
        save_path_res = "/home/jakob/Project_MayaSim/output_data/{}X8_{}/".format(test_folder, exp_folder)

    estimators = {"<mean_trajectories>":
                      lambda fnames: pd.concat([np.load(f)["trajectory"]
                                                for f in fnames]).groupby(level=0).mean(),
                  "<sigma_trajectories>":
                      lambda fnames: pd.concat([np.load(f)["trajectory"]
                                                for f in fnames]).groupby(level=0).std()
                  }

    r_bcas_sum = [0.5, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35] if not test else [0.5, 0.25]
    r_es_sum = [0.0001, 0.00015, 0.0002, 0.00025] if not test else [0.0001, 0.00025]

    r_bcas_mean = [.8, .9, 1., 1.1, 1.2, 1.3, 1.4, 1.5] if not test else [0.8, 1.3]
    r_es_mean = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20] if not test else [6, 12]

    if sub_experiment == 0:
        income_mode = 'mean'
        parameter_combinations = list(it.product(r_bcas_mean, r_es_mean, [income_mode]))
    elif sub_experiment == 1:
        income_mode = 'sum'
        parameter_combinations = list(it.product(r_bcas_sum, r_es_sum, [income_mode]))

    name = "mayasim_income_parameter_scan"
    index = {0: "r_bca", 1: 'r_es', 2: 'income_mode'}
    sample_size = 10 if not test else 2

    hdl = handle(sample_size=sample_size,
               parameter_combinations=parameter_combinations,
               index=index,
               path_raw=save_path_raw,
               path_res=save_path_res)
    hdl.compute(run_func=run_func)
    hdl.resave(eva=estimators, name=name)

    """TO DO: 1) save raw run for each parameter combination
              2) plot data"""

