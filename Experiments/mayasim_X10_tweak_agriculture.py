"""
Experiment to tweak income from agriculture and ecosystem services
such that the results are comparable to Hecktert 2013.

I start with the low trade income (r_trade=6000) configuration in X8
that is closes to the desired results but has higher es income and lower ag income.
Then, I

1) increase the value of agricultural income,
2) decrease the value of ecosystem services income

and see at which values the trajectories are similar to Fig. 5 Heckberts2013.


"""

from __future__ import print_function
import sys
import os
try:
    import cPickle as cp
except ImportError:
    import pickle as cp
import getpass
import itertools as it
import numpy as np
import pandas as pd

from pymofa.experiment_handling import experiment_handling as eh
from mayasim.model.ModelCore import ModelCore as Model
from mayasim.model.ModelParameters import ModelParameters as Parameters
from mayasim.visuals.custom_visuals import MapPlot

test = True
steps = 3000

def run_function(r_bca=0.2, r_es=0.15, r_trade=6000,
                 es_income_mode='sum',
                 ag_income_mode='sum',
                 better_ess=True,
                 kill_cropless=False,
                 filename='./'):
    """
    Set up the Model for different Parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values and Parameters, as well as the time
    development of aggregated variables for each run.

    Parameters:
    -----------
    r_bca : float > 0
        the prefactor for income from agriculture
    r_es : float
        the prefactor for income from ecosystem services
    r_trade : float
        the prefactor for income from trade
    crop_income_mode : string
        defines the mode of crop income calculation.
        possible values are 'sum' and 'mean'
    ag_income_mode : string
        defines the mode of agriculture income calculation.
        possible values are 'sum' and 'mean'
    better_ess : bool
        switch to use forest as proxy for income from eco
        system services from net primary productivity.
    kill_cropless: bool
        Switch to determine whether or not to kill cities
        without cropped cells.
    filename: string
        path to save the results to.
    """

    # initialize the Model
    if test:
        print(r_bca, r_es, r_trade, ag_income_mode, es_income_mode, better_ess, kill_cropless)

    m = Model(30, output_data_location=filename, debug=test)
    if not filename.endswith('s0.pkl'):
        m.output_geographic_data = False
        m.output_settlement_data = False

    # crop income sum/mean
    m.crop_income_mode = ag_income_mode
    # eco income sum/mean
    m.eco_income_mode = es_income_mode
    # agriculture and forest income combined True/False
    m.better_ess = better_ess
    # remove settlements without agriculture: True/False
    m.kill_cities_without_crops = kill_cropless

    # prefactors for different sources of 
    # income according to different income modes.

    # parameter for ag sum
    m.r_bca_sum = r_bca
    # parameter for ag mean
    m.r_bca_mean = r_bca

    # parameter for eco sum
    m.r_es_sum = r_es
    # parameter for eco mean
    m.r_es_mean = r_es

    # parameter for trade
    m.r_trade = r_trade

    m.precipitation_modulation = False

    # store initial conditions and Parameters

    res = {"initials": pd.DataFrame({"Settlement X Possitions":
                                     m.settlement_positions[0],
                                     "Settlement Y Possitions":
                                     m.settlement_positions[1],
                                     "Population": m.population}),
           "Parameters": pd.Series({key: getattr(m, key)
                                    for key in dir(Parameters)
                                    if not key.startswith('__')
                                    and not callable(key)})}

    # run Model

    m.run(steps)

    # Retrieve results

    res["trajectory"] = m.get_trajectory()

    try:
        with open(filename, 'wb') as dumpfile:
            cp.dump(res, dumpfile)
            return 1
    except IOError:
        return -1


def run_experiment(argv):
    """
    Take arv input variables and run sub_experiment accordingly.
    This happens in five steps:
    1)  parse input arguments to set switches
        for [test],
    2)  set output folders according to switches,
    3)  generate parameter combinations,
    4)  define names and dictionaries of callables to apply to sub_experiment
        data for post processing,
    5)  run computation and/or post processing and/or plotting
        depending on execution on cluster or locally or depending on
        experimentation mode.

    Parameters
    ----------
    argv: list[N]
        List of parameters from terminal input

    Returns
    -------
    rt: int
        some return value to show whether sub_experiment succeeded
        return 1 if sucessfull.
    """

    global test
    global steps

    # Parse switches from input
    if len(argv) > 1:
        test = int(argv[1])
    if len(argv) > 2:
        mode = int(argv[2])
    else:
        mode = 0

    # Generate paths according to switches and user name

    test_folder = ['', 'test_output/'][int(test)]
    experiment_folder = 'X10_adjust_agriculture/'
    raw = 'raw_data/'
    res = 'results/'

    if getpass.getuser() == "kolb":
        save_path_raw = "/p/tmp/kolb/Mayasim/output_data/{}{}{}".format(
            test_folder, experiment_folder, raw)
        save_path_res = "/home/kolb/Mayasim/output_data/{}{}{}".format(
            test_folder, experiment_folder, res)
    elif getpass.getuser() == "jakob":
        save_path_raw = \
            "/home/jakob/Project_MayaSim/Python/" \
            "output_data/{}{}{}".format(test_folder, experiment_folder, raw)
        save_path_res = \
            "/home/jakob/Project_MayaSim/Python/" \
            "output_data/{}{}{}".format(test_folder, experiment_folder, res)
    else:
        save_path_res = './{}'.format(res)
        save_path_raw = './{}'.format(raw)

    # Generate parameter combinations

    index = {0: "r_bca", 1: "r_es", 2: "r_trade",
             3: "es_income_mode", 4: "ag_income_mode",
             5: "better_ess", 6: "kill_cropless"}
    if test == 0:
        r_bca = [0.25]

        # parameters for old es income calculation
        #r_eco = [0.0002, 0.00018, 0.00016, 0.00014, 0.00012, 0.0001]
        #r_trade = [6000, 6200, 6400, 6600, 6800]

        # parameters for new es income calculation
        r_eco = [0.08, 0.1, 0.12, 0.16]
        r_trade = [6000, 6400, 6800]

        eco_income_modes = ['sum']
        ag_income_modes = ['sum']
        better_ess = [True]
        kill_cropless = [True, False]
        test = False
    else:
        r_bca = [Parameters.r_bca_sum]
        r_eco = [Parameters.r_es_sum]
        r_trade = [Parameters.r_trade]
        eco_income_modes = ['sum']
        ag_income_modes = ['sum']
        better_ess = [True]
        kill_cropless = [True, False]
        test = True

    # original settings with killing settlements, mean calculation of es and ag
    # and original parameter values for es and ag income calculation.
    original_config = [Parameters.r_bca_mean, Parameters.r_es_mean,
                       Parameters.r_trade, 'mean', 'mean',
                       False, True]
    # plus original config without killing cropless settlements. just to cross
    # check.
    original_config_x =  [Parameters.r_bca_mean, Parameters.r_es_mean,
                          Parameters.r_trade, 'mean', 'mean',
                          False, True]

    param_combs = list(it.product(r_bca, r_eco, r_trade,
                                  eco_income_modes, ag_income_modes,
                                  better_ess, kill_cropless))
    param_combs.append(original_config)
    param_combs.append(original_config_x)

    steps = 1000 if not test else 250
    sample_size = 4 if not test else 1

    # Define names and callables for post processing

    name1 = "aggregated_trajectory"

    estimators1 = {"<mean_trajectories>":
                   lambda fnames:
                   pd.concat([np.load(f)["trajectory"]
                              for f in fnames]).groupby(level=0).mean(),
                   "<sigma_trajectories>":
                   lambda fnames:
                   pd.concat([np.load(f)["trajectory"]
                              for f in fnames]).groupby(level=0).std()
                   }
    name2 = "all_trajectories"

    estimators2 = {"trajectory_list":
                   lambda fnames: [np.load(f)["trajectory"] for f in fnames]}

    def plot_function(steps=1, output_location='./', fnames='./'):
        print(output_location)
        print(fnames)
        input_loc = fnames[0]
        if input_loc.endswith('.pkl'):
            input_loc = input_loc[:-4]

        tail = input_loc.rsplit('/', 1)[1]
        output_location += tail
        print(tail)
        if not os.path.isdir(output_location):
            os.mkdir(output_location)
        mp = MapPlot(t_max=steps,
                     input_location=input_loc,
                     output_location=output_location)

        mp.mplot()
        mp.moviefy(namelist=['frame_'])
        return 1

    name3 = "FramePlots"
    estimators3 = {"map_plots":
                   lambda fnames: plot_function(steps=steps, output_location=save_path_res, fnames=fnames)}

    # Run computation and post processing.

    handle = eh(sample_size=sample_size,
                parameter_combinations=param_combs,
                index=index,
                path_raw=save_path_raw,
                path_res=save_path_res,
                use_kwargs=True)
    print('mode is {}'.format(mode))
    if mode == 0:
        handle.compute(run_func=run_function)
        handle.resave(eva=estimators1, name=name1)
        handle.resave(eva=estimators2, name=name2)
    elif mode == 1:
        handle.resave(eva=estimators3, name=name3, no_output=True)
    elif mode == 2:
        handle.resave(eva=estimators1, name=name1)
        handle.resave(eva=estimators2, name=name2)

    return 1


if __name__ == '__main__':

    run_experiment(sys.argv)
