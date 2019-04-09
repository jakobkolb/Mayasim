"""
Experiment to generate trajectories for different possible of income
from ecosystem and trade. These trajectories will then be used to
analyze the bifurcation parameter of the model. Also, they can be
used to find the optimal values for r_trade and r_es that reproduce
the original behavior of the model in Heckberts publication.

Parameters that are varied are
1) r_trade
2) r_es
The model runs for t=2000 time steps as I hope that this will be short enough
to finish on the cluster in 24h and long enough to provide enough data for the
analysis of frequency of oscillations from the trajectories.
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
steps = 2000


def run_function(r_bca=0.2, r_es=0.0002, r_trade=6000,
                 population_control=False,
                 n=30, crop_income_mode='sum',
                 better_ess=True,
                 kill_cropless=False, filename='./'):
    """
    Set up the Model for different Parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values and Parameters, as well as the time
    development of aggregated variables for each run.

    Parameters:
    -----------
    d_times: list of lists
        list of list of start and end dates of droughts
    d_severity : float
        severity of drought (decrease in rainfall in percent)
    r_bca : float > 0
        the prefactor for income from agriculture
    r_es : float
        the prefactor for income from ecosystem services
    r_trade : float
        the prefactor for income from trade
    population_control : boolean
        determines whether the population grows
        unbounded or if population growth decreases
        with income per capita and population density.
    n : int > 0
        initial number of settlements on the map
    crop_income_mode : string
        defines the mode of crop income calculation.
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

    m = Model(n, output_data_location=filename, debug=test)
    if not filename.endswith('s0.pkl'):
        m.output_geographic_data = False
        m.output_settlement_data = False

    m.population_control = population_control
    m.crop_income_mode = crop_income_mode
    m.better_ess = better_ess
    m.r_bca_sum = r_bca
    m.r_es_sum = r_es
    m.r_trade = r_trade
    m.kill_cities_without_crops = kill_cropless

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
        mode = None
    if len(argv) > 3:
        job_id = int(argv[3])
    else:
        job_id = 1
    if len(argv) > 4:
        max_id = int(argv[4])
    else:
        max_id = 1


    # Generate paths according to switches and user name

    test_folder = 'test_output/' if test else ''
    experiment_folder = 'X8_longterm_dynamics/'
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

    index = {0: "r_trade", 1: "r_es"}
    if test == 0:
        r_trade =   [round(x,5) for x in np.arange(6000,10000,500)]
        r_es =      [round(x,5) for x in np.arange(0.00008,0.00018,0.00001)]
        test = False
    else:
        r_trade = [6000, 7000]
        r_es = [0.0002, 0.0001]
        test = True

    param_combs = list(it.product(r_trade, r_es))

    steps = 2000 if not test else 50
    sample_size = 20 if not test else 1

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

    def plot_function(steps=1, input_location='./', output_location='./', fnames='./'):
        print(input_location)
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
                   lambda fnames: plot_function(steps=steps,
                                                input_location=save_path_raw,
                                                output_location=save_path_res,
                                                fnames=fnames)
                  }

    # Run computation and post processing.

    def chunk_arr(i, array, i_min, i_max):
        """
        split array in equally sized (except for the last
        one which is shorter) chunks and return the i-th
        """
        la = len(array)
        irange = i_max - i_min
        di = int(np.floor(la/irange)) if irange > 1 else 1
        i0 = i-i_min
        i1 = i0*di
        i2 = (i0+1)*di
        if i < i_max:
            return array[i1:i2]
        if i == i_max:
            return array[i1:]

    handle = eh(sample_size=sample_size,
                parameter_combinations=chunk_arr(job_id, param_combs, 1, max_id),
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
