"""
Experiment to test the influence of drought events.
Drought events start once the civilisation has reached
a 'complex society' state (after 200 years) and vary 
in length and severity from 0 to 100 years and 0 to 100% 
less precipitation.

Therefore, starting point is at t = 200 where the model has
reached a complex society state in all previous studies.
We also use parameters for income from trade, agriculture and
ecosystem services, that have previously proven to lead to
some influence of precipitation variability on the state of the
system.

Also, we vary the parameter for income from trade so see, if
there is a certain parameter value, that results in 
stable complex society for some drought events and collapse for others.
"""

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
from mayasim.model.ModelParameters import ModelParameters as Parameters

test = True



def run_function(d_start=200, d_length=20, d_severity=50.,
                 r_bca=0.2, r_es=0.0002, r_trade=6000,
                 population_control=False,
                 n=30, crop_income_mode='sum',
                 better_ess=True,
                 kill_cropless=False, steps=500, filename='./'):
    """
    Set up the Model for different Parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values and Parameters, as well as the time
    development of aggregated variables for each run.

    Parameters:
    -----------
    d_start : int
        starting point of drought in model time
    d_length : int
        length of drought in timesteps
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
    m.drought_times = [[d_start, d_start + d_length]]
    m.drought_severity = d_severity

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

    if test:
        m.run(3)
    else:
        m.run(steps)

    # Retrieve results

    res["trajectory"] = m.get_trajectory()
    res["final population"] = sum(m.population)
    res["final trade links"] = sum(m.degree)/2.
    res["final max cluster size"] = m.max_cluster_size

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

    test_folder = ['', 'test_output/'][int(test)]
    experiment_folder = 'X9_stability_analysis/'
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

    index = {0: "d_length", 1: "d_severity", 2: "r_trade"}
    if test == 0:
        d_length = list(range(0, 105, 5))
        d_severity = list(range(0, 105, 5))
        r_trade = list(range(6000, 9000, 1000))
        test = False
    else:
        d_length = [20]
        d_severity = [0., 60.]
        r_trade = [6000]
        test = True

    param_combs = list(it.product(d_length, d_severity, r_trade))
    print('computing results for {} parameter combinations'.format(len(param_combs)))

    if len(param_combs)%max_id != 0:
        print('number of jobs ({}) has to be multiple of max_id ({})!!'.format(len(param_combs), max_id))
        exit(-1)

    sample_size = 20 if not test else 3

    # Define names and callables for post processing

    name1 = "trajectory"
    estimators1 = {"<mean_trajectories>":
                   lambda fnames:
                   pd.concat([np.load(f)["trajectory"] for f in fnames]).groupby(level=0).mean(),
                   "<sigma_trajectories>":
                   lambda fnames:
                   pd.concat([np.load(f)["trajectory"] for f in fnames]).groupby(level=0).std()
                   }
    name2 = "all_trajectories"
    estimators2 = {"trajectory_list":
                   lambda fnames: [np.load(f)["trajectory"] for f in fnames]}

    def foo(fnames, keys):
        sample_size = len(fnames)
        key = keys[0]
        data = [np.load(f)[key] for f in fnames]
        df = pd.DataFrame(data=data, columns=[keys[0]])
        for key in keys[1:]:
            data = [np.load(f)[key] for f in fnames]
            df[key] = data
        return df

    name3 = "all_final_states"
    estimators3 = {"final states":
                   lambda fnames:
                   foo(fnames, ["final population",
                                "final trade links",
                                "final max cluster size"])
                   }


    # Run computation and post processing.

    cl = int(len(param_combs)/max_id)
    i = (job_id-1)*cl
    j = job_id*cl

    handle = eh(sample_size=sample_size,
                parameter_combinations=param_combs[i:j],
                index=index,
                path_raw=save_path_raw,
                path_res=save_path_res,
                use_kwargs=True)
    if mode == 1:
        handle.compute(run_func=run_function)
        return 0
    elif mode == 2:
        handle.resave(eva=estimators1, name=name1)
        handle.resave(eva=estimators2, name=name2)
        handle.resave(eva=estimators3, name=name3)
        return 0

    return 1





if __name__ == '__main__':

    run_experiment(sys.argv)
