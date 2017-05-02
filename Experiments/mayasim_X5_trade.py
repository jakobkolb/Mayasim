"""
I want to know, if for increasing trade income, there is a transition to a
complex society that can sustain itself.
And I want to know, what happens at that transition. Does it shift, due to
climate variability? Hypothesis: Yes, it does.

Therefore, vary two parameters: r_trade and precipitation_amplitude
"""

from __future__ import print_function

import getpass
import itertools as it
import numpy as np
import sys

import pandas as pd
try:
    import cPickle as cP
except ImportError:
    import pickle as cP

from pymofa.experiment_handling import experiment_handling as handle
from mayasim.model.ModelCore import ModelCore as Model
from mayasim.model.ModelParameters import ModelParameters as Parameters

test = True


def run_function(r_trade=6000., precip_amplitude=1.,
                 n=30, kill_cities_without_cropps=False,
                 steps=350, filename='./'):
    """Initializes and runs model and retrieves and saves data afterwards.

    Parameters
    ----------
    precip_amplitude: float
        the prefactor to the precipitation modulation. 0. means no modulation
        1. means original modulation >1. means amplified modulation.
    r_trade: float
        value of trade income
    n: int
        number of initial settlements
    kill_cities_without_cropps: bool
        switch to set whether or not to kill settlements without agriculture
    steps: int
        number of steps to run the model
    rf_filename: string
    """

    # Initialize Model

    m = Model(n=n, output_data_location=filename, debug=test)
    m.r_trade = r_trade
    m.precipitation_amplitude = precip_amplitude
    m.output_level = 'trajectory'

    # Store initial conditions and parameters:

    res = {"initials": pd.DataFrame({"Settlement X Possitions":
                                     m.settlement_positions[0],
                                     "Settlement Y Possitions":
                                     m.settlement_positions[1],
                                     "Population": m.population}),
           "Parameters": pd.Series({key: getattr(m, key)
                                    for key in dir(Parameters)
                                    if not key.startswith('__')
                                    and not callable(key)})
           }

    # Run model

    if test:
        steps = 5
    m.run(steps)

    # Save results

    res["trajectory"] = m.get_trajectory()

    try:
        with open(filename, 'wb') as dumpfile:
            cP.dump(res, dumpfile)
            return 1
    except IOError:
        return -1


def run_experiment(argv):

    global test

    if len(argv) > 1:
        test = bool(int(argv[1]))

    test_folder = ['', 'test_output/'][int(test)]

    if getpass.getuser() == "kolb":
        save_path_raw = "/p/tmp/kolb/Mayasim/output_data/{}X_ensemble".format(
            test_folder)
        save_path_res = "/home/kolb/Mayasim/output_data/{}X_ensemble".format(
            test_folder)
    elif getpass.getuser() == "jakob":
        save_path_raw = "/home/jakob/Project_MayaSim/Python/" \
                        "output_data/{}X_ensemble/".format(test_folder)
        save_path_res = save_path_raw
    else:
        save_path_res = save_path_raw = './'

        save_path_raw += "raw_data/"
        save_path_res += "results/"

    estimators = {"<mean_trajectories>":
                  lambda fnames: pd.concat([np.load(f)["trajectory"]
                                            for f in fnames]).groupby(
                      level=0).mean(),
                  "<sigma_trajectories>":
                  lambda fnames: pd.concat([np.load(f)["trajectory"]
                                            for f in fnames]).groupby(
                          level=0).std()
                  }

    precip_amplitudes = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.] \
        if not test else [0., 1.]
    r_trades = [5000., 6000., 7000., 8000., 9000., 10000.] \
        if not test else [6000., 8000.]

    parameter_combinations = list(it.product(precip_amplitudes, r_trades))

    name = "trade_income_transition"
    index = {0: 'precip_amplitude', 1: 'r_trade'}
    sample_size = 10

    h = handle(sample_size=sample_size,
               parameter_combinations=parameter_combinations,
               index=index,
               path_raw=save_path_raw,
               path_res=save_path_res,
               use_kwargs=True)

    h.compute(run_func=run_function)
    h.resave(eva=estimators, name=name)


if __name__ == "__main__":

    run_experiment(sys.argv)
