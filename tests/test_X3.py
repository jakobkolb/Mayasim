"""this is the test file for X5 the experiment
concerning the influence of trade income"""

from Experiments import mayasim_X3_trade as X3

assert X3.run_experiment(['testing', 1]) == 1
