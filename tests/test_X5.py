"""this is the test file for X5 the experiment concerning the influence of trade income"""

from ..Experiments import mayasim_X5_trade as X5

assert X5.run_experiment(['testing', 1]) == 1