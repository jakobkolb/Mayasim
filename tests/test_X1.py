"""this is the test file for X1, the first experiment"""

from Experiments import mayasim_X1_default_setup as X1

assert X1.run_experiment(['testing', 1]) == 1
