"""this is the test file for X7, the experiment concerning income from agriculture and ecosystem services"""

from Experiments import mayasim_X2_scan_r_es_and_r_agg as X7

assert X7.run_experiment(['testing', 1]) == 1
