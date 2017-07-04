"""
this is the test file for X4, the experiment that checks whether
climate variability actually matters for the overshoot and
collapse pattern
"""

from Experiments import mayasim_X4_es_income as X4

assert X4.run_experiment(['testing', 1]) == 1
