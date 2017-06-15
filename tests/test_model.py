"""
test script for the model itself. 
Just checking whether it runs without 
errors, no sanity check so far
"""

from mayasim.model.ModelCore import ModelCore as M

m_instance = M()

assert m_instance.run_test(1) == 1