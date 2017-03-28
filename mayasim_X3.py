
import os
import datetime
import sys
import getpass
import shutil
import numpy as np
from subprocess import call

if getpass.getuser() == "kolb":
    SAVE_PATH = "/p/tmp/kolb/Mayasim/output_data/X3"
elif getpass.getuser() == "jakob":
    SAVE_PATH = "/home/jakob/PhD/Project_MayaSim/Python/output_data/X3"

if len(sys.argv)>1:
    sub_experiment = int(sys.argv[1])
else:
    sub_experiment = 0



# Default experiment with standard parameters:
if sub_experiment == 0:
    print 'starting sub-experiment No 0'
    from mayasim_model.model import model

    N = 30
    t_max = 325

    r_bca_values = np.linspace(0.08, 0.2, 7)
    print r_bca_values

    for i, r_bca_value in enumerate(r_bca_values):
        print r_bca_value

        save_location = SAVE_PATH + "_%03f_npc"%(r_bca_value,)
        
        if os.path.exists(save_location):
            shutil.rmtree(save_location)
        os.makedirs(save_location)
        save_location += "/"

        m = model(N)
        m.output_level = 'spatial'
        m.crop_income_mode = 'sum'
        m.r_bca = r_bca_value
        m.population_control = False
        m.run(t_max, save_location)
        m.save_run_variables(save_location)
        call(["python", "visuals/mayasim_visuals.py", save_location, `t_max`])

# Experiment with crop income that is calculated as the
# sum over all cropped cells
if sub_experiment == 1:
    print 'starting sub-experiment No 1'
    from mayasim_model.model import model

    N = 30
    t_max = 325

    r_bca_values = np.linspace(0.08, 0.2, 7)

    for i, r_bca_value in enumerate(r_bca_values):
        print r_bca_value

        save_location = SAVE_PATH + "_%03f_pc"%(r_bca_value,)
        
        if os.path.exists(save_location):
            shutil.rmtree(save_location)
        os.makedirs(save_location)
        save_location += "/"

        m = model(N)
        m.output_level = 'spatial'
        m.crop_income_mode = 'sum'
        m.r_bca = r_bca_value
        m.population_control = True
        m.run(t_max, save_location)
        m.save_run_variables(save_location)
        call(["python", "visuals/mayasim_visuals.py", save_location, `t_max`])

#plot results for both sub experiments
if sub_experiment == 2:

    t_max = 325


    r_bca_values = np.linspace(0.08, 0.2, 0.02)

    for i, r_bca_value in enumerate(r_bca_values):

        save_location = SAVE_PATH + "_%03d_npc"%(r_bca_value,)
        call(["python", "visuals/mayasim_visuals.py", save_location, `t_max`])

        save_location = SAVE_PATH + "_%03d_pc"%(r_bca_value,)
        call(["python", "visuals/mayasim_visuals.py", save_location, `t_max`])
