
import os
import sys
import getpass
import shutil
import numpy as np
from subprocess import call
from visuals.moviefy import moviefy

if getpass.getuser() == "kolb":
    SAVE_PATH_RAW = "/p/tmp/kolb/Mayasim/output_data/X3"
    SAVE_PATH_RES = "/home/kolb/Mayasim/output_data/X3"
elif getpass.getuser() == "jakob":
    SAVE_PATH_RAW = "/home/jakob/PhD/Project_MayaSim/Python/output_data/raw/X3"
    SAVE_PATH_RES = "/home/jakob/PhD/Project_MayaSim/Python/output_data/X3"

if len(sys.argv) > 1:
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

        save_location_RAW = SAVE_PATH_RAW + "_%03f_npc"%(r_bca_value,)
        save_location_RES = SAVE_PATH_RES + "_%03f_npc_plots"%(r_bca_value,)

        if os.path.exists(save_location_RAW):
            shutil.rmtree(save_location_RAW)
        os.makedirs(save_location_RAW)
        save_location_RAW += "/"
        if os.path.exists(save_location_RES):
            shutil.rmtree(save_location_RES)
        os.makedirs(save_location_RES)
        save_location_RES += "/"

        m = model(N)
        m.output_level = 'spatial'
        m.crop_income_mode = 'sum'
        m.r_bca = r_bca_value
        m.population_control = False
        m.run(t_max, save_location_RAW)
        m.save_run_variables(save_location_RAW)
        call(["python", "visuals/mayasim_visuals.py", save_location_RAW,
              SAVE_PATH_RES, `t_max`])
        moviefy(save_location_RES)

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

        save_location_RAW = SAVE_PATH_RAW + "_%03f_pc"%(r_bca_value,)
        save_location_RES = SAVE_PATH_RES + "_%03f_pc_plots"%(r_bca_value,)

        if os.path.exists(save_location_RAW):
            shutil.rmtree(save_location_RAW)
        os.makedirs(save_location_RAW)
        save_location_RAW += "/"
        if os.path.exists(save_location_RES):
            shutil.rmtree(save_location_RES)
        os.makedirs(save_location_RES)
        save_location_RES += "/"

        m = model(N)
        m.output_level = 'spatial'
        m.crop_income_mode = 'sum'
        m.r_bca = r_bca_value
        m.population_control = True
        m.run(t_max, save_location_RAW)
        m.save_run_variables(save_location_RAW)
        call(["python", "visuals/mayasim_visuals.py", save_location_RAW,
              SAVE_PATH_RES, `t_max`])
        moviefy(save_location_RES)

# plot results for both sub experiments
if sub_experiment == 2:
    print 'plotting only'

    t_max = 325

    r_bca_values = np.linspace(0.08, 0.2, 7)
    print r_bca_values

    for i, r_bca_value in enumerate(r_bca_values):
        print r_bca_value

        save_location_RAW = SAVE_PATH_RAW + "_%03f_npc"%(r_bca_value,)
        save_location_RES = SAVE_PATH_RES + "_%03f_npc_plots"%(r_bca_value,)
        call(["python", "visuals/mayasim_visuals.py", save_location_RAW,
              SAVE_PATH_RES, `t_max`])
        moviefy(save_location_RES)

        save_location_RAW = SAVE_PATH_RAW + "_%03f_pc"%(r_bca_value,)
        save_location_RES = SAVE_PATH_RES + "_%03f_pc_plots"%(r_bca_value,)
        call(["python", "visuals/mayasim_visuals.py", save_location_RAW,
              SAVE_PATH_RES, `t_max`])
        moviefy(save_location_RES)
