import getpass
import os
import shutil
import sys
from subprocess import call

from visuals.moviefy import moviefy

if getpass.getuser() == "kolb":
    SAVE_PATH_RAW = "/p/tmp/kolb/Mayasim/output_data/X2"
    SAVE_PATH_RES = "/home/kolb/Mayasim/output_data/X2"
elif getpass.getuser() == "jakob":
    SAVE_PATH_RAW = "/home/jakob/PhD/Project_MayaSim/Python/output_data/raw/X2"
    SAVE_PATH_RES = "/home/jakob/PhD/Project_MayaSim/Python/output_data/X2"
else:
    SAVE_PATH_RAW = "./RAW"
    SAVE_PATH_RES = "./RES"

if len(sys.argv) > 1:
    sub_experiment = int(sys.argv[1])
else:
    sub_experiment = 0

# Default experiment with standard Parameters:
if sub_experiment == 0:
    from mayasim_model.model import Model

    N = 30
    t_max = 325

    save_location_RAW = SAVE_PATH_RAW + '_0_npc'
    save_location_RES = SAVE_PATH_RES + '_0_npc_plots'

    if os.path.exists(save_location_RAW):
        shutil.rmtree(save_location_RAW)
    os.makedirs(save_location_RAW)
    save_location_RAW += "/"
    if os.path.exists(save_location_RES):
        shutil.rmtree(save_location_RES)
    os.makedirs(save_location_RES)
    save_location_RES += "/"

    m = Model(N)
    m.population_control = False
    m.run(t_max, save_location_RAW)
    call(["python", "visuals/mayasim_visuals.py", save_location_RAW,
          save_location_RES, repr(t_max)])

# Experiment with crop income that is calculated as the
# sum over all cropped cells
if sub_experiment == 1:
    from mayasim_model.model import Model

    N = 30
    t_max = 325

    save_location_RAW = SAVE_PATH_RAW + '_1_npc'
    save_location_RES = SAVE_PATH_RES + '_1_npc_plots'

    if os.path.exists(save_location_RAW):
        shutil.rmtree(save_location_RAW)
    os.makedirs(save_location_RAW)
    save_location_RAW += "/"
    if os.path.exists(save_location_RES):
        shutil.rmtree(save_location_RES)
    os.makedirs(save_location_RES)
    save_location_RES += "/"

    m = Model(N)
    m.crop_income_mode = "sum"
    m.population_control = False
    m.run(t_max, save_location_RAW)
    call(["python", "visuals/mayasim_visuals.py", save_location_RAW,
          save_location_RES, repr(t_max)])

# plot results for both sub experiments
if sub_experiment == 2:

    t_max = 325

    for x in [0, 1]:
        save_location_RES = SAVE_PATH_RES + '_{:d}_npc_plots'.format(x) + "/"
        moviefy(save_location_RES)
