import getpass
import os
import shutil
import sys
from subprocess import call

from Python.MayaSim.visuals import moviefy

if getpass.getuser() == "kolb":
    SAVE_PATH_RAW = "/p/tmp/kolb/Mayasim/output_data/X1"
    SAVE_PATH_RES = "/home/kolb/Mayasim/output_data/X1"
elif getpass.getuser() == "jakob":
    SAVE_PATH_RAW = "/home/jakob/PhD/Project_MayaSim/Python/output_data/raw/X1"
    SAVE_PATH_RES = "/home/jakob/PhD/Project_MayaSim/Python/output_data/X1"
else:
    SAVE_PATH_RAW = "./RAW"
    SAVE_PATH_RES = "./RES"

if len(sys.argv) > 1:
    sub_experiment = int(sys.argv[1])
else:
    sub_experiment = 0

# Default experiment with standard Parameters:
if sub_experiment == 0:
    from Python.MayaSim.model import Model

    N = 30
    t_max = 325
    save_location_RAW = SAVE_PATH_RAW + '_0_pc'
    save_location_RES = SAVE_PATH_RES + '_0_pc_plots'

    if os.path.exists(save_location_RAW):
        shutil.rmtree(save_location_RAW)
    os.makedirs(save_location_RAW)
    save_location_RAW += "/"
    if os.path.exists(save_location_RES):
        shutil.rmtree(save_location_RES)
    os.makedirs(save_location_RES)
    save_location_RES += "/"

    m = Model(N)
    m.population_control = True
    m.run(t_max, save_location_RAW)
    call(["python", "visuals/mayasim_visuals.py", save_location_RAW,
          save_location_RES, repr(t_max)])

# Experiment with crop income that is calculated as the
# sum over all cropped cells
if sub_experiment == 1:
    from Python.MayaSim.model import Model

    N = 30
    t_max = 325
    save_location_RAW = SAVE_PATH_RAW + '_1_pc'
    save_location_RES = SAVE_PATH_RES + '_1_pc_plots'

    if os.path.exists(save_location_RAW):
        shutil.rmtree(save_location_RAW)
    os.makedirs(save_location_RAW)
    save_location_RAW += "/"
    if os.path.exists(save_location_RES):
        shutil.rmtree(save_location_RES)
    os.makedirs(save_location_RES)
    save_location_RES += "/"

    m = Model(N)
    m.population_control = True
    m.crop_income_mode = "sum"
    m.run(t_max, save_location_RAW)
    call(["python", "visuals/mayasim_visuals.py", save_location_RAW,
          save_location_RES, repr(t_max)])

# movies of results for both sub experiments
if sub_experiment == 2:

    t_max = 325
    for ex in [0, 1]:
        save_location_RES = SAVE_PATH_RES + '_{}_pc_plots/'.format(ex)
        moviefy(save_location_RES)

# plot results without movies (works on cluster)
if sub_experiment == 3:

    N = 30
    t_max = 325
    for ex in [0, 1]:
        save_location_RES = SAVE_PATH_RES + '_{}_pc_plots/'.format(ex)
        save_location_RAW = SAVE_PATH_RAW + '_{}_pc/'.format(ex)
        call(["python", "visuals/mayasim_visuals.py", save_location_RAW,
              save_location_RES, repr(t_max)])
