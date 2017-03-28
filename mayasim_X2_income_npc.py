
import os
import datetime
import sys
import getpass
import shutil
from subprocess import call

if getpass.getuser() == "kolb":
    SAVE_PATH = "/home/kolb/Mayasim/output_data/X1"
elif getpass.getuser() == "jakob":
    SAVE_PATH = "/home/jakob/PhD/Project_MayaSim/Python/output_data/X1"

if len(sys.argv)>1:
    sub_experiment = int(sys.argv[1])
else:
    sub_experiment = 0

# Default experiment with standard parameters:
if sub_experiment == 0:
    from mayasim_model.model import model

    N = 30
    t_max = 325
    save_location = SAVE_PATH + '_0_npc'
    
    if os.path.exists(save_location):
        shutil.rmtree(save_location)
    os.makedirs(save_location)
    save_location += "/"

    m = model(N)
    m.population_control = False
    m.run(t_max, save_location)
    call(["python", "visuals/mayasim_visuals.py", save_location, `t_max`])

# Experiment with crop income that is calculated as the
# sum over all cropped cells
if sub_experiment == 1:
    from mayasim_model.model import model

    N = 30
    t_max = 325
    save_location = SAVE_PATH + '_1_npc'

    if os.path.exists(save_location):
        shutil.rmtree(save_location)
    os.makedirs(save_location)
    save_location += "/"

    m = model(N)
    m.crop_income_mode = "sum"
    m.population_control = False
    m.run(t_max, save_location)
    call(["python", "visuals/mayasim_visuals.py", save_location, `t_max`])

#plot results for both sub experiments
if sub_experiment == 2:

    t_max = 325

    save_location = SAVE_PATH + '_0_npc/'
    call(["python", "/visuals/mayasim_visuals.py", save_location, `t_max`])

    save_location = SAVE_PATH + '_1_npc/'
    call(["python", "visuals/mayasim_visuals.py", save_location, `t_max`])
