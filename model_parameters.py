import numpy as np


### *******************************************************************
### RUN PARAMETERS:

#INITIAL NUMBER OF SETTLEMENTS
settlement_start_number = 30  	# 30
min_init_inhabitants = 1000	# 1000
max_init_inhabitants = 2000	# 2000

timesteps = 325			# 325

### *******************************************************************
### MODEL PARAMETERS:
### *******************************************************************


### *******************************************************************
### ECOSYSTEM


#RAINFALL PARAMETERS
veg_rainfall = 0.
# multiplier for precipitation according to climate cycle
precipitationModulation = np.array([-0.06,-0.12,-0.18,-0.12,-0.06,0.,0.06,0.])
# length of period with constant rainfall
climate_var = 3

#WATER FLOW PARAMETERS
#percentage of raindrops that is infitrated per iteration
infitration = 0
#Percentage of cells that receive raindrops 
#(might be redundant, have to look into that -- looked into that, proves to be useful now)
#speeds up water flow by approx. 1/precip_percent
precip_percent = 0.25

#FOREST PARAMETERS
#rate of natural forest depreciation per unit time
natprobdec = 0.003
#Threshold for forest regeneration 1 -> 2
state_change_s2 = 40.
#Threshold for forest regeneration 2 -> 3
state_change_s3 = 100.
#number of state 3 neighbors required for 2 -> 3 regeneration
min_number_of_s3_neighbours = 2

### *******************************************************************
### AGRICULTURE


#WEIGHTS FOR AGRICULTURAL PRODUCTIVITY
a_npp = 0.14 # weight for net primary productivity
a_sp = 84. # for soil productivity
a_s = 18. # for slope
a_wf = 400. # water flow 

#WEIGHTS FOR BENEFIT COST ASSESSMENT TO CROP NEW CELL
max_yield = 1100
origin_shift = 1.11
slope_yield = -0.0052

#PARAMETERS FOR CROPPING CELLS
estab_cost = 900
ag_travel_cost = 950

#PARAMETERS FOR SOIL DEGRADATION:
deg_rate = 5.0 # 5.0 - degradation rate for cropped cells
reg_rate = 2.5 # 2.5 - regeneration rate for state 3 forest cells

#WEIGHTS FOR ECOSYSTEM SERVICES
e_ag = 0.06 # weight for agricultural productivity
e_r = 0. # rain not included in current netlogo version
e_wf = 40. # water flow
e_f = 45. # forest

### *******************************************************************
### SOCIO-ECONOMY


#WEIGHTS FOR INCOME CALCULATION
r_bca = 1.1	# 1.1 - weight agriculture income
r_es = 10. 	# 10. - weight ecosystem services
r_trade = 6000. # 6000. wdight trade income

#DEMOGRAPHIC PARAMETERS
birth_rate_parameter = 0.15

# optionally make birth rate scale inversely with p.c. income
popcontrol = False
max_birth_rate =  0.15 
min_birth_rate = -0.2
shift = 0.325

# death rate correlates inversely with real income per capita
min_death_rate = 0.005
max_death_rate = 0.25

#MIGRATION PREFERENCE PARAMETERS
mig_TC_pref = -0.1
mig_ES_pref =  0.3
        

### *******************************************************************
