import numpy as np
import scipy.sparse as sparse
import scipy.ndimage as ndimage
import os
import matplotlib.pyplot as plot
import datetime
from mayasim_parameters import *

#recompile fortran routines to make sure dependences
#are matching the available libraries of the particular
#system before importing them.
import subprocess

subprocess.call("./f2py_compile.sh", shell=True)

from f90routines import f90routines

class model:

    def __init__(self, number_settlements_In):
        
        ### *******************************************************************
        ### MODEL DATA SOURCES
        ### *******************************************************************

        input_data_location = "./input_data/"

        # documentation for TEMPERATURE and PRECIPITATION data can be found here: http://www.worldclim.org/formats
        # apparently temperature data is given in x*10 format to allow for smaller file sizes.
        self.temp = np.load(input_data_location + '0_RES_432x400_temp.npy')/10.

        # precipitation in mm or liters pere square meter (comparing the numbers to 
        # numbers from wikipedia suggests, that it is given per year) 
        self.precip = np.load(input_data_location + '0_RES_432x400_precip.npy')    
         
        # in meters above sea level                                                
        self.elev = np.load(input_data_location + '0_RES_432x400_elev.npy')        
        self.slope = np.load(input_data_location + '0_RES_432x400_slope.npy')

        # documentation for SOIL PRODUCTIVITY is given at:
        # http://www.fao.org/geonetwork/srv/en/main.home?uuid=f7a2b3c0-bdbf-11db-a0f6-000d939bc5d8
        # The soil production index considers the suitability of the best adapted crop to each soils 
        # condition in an area and makes a weighted average for all soils present in a pixel based 
        # on the formula: 0.9 * VS + 0.6 * S + 0.3 * MS + 0 * NS. Values range from 0 (bad) to 6 (good)
        self.soilprod = np.load(input_data_location + '0_RES_432x400_soil.npy')
        # to eccount for missing and corrupted data
        self.soilprod[self.soilprod>6] = 6                        
        # smoothen soil productivity dataset
        self.soilprod = ndimage.gaussian_filter(self.soilprod,sigma=(2,2),order=0)
        # and set to zero for non land cells
        self.soilprod[np.isnan(self.elev)] = 0

        ### *******************************************************************
        ### MODEL MAP INITIALIZATION
        ### *******************************************************************

        ### dimensions of the map
        self.rows, self.columns = self.precip.shape
        self.height, self.width = 914., 840. # height and width in km
        self.pixel_dim = self.width/self.columns
        self.cell_width  = self.width/self.columns
        self.cell_height = self.height/self.rows
        self.land_patches = np.asarray(np.where(np.isfinite(self.elev)))
        self.number_of_land_patches = np.shape(self.land_patches)[1]

        self.area = 516484./len(self.land_patches[0])
        self.elev[:,0] = np.inf
        self.elev[:,-1] = np.inf
        self.elev[0,:] = np.inf
        self.elev[-1,:] = np.inf
        #create a list of the index values i = (x, y) of the land patches with finite elevation h
        self.list_of_land_patches = [i for i, h in np.ndenumerate(self.elev) if np.isfinite(self.elev[i])]

        # initialize soil degradation and population gradient (influencing the forest)
        self.soil_deg = np.zeros((self.rows,self.columns))
        self.pop_gradient = np.zeros((self.rows,self.columns))

        ### *******************************************************************
        ### INITiALIZE ECOSYSTEM
        ### *******************************************************************

        self.forest_state               = np.zeros((self.rows,self.columns),dtype=int)
        self.forest_memory              = np.zeros((self.rows,self.columns),dtype=int)
        self.cleared_land_neighbours    = np.zeros((self.rows,self.columns),dtype=int)
        ### The forest has three states: 3=climax forest, 2=secondary regrowth, 1=cleared land.   
        for i in self.list_of_land_patches:
            self.forest_state[i] = 3
    
        ### variables describing total amount of water and water flow
        self.water = np.zeros((self.rows,self.columns))
        self.flow = np.zeros((self.rows,self.columns))
        self.spaciotemporal_precipitation = np.zeros((self.rows,self.columns))

        #initialize the trajectories of the water drops
        self.x = np.zeros((self.rows,self.columns),dtype="int")
        self.y = np.zeros((self.rows,self.columns),dtype="int")

        # define relative coordinates of the neighbourhood of a cell
        self.neighbourhood = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
        self.f90neighbourhood = np.asarray(self.neighbourhood).T

        ### *******************************************************************
        ### INITIALIZE SOCIETY
        ### *******************************************************************

        # crop_income specifies the way, crop income is calculated.
        # possible: 'mean' or 'sum'. default is mean.
        self.crop_income_mode = "mean"

        self.number_settlements = number_settlements_In
        # distribute specified number of settlements on the map
        self.settlement_positions = self.land_patches[:,np.random.choice(
                len(self.land_patches[1]),number_settlements_In).astype('int')]
                
        self.age = [0]*number_settlements_In
        
        # demographic variables
        self.population_control = population_control_parameter
        self.birth_rate = [birth_rate_parameter]*number_settlements_In
        self.death_rate = [0.1 + 0.05 * np.random.random() for i in range(number_settlements_In)]
        self.population = list(np.random.randint(min_init_inhabitants,max_init_inhabitants,number_settlements_In).astype(float))
        self.mig_rate =  [0.]*number_settlements_In
        self.out_mig = [0]*number_settlements_In
        self.pioneer_set = []
        self.failed = 0
        
        #index list for populated and abandoned cities
        #used until removal of dead cities is implemented.
        self.populated_cities = range(number_settlements_In)
        self.dead_cities = []
        
        # agricultural influence
        self.number_cells_in_influence = [0]*number_settlements_In
        self.area_of_influence = [0.]*number_settlements_In
        self.coordinates = np.indices((self.rows,self.columns))
        self.cells_in_influence = [None]*number_settlements_In # will be a list of arrays 

        self.cropped_cells = [None]*number_settlements_In
        # for now, cropped cells are only the city positions.
        # first cropped cells are added at the first call of
        # get_cropped_cells()
        for city in self.populated_cities:
            self.cropped_cells[city] = [[self.settlement_positions[0,city]],[self.settlement_positions[1,city]]]
            

        self.occupied_cells = np.zeros((self.rows,self.columns))
        self.number_cropped_cells = [0]*number_settlements_In
        self.crop_yield = [0.]*number_settlements_In
        self.eco_benefit = [0.]*number_settlements_In
        self.available = 0
       
        #Trade Variables
        self.adjacency = np.zeros((number_settlements_In,number_settlements_In))
        self.rank = [0]*number_settlements_In
        self.degree = [0]*number_settlements_In
        self.comp_size = [0]*number_settlements_In
        self.centrality = [0]*number_settlements_In
        self.trade_income = [0]*number_settlements_In
        
        # total real income per capita
        self.real_income_pc = [0]*number_settlements_In

    def update_precipitation(self,t):
        ### Modulates the initial precip dataset with a 24 timestep period.
        ### Returns a field of rainfall values for each cell.
        ### If veg_rainfall > 0, cleared_land_neighbours decreases rain.

        ### TO DO: The original model increases specialization every time
        ### rainfall decreases, assuming that trade gets more important to
        ### compensate for agriculture decline
        ### *******************************************************************
###EQUATION###################################################################            
        self.spaciotemporal_precipitation = self.precip*(1 
            + precipitationModulation[(np.ceil(t/climate_var)%8).astype(int)]) - veg_rainfall*self.cleared_land_neighbours
###EQUATION###################################################################            

    def get_waterflow(self):

        ## waterflow: takes rain as an argument, uses elev, returns 
        ## water flow distribution
        ## the precip percent parameter that reduces the amount of raindrops that have to be moved.
        ## thereby inceases performance.


        ## f90waterflow takes as arguments:
        ## list of coordinates of land cells (2xN_land)
        ## elevation map in (height x width)
        ## rain_volume per cell map in (height x width) 
        ## rain_volume and elevation must have same units: height per cell
        ## neighbourhood offsets
        ## height and width of map as integers,
        ## Number of land cells, N_land

        # convert precipitation from mm to meters
        # NOTE: I think, this should be 1e-3 to convert from mm to meters though...
        # but 1e-5 is what they do in the original version.
        rain_volume = np.nan_to_num(self.spaciotemporal_precipitation * 1e-5)
        max_x, max_y = self.rows, self.columns
        err, self.flow, self.water = f90routines.f90waterflow(self.land_patches, self.elev, rain_volume, self.f90neighbourhood, max_x, max_y, self.number_of_land_patches)

        return self.water, self.flow
  

    ### to evolve the forest_state
    def forest_evolve(self, npp):
        npp_mean = np.nanmean(npp)
        ### Iterate over all cells repeatedly and regenerate or degenerate
        for repeat in xrange(4):
            for i in self.list_of_land_patches: 
                ### Forest regenerates faster [slower] (linearly) , if net primary productivity on the patch
                ### is above [below] average.
                threshold = npp_mean/npp[i]
                
                ### Degradation:
                ### Decrement with probability 0.003
                ### if there is a settlement around, degrade with higher probability
                Probdec = natprobdec * ( 2*self.pop_gradient[i] + 1 )
                if np.random.random() <= Probdec:
                    if (self.forest_state[i] == 3 ):
                        self.forest_state[i] = 2
                        self.forest_memory[i] = state_change_s2
                    elif (self.forest_state[i] == 2 ):
                        self.forest_state[i] = 1
                        self.forest_memory[i] = 0
                        
                ### Regeneration:"
                ### recover if tree = 1 and memory > threshold 1
                if (self.forest_state[i] == 1 and self.forest_memory[i] > state_change_s2*threshold):
                    self.forest_state[i] = 2
                    self.forest_memory[i] = state_change_s2
                ### recover if tree = 2 and memory > threshold 2 
                ### and certain number of neighbours are climax forest as well
                if (self.forest_state[i] == 2 and self.forest_memory[i] > state_change_s3*threshold):
                    state_3_neighbours = np.sum(self.forest_state[i[0]-1:i[0]+2,i[1]-1:i[1]+2] == 3)
                    if (state_3_neighbours > min_number_of_s3_neighbours):
                        self.forest_state[i] = 3
                    
                ### finally, increase memory by one
                self.forest_memory[i] += 1
        ### calculate cleared land neighbours for output:
        for i in self.list_of_land_patches:
            self.cleared_land_neighbours[i] = np.sum(self.forest_state[i[0]-1:i[0]+2,i[1]-1:i[1]+2] == 1)
            
        return
            

    ###*******************************************************************

    def net_primary_prod(self):
        ### net_primaty_prod is the minimum of a quantity derived from local temperature and rain
        ### Why is it rain and not 'surface water' according to the waterflow model??
    ###EQUATION###################################################################            
        npp = 3000 * np.minimum(1 - np.exp((-6.64e-4 * self.spaciotemporal_precipitation)),
                                  1./(1+np.exp(1.315-(0.119 * self.temp))))
    ###EQUATION###################################################################            
        return npp

    ###*******************************************************************

    def get_ag(self,npp,wf):
        ### agricultural productivit is calculated via a linear additive model from
        ### net primary productivity, soil productivity, slope, waterflow and soil degradation
        ### of each patch.
###EQUATION###################################################################            
        return a_npp*npp + a_sp*self.soilprod - a_s*self.slope - a_wf*wf - self.soil_deg
###EQUATION###################################################################            
        
    ###*******************************************************************
        
    def get_ecoserv(self,ag,wf):
        ### Ecosystem Services are calculated via a linear additive model from 
        ### agricultural productivity (ag), waterflow through the cell (wf) and forest 
        ### state on the cell (forest) \in [1,3], 
        ### The recent version of mayasim limits value of ecosystem services to 1 < ecoserv < 250,
        ### it also proposes to include population density (pop_gradient) and precipitation (rain)
###EQUATION###################################################################            
        return e_ag*ag + e_wf*wf + e_f*(self.forest_state-1.) #+ e_r*rain(t) - e_deg * pop_gradient
###EQUATION###################################################################            

    
######################################################################
### The Society
######################################################################
 

    def benefit_cost(self, ag_In):
        ### Benefit cost assessment
        return (max_yield*(1-origin_shift*np.exp(slope_yield*ag_In)))

    def get_cells_in_influence(self):
        ### creates a list of cells for each city that are under its influence.
        ### these are the cells that are closer than population^0.8/60 (which is
        ### not explained any further...
        self.area_of_influence = [(x**0.8)/60. for x in self.population]
        for city in self.populated_cities:
            stencil = (self.area*(
            (self.settlement_positions[0][city] - self.coordinates[0])**2 +
            (self.settlement_positions[1][city] - self.coordinates[1])**2)
            )    <= self.area_of_influence[city]**2
            self.cells_in_influence[city] = self.coordinates[:,stencil]
        self.number_cells_in_influence = [len(x[0]) for x in self.cells_in_influence]
        
        return
    
    def get_cropped_cells(self,bca):
        # updates the cropped cells for each city with positive population.
        # calculates the utility for each cell (depending on distance from the respective city)
        # If population per cropped cell is lower then min_people_per_cropped_cell,
        # cells are abandoned. Cells with negative utility are also abandoned.
        # If population per cropped cell is higher than
        # max_people_per_cropped_cell, new cells are cropped.
        # Newly cropped cells are chosen such that they have highest utility
        abandoned = 0
        sown = 0

        # for each settlement: how many cells are currently cropped ?
        self.number_cropped_cells = np.array([len(x[0]) for x in self.cropped_cells])
        
        # agricultural population density (people per cropped land) 
        # determines the number of cells that can be cropped
        ag_pop_density = self.population/(self.number_cropped_cells * self.area)
        occup = np.concatenate(self.cropped_cells,axis=1)
        for index in xrange(len(occup[0])):
            self.occupied_cells[occup[0,index],occup[1,index]] = 1
        self.age = [x+1 for x in self.age]
        # for each settlement: which cells to crop ?
        # calculate utility first!
        for city in self.populated_cities:
            distances = np.sqrt(self.area*(
                (self.settlement_positions[0][city] - self.coordinates[0])**2 +
                (self.settlement_positions[1][city] - self.coordinates[1])**2))
###EQUATION###################################################################            
            utility = bca - estab_cost - (ag_travel_cost * distances)/np.sqrt(self.population[city])
###EQUATION###################################################################            

            # 1.) abandon cells if population too low after cities age > 5 years
            if (ag_pop_density[city] < min_people_per_cropped_cell and self.age[city] > 5):
                for number_lost_cells in xrange(np.ceil(30/ag_pop_density[city]).astype('int')):
                    # give up cell with lowest utility
                    cells = self.cropped_cells[city]               
                    abandon_index = np.argmin(utility[cells[0],cells[1]])
                    coord = self.cropped_cells[city][:,abandon_index]
                    self.cropped_cells[city] = np.delete(self.cropped_cells[city],abandon_index,1)
                    self.occupied_cells[coord[0],coord[1]] = 0
                    abandoned += 1

            # 2.) include new cells if population exceeds a threshold 
            for number_new_cells in xrange(np.floor(ag_pop_density[city]/max_people_per_cropped_cell).astype('int')):   
                # choose uncropped cell with maximal utility
                influence = np.zeros((self.rows,self.columns))
                influence[self.cells_in_influence[city][0],self.cells_in_influence[city][1]] = 1
                self.available = np.logical_and(influence.astype('bool'),np.logical_not(self.occupied_cells.astype('bool')))
                if np.any(utility[self.available] > 0):
                    newcell_index = np.unravel_index(np.nanargmax(utility*self.available),utility.shape)
                    self.occupied_cells[newcell_index] = 1
                    self.cropped_cells[city] = np.append(self.cropped_cells[city]
                                ,np.array([[newcell_index[0]],[newcell_index[1]]]),1)
                    sown += 1
                           
                    
            # 3.) abandon cells with utility <= 0
            ut_negative = utility[self.cropped_cells[city][0],self.cropped_cells[city][1]]<=0
            if ( np.sum(ut_negative) > 0):
                abandon_ind = np.where(ut_negative)[0]
                #coor = self.cropped_cells[city][:][abandon_ind]
                coor = [[self.cropped_cells[city][0][ind_x] for ind_x in abandon_ind],[self.cropped_cells[city][1][ind_y] for ind_y in abandon_ind]]
                self.cropped_cells[city] = np.delete(self.cropped_cells[city],abandon_ind,1)
                self.occupied_cells[coor[0],coor[1]] = 0
                abandoned += len(abandon_ind)
        
        # Finally, update list of lists containing cropped cells for each city with 
        # positive population. 
        self.number_cropped_cells = [len(self.cropped_cells[city][0]) for city in range(len(self.population))]

        return abandoned, sown

    def get_pop_mig(self):
        # gives population and out-migration
        print "number of settlements", sum([1 if value>0 else 0 for index, value in enumerate(self.population)])
        
        # death rate correlates inversely with real income per capita
        
        death_rate_diffe = max_death_rate - min_death_rate

        self.death_rate = [-death_rate_diffe * self.real_income_pc[i] + max_death_rate for i in range(len(self.real_income_pc))]
        self.death_rate = [min_death_rate if value<min_death_rate else max_death_rate \
                if value>max_death_rate else self.death_rate[i] \
                for i, value in enumerate(self.death_rate)]
        
        # population control
        if self.population_control == True:
            self.birth_rate = [-(max_birth_rate - min_birth_rate)/10000. * value + shift if value > 5000 else birth_rate_parameter for value in self.population]
        self.population = [int((1. + self.birth_rate[i] - self.death_rate[i])*value) for i, value in enumerate(self.population)]
        self.population = [value if value>0 else 0 for value in self.population]
        
        ### TODO: connect with other model functions
        estab_cost = 900
        self.population = [0 if value < estab_cost*0.4 else value for value in self.population]
        min_mig_rate = 0.
        max_mig_rate = 0.15
        mig_rate_diffe = max_mig_rate - min_mig_rate
        
        # outmigration rate also correlates inversely with real income per capita
        self.mig_rate = [-mig_rate_diffe * self.real_income_pc[i] + max_mig_rate for i in range(len(self.real_income_pc))]
        self.mig_rate = [min_mig_rate if value<min_mig_rate else max_mig_rate if value>max_mig_rate else value for value in self.mig_rate]
        self.out_mig = [int(self.mig_rate[i]*self.population[i]) for i in range(len(self.population))]
        self.out_mig = [value if value>0 else 0 for value in self.out_mig]

        return
        
    ### impact of sociosphere on ecosphere
    def update_pop_gradient(self):
        # pop gradient quantifies the disturbance of the forest by population
        self.pop_gradient = np.zeros((self.rows,self.columns))
        for city in self.populated_cities:
            distance = np.sqrt(self.area*(
                (self.settlement_positions[0][city] - self.coordinates[0])**2 +
                (self.settlement_positions[1][city] - self.coordinates[1])**2))
                
###EQUATION###################################################################            
            self.pop_gradient[self.cells_in_influence[city][0],self.cells_in_influence[city][1]] += self.population[city]/(300*(1+distance[self.cells_in_influence[city][0],self.cells_in_influence[city][1]]))
###EQUATION###################################################################            
            self.pop_gradient[self.pop_gradient>15] = 15
        
    def evolve_soil_deg(self):
        ### soil degrades for cropped cells
        
        cropped = np.concatenate(self.cropped_cells,axis=1)
        self.soil_deg[cropped[0],cropped[1]] += deg_rate
        self.soil_deg[self.forest_state==3] -= reg_rate
        self.soil_deg[self.soil_deg<0] = 0
        
    ###----------------------------------------------------------
    ### functions for trading
    
    def get_rank(self):
        ### depending on population ranks are assigned
        ### attention: ranks are reverted with respect to Netlogo MayaSim !
        ### 1 => 3 ; 2 => 2 ; 3 => 1 
        tresh_1 = 4000.
        tresh_2 = 7000.
        tresh_3 = 9500.
        self.rank = [1 if value > tresh_1 else 2 if value > tresh_2 else 3 if value > tresh_3 else 0 for index, value in enumerate(self.population)]
        return
        
    def build_routes(self):
        ### cities with rank>0 are traders and establish links to neighbours
        for city in self.populated_cities:
            if (self.rank[city] != 0 and self.degree[city] <= self.rank[city]):
                
                distances = (np.sqrt(self.area*((self.settlement_positions[0][city] - self.settlement_positions[0])**2 +
                    (self.settlement_positions[1][city] - self.settlement_positions[1])**2)))
                distances[city] = np.nan # do not choose yourself as nearest neighbour
                distances[self.population<=0] = np.nan # do not choose failed cities
                distances[self.population==np.nan] = np.nan 
                if self.rank[city] == 3:
                    treshold = 31. * (4000./9500.*0.5 +1.)
                elif self.rank[city] == 2:
                    treshold = 31. * (7000./9500.*0.5 +1.)
                elif self.rank[city] == 1:
                    treshold = 31.
                nearby = (distances <= treshold)
                # if there are traders nearby, connect to the one with highest population
                if sum(nearby) != 0:
                    new_partner = np.nanargmax(self.population*nearby)
                    self.adjacency[city,new_partner] = self.adjacency[new_partner,city] = 1
                    
        cut_links = np.array([[True if value > 0 else False for value in self.population]])
        self.adjacency *= cut_links*cut_links.T
        return cut_links
        
    def get_comps(self):
        # convert adjacency matrix to compressed sparse row format
        adjacency_CSR = sparse.csr_matrix(self.adjacency)

        # extract data vector, row index vector and index pointer vector
        A = adjacency_CSR.data
        # add one to make indexing compatible to fortran (where indices start counting with 1)
        JA = adjacency_CSR.indices+1
        IC = adjacency_CSR.indptr+1

        #determine length of data vectors
        l_A = np.shape(A)[0]
        l_IC = np.shape(IC)[0]
        
        # if data vector is not empty, pass data to fortran routine.
        # else, just fill the centrality vector with ones.
        if l_A> 0:
            tmp_comp_size, tmp_degree = f90routines.f90sparsecomponents(IC, A, JA, self.number_settlements, l_IC, l_A)
            self.comp_size, self.degree = list(tmp_comp_size), list(tmp_degree)
        elif l_A == 0:
            self.comp_size, self.degree = [0]*(l_IC-1), [0]*(l_IC-1)
        return

    def get_centrality(self):
        # convert adjacency matrix to compressed sparse row format
        adjacency_CSR = sparse.csr_matrix(self.adjacency)

        # extract data vector, row index vector and index pointer vector
        A = adjacency_CSR.data
        # add one to make indexing compatible to fortran (where indices start counting with 1)
        JA = adjacency_CSR.indices+1
        IC = adjacency_CSR.indptr+1

        #determine length of data vectors
        l_A = np.shape(A)[0]
        l_IC = np.shape(IC)[0]
        print 'number of trade links:', sum(A)/2
        
        # if data vector is not empty, pass data to fortran routine.
        # else, just fill the centrality vector with ones.
        if l_A> 0:
            tmp_centrality = f90routines.f90sparsecentrality(IC, A, JA, self.number_settlements, l_IC, l_A)
            self.centrality = list(tmp_centrality) 
        elif l_A == 0:
            self.centrality = [1]*(l_IC-1)

        return

    def get_crop_income(self,bca):
        # agricultural benefit of cropping
        for city in self.populated_cities:
            crops = bca[self.cropped_cells[city][0],self.cropped_cells[city][1]]
###EQUATION###################################################################
            if self.crop_income_mode == "mean":
                self.crop_yield[city] = np.nanmean(crops[crops>0])
            elif self.crop_income_mode == "sum":
                self.crop_yield[city] = np.nansum(crops[crops>0])
###EQUATION###################################################################            
        self.crop_yield = [0 if np.isnan(self.crop_yield[index]) \
                else self.crop_yield[index] for index in range(len(self.crop_yield))]
        return
        
    def get_eco_income(self,es):
        # benefit from ecosystem services of cells in influence
        for city in self.populated_cities:
###EQUATION###################################################################            
            self.eco_benefit[city] = np.nanmean(es[self.cells_in_influence[city]])
        self.eco_benefit[self.population==0] = 0
###EQUATION###################################################################            
        return

    def get_trade_income(self):
###EQUATION###################################################################            
        self.trade_income = [1./30.*( 1 + self.comp_size[i]/self.centrality[i])**0.9 for i in range(len(self.centrality))]
        self.trade_income = [1 if value>1 else 0 if (value<0 or self.degree[index]==0) else value for index, value in enumerate(self.trade_income)]
###EQUATION###################################################################            
        return
    
    def get_real_income_pc(self):
        ### combine agricultural, ecosystem service and trade benefit
###EQUATION###################################################################            
        self.real_income_pc = [(r_bca * self.crop_yield[index] \
                + r_es * self.eco_benefit[index] \
                + r_trade * self.trade_income[index]) \
                /self.population[index] \
                if value > 0 else 0 \
                for index, value in enumerate(self.population)]
###EQUATION###################################################################            
        return
       
    def migration(self,es):
        ### if outmigration rate exceeds threshold, found new settlement
        
        vacant_lands = np.isfinite(es)
        influenced_cells = np.concatenate(self.cells_in_influence,axis=1)
        vacant_lands[influenced_cells[0],influenced_cells[1]] = 0
        vacant_lands = np.asarray(np.where(vacant_lands == 1))
        for city in self.populated_cities:
            if (self.out_mig[city] > 400 and np.random.rand() <= 0.5 and len(vacant_lands[0])>=75):
                
                mig_pop = self.out_mig[city]
                self.population[city] -= mig_pop
                self.pioneer_set = vacant_lands[:,np.random.choice(len(vacant_lands[0]),75)]
                travel_cost =  np.sqrt(self.area*(
                    (self.settlement_positions[0][city] - self.coordinates[0])**2 +
                    (self.settlement_positions[1][city] - self.coordinates[1])**2))
                utility = mig_ES_pref * es + mig_TC_pref * travel_cost 
                utofpio = utility[self.pioneer_set[0],self.pioneer_set[1]]
                new_loc = self.pioneer_set[:,np.nanargmax(utofpio)]
                neighbours = (np.sqrt(self.area*((new_loc[0] - self.settlement_positions[0][self.population>0])**2 +
                    (new_loc[1] - self.settlement_positions[1][self.population>0])**2))) <= 7.5
                summe = np.sum(neighbours)
                
                if summe==0:
                    self.newcity(new_loc[0],new_loc[1],mig_pop)
                    index = (vacant_lands[0,:]==new_loc[0])&(vacant_lands[1,:]==new_loc[1])
                    np.delete(vacant_lands,index,1)
        
        
    def kill_cities(self):

        dead_city_indices = [index for index in range(len(self.population)) if (self.population[index] <= 0 or len(self.cropped_cells[index][0])<=0)]

        #remove entries from variables
        #simple lists that can be deleted elementwise
        for index in sorted(dead_city_indices, reverse=True):
            self.number_settlements -= 1
            self.failed += 1
            del self.age[index]
            del self.birth_rate[index]
            del self.death_rate[index]
            del self.population[index]
            del self.mig_rate[index]
            del self.out_mig[index]
            del self.number_cells_in_influence[index]
            del self.area_of_influence[index]
            del self.number_cropped_cells[index]
            del self.crop_yield[index]
            del self.eco_benefit[index]
            del self.rank[index]
            del self.degree[index]
            del self.comp_size[index]
            del self.centrality[index]
            del self.trade_income[index]
            del self.real_income_pc[index]
            del self.cells_in_influence[index]
            del self.cropped_cells[index]
        
        #special cases:
        self.settlement_positions = np.delete(self.settlement_positions, dead_city_indices, axis=1)
        self.adjacency = np.delete(np.delete(self.adjacency, dead_city_indices, axis=0),dead_city_indices, axis=1)


        #update list of indices for populated and dead cities

        # a) update list of populated cities
        self.populated_cities = [index for index, value in enumerate(self.population) if value>0]

        # b) update list of dead cities
        self.dead_cities = [index for index, value in enumerate(self.population) if value==0]

        return

    def newcity(self,a,b,mig_pop):
        ### extend all variables to include new city
        self.number_settlements += 1
        self.settlement_positions = np.append(self.settlement_positions,[[a],[b]],1)
        self.age.append(0)
        self.birth_rate.append(birth_rate_parameter)
        self.death_rate.append(0.1 + 0.05 * np.random.rand())
        self.population.append(mig_pop)
        self.mig_rate.append(0)
        self.out_mig.append(0)
        self.number_cells_in_influence.append(0)
        self.area_of_influence.append(0)
        self.cells_in_influence.append([[a],[b]])
        self.cropped_cells.append([[a],[b]])
        self.number_cropped_cells.append(1)
        self.crop_yield.append(0)
        self.eco_benefit.append(0)
        self.rank.append(0)
        N = len(self.adjacency)
        self.adjacency = np.append(self.adjacency,[[0]*N],0)
        self.adjacency = np.append(self.adjacency,[[0]]*(N+1),1)
        self.degree.append(0)
        self.comp_size.append(0)
        self.centrality.append(0)
        self.trade_income.append(0)
        self.real_income_pc.append(0)

    def run(self, t_max, location):

        #### initialize time step
        t = 0

        #### initialize variables
        npp = np.zeros((self.rows,self.columns))# net primary productivity
        wf = np.zeros((self.rows,self.columns)) # water flow
        ag = np.zeros((self.rows,self.columns)) # agricultural productivity
        es = np.zeros((self.rows,self.columns)) # ecosystem services
        bca = np.zeros((self.rows,self.columns))# benefit cost map for agriculture


        print "timeloop starts now"

        while t <= t_max:
            t += 1
            print "time = ", t
            
            ### evolve subselfs
            # ecosystem
            self.update_precipitation(t)
            npp = self.net_primary_prod()
            self.forest_evolve(npp)
            wf = self.get_waterflow()[1] # this is curious: only waterflow is used, water level is abandoned.
            ag = self.get_ag(npp,wf)
            es = self.get_ecoserv(ag,wf)
            bca = self.benefit_cost(ag)
            
            # society
            self.get_cells_in_influence()
            abandoned, sown = self.get_cropped_cells(bca)
            self.get_crop_income(bca)
            self.get_eco_income(es)
            self.evolve_soil_deg()
            self.update_pop_gradient()
            self.get_rank()
            cl = self.build_routes()
            self.get_comps()
            self.get_centrality()
            self.get_trade_income()
            self.get_real_income_pc()
            self.get_pop_mig()
            self.migration(es)
            self.kill_cities()

            ### save variables of interest      
            np.save(location+"rain_%03d.npy"%(t,),self.spaciotemporal_precipitation)
            np.save(location+"npp_%03d.npy"%(t,),npp)
            np.save(location+"forest_%03d.npy"%(t,),self.forest_state)
            np.save(location+"waterflow_%03d.npy"%(t,),wf)
            np.save(location+"AG_%03d.npy"%(t,),ag)
            np.save(location+"ES_%03d.npy"%(t,),es)
            np.save(location+"bca_%03d.npy"%(t,),bca)
            
            def stack_ragged(array_list, axis=0):
                lengths = [np.shape(a)[axis] for a in array_list]
                idx = np.cumsum(lengths[:-1])
                stacked = np.concatenate(array_list, axis=axis)
                return stacked, idx
            def save_stacked_array(fname, array_list, axis=0):
                stacked, idx = stack_ragged(array_list, axis=axis)
                np.savez(fname, stacked_array=stacked, stacked_index=idx)
            def load_stacked_arrays(fname, axis=0):
                npzfile = np.load(fname)
                idx = npzfile['stacked_index']
                stacked = npzfile['stacked_array']
                return np.split(stacked, idx, axis=axis)
            
            save_stacked_array(location+"cells_in_influence_%03d"%(t,),self.cells_in_influence,axis=1)
            np.save(location+"number_cells_in_influence_%03d.npy"%(t,),self.number_cells_in_influence)
            save_stacked_array(location+"cropped_cells_%03d"%(t,),self.cropped_cells,axis=1)
            np.save(location+"number_cropped_cells_%03d.npy"%(t,),self.number_cropped_cells)
            np.save(location+"abnd_sown_%03d.npy"%(t,),np.array((abandoned,sown)))
            np.save(location+"crop_yield_%03d.npy"%(t,),self.crop_yield)
            np.save(location+"eco_benefit_pc_%03d.npy"%(t,),self.eco_benefit)
            np.save(location+"real_income_pc_%03d.npy"%(t,),self.real_income_pc)
            np.save(location+"population_%03d.npy"%(t,),self.population)
            np.save(location+"out_mig_%03d.npy"%(t,),self.out_mig)
            np.save(location+"death_rate_%03d.npy"%(t,),self.death_rate)
            np.save(location+"soil_deg_%03d.npy"%(t,),self.soil_deg)

            np.save(location+"pop_gradient_%03d.npy"%(t,),self.pop_gradient)
            np.save(location+"adjacency_%03d.npy"%(t,),self.adjacency)
            np.save(location+"degree_%03d.npy"%(t,),self.degree)
            np.save(location+"comp_size_%03d.npy"%(t,),self.comp_size)
            np.save(location+"centrality_%03d.npy"%(t,),self.centrality)
            np.save(location+"trade_income_%03d.npy"%(t,),self.trade_income)
           
            np.save(location+"number_settlements_%03d.npy"%(t,),self.number_settlements)
            np.save(location+"settlement_positions_%03d.npy"%(t,),self.settlement_positions) 

if __name__ == "__main__":

    #### initialize model
    model = model(settlement_start_number)

    ### define saving location
    comment = "testing_version"
    now = datetime.datetime.now()
    location = "output_data/"+now.strftime("%d_%m_%H-%M-%Ss")+"_Output_"+comment + '/'
    os.makedirs(location)


    ### run model
    model.run(timesteps, location)
