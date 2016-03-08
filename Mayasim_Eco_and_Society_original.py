

import numpy as np
import matplotlib.pyplot as plot
# from copy import deepcopy
import scipy.ndimage as ndimage
import os
import datetime

debug = True

#def main():

### load the initial variables
temp = np.load('0_RES_432x400_temp.npy')
temp = temp/12. ### factor 1/12 as in netlogo version
precip = np.load('0_RES_432x400_precip.npy')
elev = np.load('0_RES_432x400_elev.npy')
soilprod = np.load('0_RES_432x400_soil.npy')
np.savetxt('soilprod_clean.csv', soilprod)
soilprod[soilprod>=6] = 1.5 ### no soilprod >= 6, as in netlogo version
soilprod[elev<=1] = 1.5 ### as in netlogo version
np.savetxt('soilprod_modified.csv', soilprod)
slope = np.load('0_RES_432x400_slope.npy')

# smoothen soil productivity dataset
soilprod = ndimage.gaussian_filter(soilprod,sigma=(2,2),order=0)
soilprod[np.isnan(elev)] = 0

# dimensions of the map
rows,columns = precip.shape
height, width = 914., 840. # height and width in km
pixel_dim = width/columns
land_patches = np.asarray(np.where(np.isfinite(elev)))
area = 516484./len(land_patches[0])

elev[:,0] = np.inf
elev[:,-1] = np.inf
elev[0,:] = np.inf
elev[-1,:] = np.inf

# initalize soil degradation and population gradient (influencing the forest)
soil_deg = np.zeros((rows,columns))

# initialize timestep
t = 0

### *******************************************************************
### Ecosystem. Comprises rain and npp, waterflow
### and forest
###
### *******************************************************************

def rain_function(t,cleared_land_neighbours):
    ### Modulates the initial precip dataset with a 24 timestep period.
    ### If veg_rainfall != 0, cleared_land_neighbours decrease rain.
    modulation = np.array([-0.06,-0.12,-0.18,-0.12,-0.06,0.,0.06,0.])
    veg_rainfall = 0.
    return (precip*(1+modulation[(np.ceil(t/3)%8).astype(int)]) - veg_rainfall*cleared_land_neighbours)
    

def net_primary_prod(t,rain):
    ### net_primaty_prod is the minimum of a quantity derived from rain and temperature
    npp = 3000 * np.minimum(1 - np.exp((-6.64e-4 * rain)),
                              1./(1+np.exp(1.315-(0.119 * temp))))
    return npp

###*******************************************************************

class Waterflow:
    ### variable describing total amount of water and water flow
    water = np.zeros((rows,columns))
    flow = np.zeros((rows,columns))
    
    ### define schemes translating the return of nanargmin to changes
    ### in position of the raindrops. XXX: direction 4->0!
    schemex = np.array([-1,-1,-1,0,0,0,1,1,1])
    schemey = np.array([-1,0,1,-1,0,1,-1,0,1])
    
    def get_waterflow(self,rain_volume_In):
        ### waterflow: takes rain as an argument, uses elev, returns 
        ### water flow distribution
        x = np.zeros((rows,columns),dtype="int")
        y = np.zeros((rows,columns),dtype="int")
        precip_percent = 0.25 # in netlogo, there is a "precip-percent slider"
        rain_volume = rain_volume_In * 1e-5 / precip_percent 
        rain_volume = rain_volume * (np.random.rand(432,400)<=precip_percent)
        
        generator = (i for i, x in np.ndenumerate(elev) if np.isfinite(elev[i]))
        for i,j in generator:
            x[i][j] = i
            y[i][j] = j
        
        self.flow[:,:] = 0 ## set the flow and water distribution to zero
        self.water[:,:] = 0 # rain_volume[:,:]
        
        ## let the drops move 10 times to the lowest adjacent cell 
        for wfStep in xrange(10):
            Z = elev + self.water
            self.water[:,:] = 0
            generator = (i for i, x in np.ndenumerate(elev) if np.isfinite(elev[i]))
            for (i,j) in generator:
                xx = x[i,j]
                yy = y[i,j]
                options = [Z[xx-1,yy-1],Z[xx-1,yy],Z[xx-1,yy+1],Z[xx,yy-1],Z[xx,yy],Z[xx,yy+1],Z[xx+1,yy-1],Z[xx+1,yy],Z[xx+1,yy+1]]
                direction = np.where(options == np.nanmin(options))
                if len(direction[0])>1:
                    direction = np.random.choice(direction[0])
                else:
                    direction = direction[0][0]
                # direction = np.nanargmin([Z[xx,yy],Z[xx-1,yy-1],Z[xx-1,yy],Z[xx-1,yy+1],Z[xx,yy-1],Z[xx,yy+1],Z[xx+1,yy-1],Z[xx+1,yy],Z[xx+1,yy+1]])
                
                x[i,j] += self.schemex[direction]
                y[i,j] += self.schemey[direction]
                ### if drop stays where it was, it does not contribute to flow
                self.flow[x[i,j],y[i,j]] += (direction!=4)*rain_volume[i,j] 
                self.water[x[i,j],y[i,j]] += rain_volume[i,j]

        if debug:
	    print 'total water', sum(sum(self.water))
	    print 'total rain', sum(sum(rain_volume))
	    print 'total flow', sum(sum(self.flow))
            fig = plot.figure()
            ax1 = fig.add_subplot(121)
            plot.imshow(self.flow, vmin=0, vmax=2)
            plot.colorbar()
            ax2 = fig.add_subplot(122)
            plot.imshow(self.water, vmin=0, vmax=.5)
            plot.colorbar()
            plot.show()
        return self.water, self.flow
  
###*******************************************************************

class Forest:
    # upon initialization:
    def __init__(self):
        self.forest_state = np.zeros((rows,columns),dtype=int)
        # self.forest_state[:,:] = np.nan # np.nan doesnt seem to exist for int
        ### define generator for as all tuples (i,j) on land
        generator = (i for i, x in np.ndenumerate(elev) if np.isfinite(elev[i]))
        ### The forest has three states: 3=climax forest, 2=secondary regrowth, 1=cleared land.   
        for i in generator:
            self.forest_state[i] = 3
        self.forest_memory = np.zeros((rows,columns),dtype=int)
        self.state_1_neighbours = np.zeros((rows,columns),dtype=int)
    
    ### to evolve the forest_state
    def forest_evolve(self, npp,pop_gradient):
        npp_mean = np.nanmean(npp)
        for repeat in xrange(4):
            generator = (i for i, x in np.ndenumerate(elev) if np.isfinite(elev[i]))
            for i in generator:  
                threshold = npp_mean/npp[i]
                
                ### Degradation:
                ### Decrement with probability 0.003
                ### if there is a settlement around, decrease with higher probability
                natprobdec = 0.003
                Probdec = natprobdec * ( 2*pop_gradient[i] + 1 )
                if np.random.random() <= Probdec:
                    if (self.forest_state[i] == 3 ):
                        self.forest_state[i] = 2
                        self.forest_memory[i] = 40
                    elif (self.forest_state[i] == 2 ):
                        self.forest_state[i] = 1
                        self.forest_memory[i] = 0
                        
                ### Regeneration:
                ### recover if tree = 1 and memory > threshold 1
                if (self.forest_state[i] == 1 and self.forest_memory[i] > 40*threshold):
                    self.forest_state[i] = 2
                    self.forest_memory[i] = 40
                    
                ### recover if tree = 2 and memory > threshold 2 
                ### and certain number of neighbours are climax forest as well
                state_3_neighbours = np.sum(self.forest_state[i[0]-1:i[0]+2,i[1]-1:i[1]+2] == 3)
                self.state_1_neighbours[i] = np.sum(self.forest_state[i[0]-1:i[0]+2,i[1]-1:i[1]+2] == 1)
                if (self.forest_state[i] == 2 and self.forest_memory[i] > 100*threshold and state_3_neighbours > 2):
                    self.forest_state[i] = 3
                    
                ### finally, increase memory by one
                self.forest_memory[i] += 1
            
        return self.forest_state, self.state_1_neighbours
        
###*******************************************************************

def get_agri(npp,wf):
    ### agricultural productivity
    a_npp = 0.14 # weight for npp
    a_sp = 84. # for soil productivity
    a_s = 18. # for slope
    a_wf = 400. # waterflow 
    return a_npp*npp + a_sp*soilprod - a_s*slope - a_wf*wf - soil_deg
    
###*******************************************************************
    
def get_ecoserv(ag,wf,forest):
    ### Ecosystem Services
    e_ag = 0.06 # weight for agricultural productivity
    #e_r = 1. # XXX: rain not included in current netlogo version
    e_wf = 40. #waterflow
    e_f = 45. #forest
    return e_ag*ag + e_wf*wf + e_f*(forest-1.) #+ e_r*rain(t)
    
######################################################################
### The Society
######################################################################
    
###*******************************************************************

def benefit_cost(ag_In):
    ### Benefit cost assessment
    max_yield = 1100
    origin_shift = 1.11
    slope_yield = -0.0052
    return (max_yield*(1-origin_shift*np.exp(slope_yield*ag_In)))

###*******************************************************************
 
class Settlements:
    ### Initialize settlements
    def __init__(self,number_settlements_In):
        self.number_settlements = number_settlements_In
        # distribute specified number of settlements on the map
        self.settlement_positions = land_patches[:,np.random.choice(
                len(land_patches[1]),number_settlements_In).astype('int')]
                
        self.age = np.zeros((number_settlements_In))
        
        # demographic variables
        self.birth_rate =  np.empty(number_settlements_In)
        self.birth_rate.fill(0.15)
        self.death_rate =  0.1 + 0.05 * np.random.random(number_settlements_In)
        self.population =  np.random.randint(1000,2000,number_settlements_In)
        self.mig_rate =  np.zeros((number_settlements_In))
        self.out_mig = np.zeros((number_settlements_In)).astype('int')
        self.pioneer_set = []
        self.failed = 0
        
        # agricultural influence
        self.number_cells_in_influence = np.zeros((number_settlements_In))
        self.area_of_influence = np.zeros((number_settlements_In))
        self.coordinates = np.indices((rows,columns))
        self.cells_in_influence = [None]*number_settlements_In # will be a list of arrays 

        self.cropped_cells = [None]*number_settlements_In
        for city in xrange(number_settlements_In):
            self.cropped_cells[city] = np.array([[self.settlement_positions[0,city]],[self.settlement_positions[1,city]]])
            

        self.occupied_cells = np.zeros((rows,columns))
        self.number_cropped_cells = np.zeros((number_settlements_In))
        self.crop_yield = np.zeros((number_settlements_In))
        self.eco_benefit = np.zeros((number_settlements_In))
        self.available = 0
        
        # trade variables
        self.rank = np.zeros((number_settlements_In))
        self.adjacency = np.zeros((number_settlements_In,number_settlements_In))
        self.degree = np.zeros((number_settlements_In))
        self.comp_size = np.zeros((number_settlements_In))
        self.centrality = np.zeros((number_settlements_In))
        self.trade_strength = np.zeros((number_settlements_In))
        
        # total real income per capita
        self.real_income_pc = np.zeros((number_settlements_In))
  
    def get_cells_in_influence(self):
        self.area_of_influence = (self.population**0.8)/60.
        for city in np.where(self.population!=0)[0]:
            
            stencil = (area*(
            (self.settlement_positions[0][city] - self.coordinates[0])**2 +
            (self.settlement_positions[1][city] - self.coordinates[1])**2)
            )    <= self.area_of_influence[city]**2
            
            self.cells_in_influence[city] = self.coordinates[:,stencil]
        self.number_cells_in_influence = np.array([len(x[0]) for x in self.cells_in_influence])
        for city in np.where(self.population==0)[0]:
            self.cells_in_influence[city] = np.array([[],[]],dtype='int')
        self.number_cells_in_influence[self.population==0] = 0
        
        return self.cells_in_influence, self.number_cells_in_influence 
    
    def get_cropped_cells(self,bca):
        estab_cost = 900
        ag_travel_cost = 950
        abandoned = 0
        sown = 0
        # for each settlement: how many cells are currently cropped ?
        self.number_cropped_cells = np.array([len(x[0]) for x in self.cropped_cells])
        
        # agricultural population density determines the number of cells that can be cropped
        ag_pop_density = self.population/(self.number_cropped_cells * area)
        occup = np.concatenate(self.cropped_cells,axis=1)
        for index in xrange(len(occup[0])):
            self.occupied_cells[occup[0,index],occup[1,index]] = 1
        
        self.age += 1
        # for each settlement: which cells to crop ?
        for city in np.where(self.population!=0)[0]:
            distances = np.sqrt(area*(
                (self.settlement_positions[0][city] - self.coordinates[0])**2 +
                (self.settlement_positions[1][city] - self.coordinates[1])**2))
            utility = bca - estab_cost - (ag_travel_cost * distances)/np.sqrt(self.population[city])
            # 1.) abandon cells if population too low after cities age > 5 years
            if (ag_pop_density[city] < 40 and self.age[city] > 5):
                for number_lost_cells in xrange(np.ceil(30/ag_pop_density[city]).astype('int')):
                    # give up cell with lowest utility
                    cells = self.cropped_cells[city]               
                    abandon_index = np.argmin(utility[cells[0],cells[1]])
                    coord = self.cropped_cells[city][:,abandon_index]
                    self.cropped_cells[city] = np.delete(self.cropped_cells[city],abandon_index,1)
                    self.occupied_cells[coord[0],coord[1]] = 0
                    abandoned += 1

            # 2.) include new cells if population exceeds a threshold 
            for number_new_cells in xrange(np.floor(ag_pop_density[city]/125).astype('int')):   
                # choose uncropped cell with maximal utility
                influence = np.zeros((rows,columns))
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
                abandon_ind = np.where(ut_negative)
                coor = self.cropped_cells[city][:,abandon_ind]
                self.cropped_cells[city] = np.delete(self.cropped_cells[city],abandon_ind,1)
                self.occupied_cells[coor[0],coor[1]] = 0
                abandoned += len(abandon_ind)

                
        for city in np.where(self.population==0)[0]:
            self.cropped_cells[city] = np.array([[],[]],dtype='int')
        self.number_cropped_cells[self.population==0] = 0
        
        self.number_cropped_cells = np.array([len(x[0]) for x in self.cropped_cells])
        self.population[self.number_cropped_cells==0] = 0 
        return self.age, self.cropped_cells, self.number_cropped_cells, abandoned, sown, self.occupied_cells

    def get_crop_yield(self,bca):
        # agricultural benefit of cropping
        for city in np.where(self.population!=0)[0]:
            crops = bca[self.cropped_cells[city][0],self.cropped_cells[city][1]]
            self.crop_yield[city] = np.nanmean(crops[crops>0])
            
        self.crop_yield[np.isnan(self.crop_yield)] = 0
        self.crop_yield[self.population==0] = 0
        return self.crop_yield
        
    def get_eco_benefit(self,es):
        # benefit from ecosystem services of cells in influence
        for city in np.where(self.population!=0)[0]:
            self.eco_benefit[city] = np.nanmean(es[self.cells_in_influence[city]])
        self.eco_benefit[self.population==0] = 0
        return self.eco_benefit
            
    def get_pop_mig(self):
        # gives population and out-migration
        print "number of settlements", self.number_settlements
        
        # death rate correlates inversely with real income per capita
        min_death_rate = 0.005
        max_death_rate = 0.25
        death_rate_diffe = max_death_rate - min_death_rate
        
        self.death_rate = -death_rate_diffe * self.real_income_pc + max_death_rate
        self.death_rate[self.death_rate<min_death_rate] = min_death_rate
        self.death_rate[self.death_rate>max_death_rate] = max_death_rate
        
        # population control
        popcontrol = True
        if popcontrol == True:
            max_birth_rate =  0.15 
            min_birth_rate = -0.2
            shift = 0.325
            self.birth_rate[self.population>=5000] = -(max_birth_rate - min_birth_rate)/10000. * self.population[self.population>=5000] + shift
            
        self.population += (self.birth_rate - self.death_rate)*self.population
        self.failed += np.sum(self.population<=0)
        self.population[self.population<=0] = 0
        
        ### TODO: connect with other model functions
        estab_cost = 900
        self.population[self.population<estab_cost*0.4] = 0
        min_mig_rate = 0.
        max_mig_rate = 0.15
        mig_rate_diffe = max_mig_rate - min_mig_rate
        
        # outmigration rate also correlates inversely with real income per capita
        self.mig_rate = -mig_rate_diffe * self.real_income_pc + max_mig_rate
        self.mig_rate[self.mig_rate<min_mig_rate] = min_mig_rate
        self.mig_rate[self.mig_rate>max_mig_rate] = max_mig_rate
        self.out_mig = (self.mig_rate * self.population).astype('int')
        self.out_mig[self.out_mig<0] = 0

        return self.population, self.out_mig, self.death_rate
        
    ### impact of sociosphere on ecosphere
    def get_pop_gradient(self):
        # pop gradient quantifies the disturbance of the forest by population
        pop_gradient = np.zeros((rows,columns))
        for city in np.where(self.population!=0)[0]:
            distance = np.sqrt(area*(
                (self.settlement_positions[0][city] - self.coordinates[0])**2 +
                (self.settlement_positions[1][city] - self.coordinates[1])**2))
                
            pop_gradient[self.cells_in_influence[city][0],self.cells_in_influence[city][1]] += self.population[city]/(300*(1+distance[self.cells_in_influence[city][0],self.cells_in_influence[city][1]]))
            pop_gradient[pop_gradient>15] = 15
         
        return pop_gradient
        
    def evolve_soil_deg(self,soil_deg,forest):
        ### soil degrades for cropped cells
        deg_rate = 5.0
        reg_rate = 2.5
        cropped = np.concatenate(self.cropped_cells,axis=1)
        soil_deg[cropped[0],cropped[1]] += deg_rate
        soil_deg[forest==3] -= reg_rate
        soil_deg[soil_deg<0] = 0
        return soil_deg
        
    ###----------------------------------------------------------
    ### functions for trading
    
    def get_rank(self):
        ### depending on population ranks are assigned
        ### attention: ranks are reverted with respect to Netlogo MayaSim !
        ### 1 => 3 ; 2 => 2 ; 3 => 1 
        self.rank = np.zeros(self.number_settlements)
        tresh_1 = 4000.
        tresh_2 = 7000.
        tresh_3 = 9500.
        self.rank[self.population > tresh_1] = 1
        self.rank[self.population > tresh_2] = 2
        self.rank[self.population > tresh_3] = 3
        return self.rank
        
    def build_routes(self):
        ### cities with rank>0 are traders and establish links to neighbours
        for city in np.where(self.population!=0)[0]:
            if (self.rank[city] != 0 and self.degree[city] <= self.rank[city]):
                
                distances = (np.sqrt(area*((self.settlement_positions[0][city] - self.settlement_positions[0])**2 +
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
                    
        cut_links = self.population!=0
        cut_links = np.expand_dims(cut_links,0)
        self.adjacency *= cut_links*cut_links.T
        return self.adjacency, cut_links
        
    def get_comps(self):
        ### find components of trade network and their size
        a = self.adjacency
        N = self.number_settlements
        site = np.zeros(N)
        this_comp = np.zeros(N)
        visited = np.zeros(N)
        self.comp_size = np.zeros(N) 
        # nodes with degree zero
        degree = np.sum(a,axis=0)
        visited[degree==0] = np.nan
        
        label = 0
        while np.any(visited == 0):
            label += 1 # new label for new component
            this_comp = np.zeros(N)
            site = np.zeros(N)
            start_index = np.where(visited==0)[0][0] #start at unvisited site
            site[start_index] = 1
            this_comp += site
            while (np.any(this_comp > 0)): # hop to neighbours while new ones pop up
                this_comp[this_comp>0] = np.nan            
                site = a.dot(site)
                this_comp += site
            self.comp_size[np.isnan(this_comp)] = sum(np.isnan(this_comp))
            visited += this_comp # finally, label this component as visited
        self.degree = np.sum(self.adjacency,axis=0)
        return self.degree, self.comp_size
        
    def get_centrality(self):
        ### here, centrality means lenght of longest shortest path within component 
        a = self.adjacency
        N = self.number_settlements
        self.centrality = np.zeros(N)
        for city in xrange(N):
            site = np.zeros(N)
            site[city] = 1
            visited = np.zeros(N)
            visited += site   
            while (np.any(visited > 0)): # hop to neighbours while new ones pop up
                self.centrality[city] += 1
                visited[visited>0] = np.nan           
                site = a.dot(site)
                visited += site
        return self.centrality
        
    def get_trade_strength(self):
        self.trade_strength = 1./30.*( 1 + self.comp_size/self.centrality )**0.9
        self.trade_strength[self.trade_strength>1] = 1
        self.trade_strength[self.trade_strength<0] = 0
        self.trade_strength[self.degree==0] = 0
        return self.trade_strength
    
    def get_real_income_pc(self):
        ### combine agricultural, ecosystem service and trade benefit
        r_bca = 1.1
        r_es = 10. 
        r_trade = 6000.
        
        self.real_income_pc = r_bca * self.crop_yield + r_es * self.eco_benefit + r_trade * self.trade_strength
        self.real_income_pc = self.real_income_pc / self.population
        return self.real_income_pc
       
    def migration(self,es):
        ### if outmigration rate exceeds threshold, found new settlement
        mig_TC_pref = -0.1
        mig_ES_pref =  0.3
        vacant_lands = np.isfinite(es)
        influenced_cells = np.concatenate(cells_in_influence,axis=1)
        vacant_lands[influenced_cells[0],influenced_cells[1]] = 0
        vacant_lands = np.asarray(np.where(vacant_lands == 1))
        for city in np.where(self.population!=0)[0]:
            if (self.out_mig[city] > 400 and np.random.rand() <= 0.5 and len(vacant_lands[0])>=75):
                
                mig_pop = self.out_mig[city]
                self.population[city] -= mig_pop
                self.pioneer_set = vacant_lands[:,np.random.choice(len(vacant_lands[0]),75)]
                travel_cost =  np.sqrt(area*(
                    (self.settlement_positions[0][city] - self.coordinates[0])**2 +
                    (self.settlement_positions[1][city] - self.coordinates[1])**2))
                utility = mig_ES_pref * es + mig_TC_pref * travel_cost 
                utofpio = utility[self.pioneer_set[0],self.pioneer_set[1]]
                new_loc = self.pioneer_set[:,np.nanargmax(utofpio)]
                neighbours = (np.sqrt(area*((new_loc[0] - self.settlement_positions[0][self.population>0])**2 +
                    (new_loc[1] - self.settlement_positions[1][self.population>0])**2))) <= 7.5
                summe = np.sum(neighbours)
                
                if summe==0:
                    self.newcity(new_loc[0],new_loc[1],mig_pop)
                    index = (vacant_lands[0,:]==new_loc[0])&(vacant_lands[1,:]==new_loc[1])
                    np.delete(vacant_lands,index,1)
        
        
    def newcity(self,a,b,mig_pop):
        ### extend all variables to include new city
        self.number_settlements += 1
        self.settlement_positions = np.append(self.settlement_positions,[[a],[b]],1)
        
        self.age = np.append(self.age,0)
        
        self.birth_rate = np.append(self.birth_rate,0.15)
        self.death_rate = np.append(self.death_rate,0.1 + 0.05 * np.random.rand())
        self.population = np.append(self.population,mig_pop)
        self.mig_rate = np.append(self.mig_rate,0)
        self.out_mig = np.append(self.out_mig,0)
        
        self.number_cells_in_influence = np.append(self.number_cells_in_influence,0)
        self.area_of_influence = np.append(self.area_of_influence,0)
        self.cells_in_influence.append(np.array(([a],[b])))
        
        self.cropped_cells.append(np.array([[a],[b]]))

        self.number_cropped_cells = np.append(self.number_cropped_cells,1)
        self.crop_yield = np.append(self.crop_yield,0)
        self.eco_benefit = np.append(self.eco_benefit,0)
    
        self.rank = np.append(self.rank,0)
        N = len(self.adjacency)
        self.adjacency = np.append(self.adjacency,[[0]*N],0)
        self.adjacency = np.append(self.adjacency,[[0]]*(N+1),1)
        self.degree = np.append(self.degree,0)
        self.comp_size = np.append(self.comp_size,0)
        self.centrality = np.append(self.centrality,0)
        self.trade_strength = np.append(self.trade_strength,0)
        
        self.real_income_pc = np.append(self.real_income_pc,0)
         
###*******************************************************************
### run
###*******************************************************************

#### initialize objects of the classes
flowmodel = Waterflow()
forestmodel = Forest()

societymodel = Settlements(30)



#### initialize variables
rain = np.zeros((rows,columns)) 
npp = np.zeros((rows,columns))
forest = np.zeros((rows,columns))
cleared_land_neighbours = np.zeros((rows,columns))
wf = np.zeros((rows,columns))
ag = np.zeros((rows,columns))
es = np.zeros((rows,columns))
bca = np.zeros((rows,columns))
pop_gradient = np.zeros((rows,columns))

### save 
comment = "birth_control"
now = datetime.datetime.now()
location = now.strftime("%d_%m_%H-%M-%Ss")+"_Output_"+comment
os.mkdir(location)
location += "/"


timesteps = 325
print "timeloop starts now"
for step in xrange(timesteps):
    t =  step+1
    print "time = ", t
    
    ### evolve submodels
    # ecosystem
    rain = rain_function(t,cleared_land_neighbours)
    npp = net_primary_prod(t,rain)
    forest, cleared_land_neighbours = forestmodel.forest_evolve(npp,pop_gradient)
    wf = flowmodel.get_waterflow(rain)[1]
    ag = get_agri(npp,wf)
    es = get_ecoserv(ag,wf,forest)
    bca = benefit_cost(ag)
    
    # society
    cells_in_influence, number_cells_in_influence = societymodel.get_cells_in_influence()
    age, cropped_cells, number_cropped_cells, abandoned, sown, occupied = societymodel.get_cropped_cells(bca)
    crop_yield = societymodel.get_crop_yield(bca)
    eco_benefit = societymodel.get_eco_benefit(es)
    population, out_mig, death_rate = societymodel.get_pop_mig()
    soil_deg = societymodel.evolve_soil_deg(soil_deg,forest)
    pop_gradient = societymodel.get_pop_gradient()
    rank = societymodel.get_rank()
    adjacency, cl = societymodel.build_routes()
    degree, comp_size = societymodel.get_comps()
    centrality = societymodel.get_centrality()
    trade_strength = societymodel.get_trade_strength()
    real_income_pc = societymodel.get_real_income_pc()
    number_settlements = societymodel.number_settlements
    settlement_positions = societymodel.settlement_positions
    societymodel.migration(es)

    ### save variables of interest      
    np.save(location+"rain_%d.npy"%(t,),rain)
    np.save(location+"npp_%d.npy"%(t,),npp)
    np.save(location+"forest_%d.npy"%(t,),forest)
    np.save(location+"waterflow_%d.npy"%(t,),wf)
    np.save(location+"AG_%d.npy"%(t,),ag)
    np.save(location+"ES_%d.npy"%(t,),es)
    np.save(location+"bca_%d.npy"%(t,),bca)
    
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
    
    save_stacked_array(location+"cells_in_influence_%d"%(t,),cells_in_influence,axis=1)
    np.save(location+"number_cells_in_influence_%d.npy"%(t,),number_cells_in_influence)
    save_stacked_array(location+"cropped_cells_%d"%(t,),cropped_cells,axis=1)
    np.save(location+"number_cropped_cells_%d.npy"%(t,),number_cropped_cells)
    np.save(location+"abnd_sown_%d.npy"%(t,),np.array((abandoned,sown)))
    np.save(location+"crop_yield_%d.npy"%(t,),crop_yield)
    np.save(location+"eco_benefit_pc_%d.npy"%(t,),eco_benefit)
    np.save(location+"real_income_pc_%d.npy"%(t,),real_income_pc)
    np.save(location+"population_%d.npy"%(t,),population)
    np.save(location+"out_mig_%d.npy"%(t,),out_mig)
    np.save(location+"death_rate_%d.npy"%(t,),death_rate)
    np.save(location+"soil_deg_%d.npy"%(t,),soil_deg)
    np.save(location+"pop_gradient_%d.npy"%(t,),pop_gradient)
    np.save(location+"adjacency_%d.npy"%(t,),adjacency)
    np.save(location+"degree_%d.npy"%(t,),degree)
    np.save(location+"comp_size_%d.npy"%(t,),comp_size)
    np.save(location+"centrality_%d.npy"%(t,),centrality)
    np.save(location+"trade_strength_%d.npy"%(t,),trade_strength)
    
    np.save(location+"number_settlements_%d.npy"%(t,),number_settlements)
    np.save(location+"settlement_positions_%d.npy"%(t,),settlement_positions) 

#if __name__ == "__main__":
#    main()
