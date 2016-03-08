import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import matplotlib.gridspec as gridspec

print sys.argv


################################################################################

N = 110

data_directory = 'data/' + sys.argv[1]+ '/'
picture_directory = 'data/' + sys.argv[1] + '_plots/'

if not os.path.exists(picture_directory):
    os.makedirs(picture_directory)

################################################################################

number_settlements = np.load(data_directory+"number_settlements_"+str(N)+".npy")

### initialize variables to track overall evolution
rain_evo = np.zeros((3,N))
npp_evo = np.zeros((3,N))
forest_evo = np.zeros((3,N))
wf_evo = np.zeros((3,N))
ag_evo = np.zeros((3,N))
es_evo = np.zeros((3,N)) 
bca_evo = np.zeros((3,N))
cells_in_influence_evo = np.zeros((N,number_settlements)) 
cropped_cells_evo = np.zeros((N,number_settlements)) 
crop_yield_evo = np.zeros((N,number_settlements)) 
eco_benefit_evo = np.zeros((N,number_settlements))
abnd_sown_evo = np.zeros((N,2))
real_income_pc_evo = np.zeros((N,number_settlements)) 
population_evo = np.zeros((N,number_settlements))
death_rate_evo = np.zeros((N,number_settlements))
out_mig_evo = np.zeros((N,number_settlements))
comp_size_evo = np.zeros((N,number_settlements))
centrality_evo = np.zeros((N,number_settlements))  
trade_income_evo = np.zeros((N,number_settlements)) 
number_tradelinks_evo = soil_deg_evo = np.zeros((N))
degree_evo = np.zeros((N,number_settlements)) 
number_settlements_evo = np.zeros((N))
number_failed_cities_evo = np.zeros((N))
soil_deg_evo = np.zeros((3,N))
ag_pop_density_evo = np.zeros((N,number_settlements)) 

bca = np.load(data_directory+"bca_1.npy")
ocean = np.empty(bca.shape)
ocean[:] = 1
ocean[np.isfinite(bca)]=np.nan


    
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

    
################################################################################

### define functions to save a map of single timestep
def show_rain(t):
    rain = np.load(data_directory+"rain_%d.npy"%(t,))    
    fig = plt.figure()
    plt.imshow(rain)
    plt.colorbar()
    plt.title("rain, t="+str(t))
    fig.savefig(picture_directory+"rain_%d.png"%(t,),dpi=200)
    plt.close(fig)
 
def show_npp(t):
    npp = np.load(data_directory+"npp_%d.npy"%(t,))
    fig = plt.figure()
    plt.imshow(npp)
    plt.colorbar()
    plt.title("npp, t="+str(t))
    fig.savefig(picture_directory+"npp_%d.png"%(t,),dpi=200)
    plt.close(fig)
    
def show_forest(t):
    forest = np.load(data_directory+"forest_%d.npy"%(t,))
    fig = plt.figure()

    cmap = ListedColormap(['white', '#FF9900', '#66FF33','#336600'])
    norm = Normalize(vmin=0,vmax=3)
    im3 = plt.imshow(forest, cmap=cmap, norm=norm, interpolation='none')
    timestep = "%d"%(t,)
    plt.title('forest, t = '+timestep)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(['$1$','$2$','$3$']):
        cbar.ax.text(.5, (2 * j + 3) / 8.0, lab, ha='center', va='center')
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('forest state', rotation=270)
    plt.title("forest, t="+str(t))
    fig.savefig(picture_directory+"forest%d.png"%(t,),dpi=200)
    plt.close(fig)
    
def show_wf(t):
    wf = np.load(data_directory+"waterflow_%d.npy"%(t,))
    fig = plt.figure()
    plt.imshow(wf)
    plt.colorbar()
    plt.title("waterflow, t="+str(t))
    fig.savefig(picture_directory+"waterflow_%d.png"%(t,),dpi=200)
    plt.close(fig)
    
def show_ag(t):
    ag = np.load(data_directory+"AG_%d.npy"%(t,))
    fig = plt.figure()
    plt.imshow(ag,vmin=0)
    plt.colorbar()
    plt.title("agricultural productivity, t="+str(t))
    fig.savefig(picture_directory+"AG_%d.png"%(t,),dpi=200)
    plt.close(fig)
    
def show_es(t):
    es = np.load(data_directory+"ES_%d.npy"%(t,))
    fig = plt.figure()
    plt.imshow(es)
    plt.colorbar()
    plt.title("ecosystem services, t="+str(t))
    fig.savefig(picture_directory+"es_%d.png"%(t,),dpi=200)
    plt.close(fig)
    
def show_bca(t):
    bca = np.load(data_directory+"bca_%d.npy"%(t,))
    fig = plt.figure()
    plt.imshow(bca-900,cmap="YlOrBr",vmin=0,interpolation='None')
    plt.colorbar()
    plt.imshow(np.ma.masked_where(bca>900,bca),cmap=ListedColormap(['#2AF10A']),interpolation='None')
    plt.title("benefit cost assessment, t="+str(t))
    fig.savefig(picture_directory+"bca_%d.png"%(t,),dpi=300,interpolation='None')
    plt.close(fig)
   
def show_soil_deg(t):
    soil_deg = np.load(data_directory+"soil_deg_%d.npy"%(t,))
    fig = plt.figure()
    plt.imshow(soil_deg) #*(np.isfinite(bca))
    plt.colorbar()
    plt.title("soil degradation, t="+str(t))
    fig.savefig(picture_directory+"soil_deg_%d.png"%(t,),dpi=200)
    plt.close(fig)
    
def show_bca_influence_cropped(t):
    bca = np.load(data_directory+"bca_%d.npy"%(t,))
    shape  = bca.shape
    # number_settlements = np.load(data_directory+"number_settlements.npy")
    settlement_positions = np.load(data_directory+"settlement_positions_%d.npy"%(t,))
    population = np.load(data_directory+"population_%d.npy"%(t,))
    cells_in_influence = load_stacked_arrays(data_directory+"cells_in_influence_%d.npz"%(t,),axis=1)
    cropped_cells = load_stacked_arrays(data_directory+"cropped_cells_%d.npz"%(t,),axis=1)
    fig = plt.figure(figsize=(10,10))
    rows, columns = bca.shape
    #plt.imshow(bca,cmap="Blues",interpolation="nearest",extent=[0,columns,rows,0],vmin=0)
    #plt.colorbar()
    influenced = np.zeros(shape)
    cropped = np.zeros(shape)
    cities = np.zeros(shape)
    failed_cities = np.zeros(shape)
    for city in xrange(len(settlement_positions[1])):
        if len(cells_in_influence[city])>0:
            influenced[cells_in_influence[city][0],cells_in_influence[city][1]] = 1
        #plt.scatter(b,a,s=1,marker=',',color='Green')
        if len(cropped_cells[city])>0:
            cropped[cropped_cells[city][0],cropped_cells[city][1]] = 1
        #plt.scatter(d,c,s=1,marker=',',color='Black')
    
    cities[settlement_positions[0],settlement_positions[1]] = 1
    failed_cities[settlement_positions[0][population==0],settlement_positions[1][population==0]] = 1
    cropped = np.ma.masked_where(cropped ==0, cropped)
    influenced =  np.ma.masked_where(influenced ==0, influenced)
    cities =  np.ma.masked_where(cities ==0, cities)
    failed_cities = np.ma.masked_where(failed_cities==0,failed_cities)
    
    cmap1 = ListedColormap(['#2FDE86'])
    cmap2 = ListedColormap(['#992C2C'])
    cmap3 = ListedColormap(['#000000'])
    cmap4 = ListedColormap(['#FF0000'])
    plt.imshow(ocean,interpolation='None')
    plt.imshow(np.ma.masked_where(bca>900,bca),cmap=ListedColormap(['#F10AC8']),interpolation='None')
    plt.imshow(influenced,cmap=cmap1,interpolation='None',alpha=0.5)
    plt.imshow(cropped,cmap=cmap2,interpolation='None')
    plt.imshow(cities,cmap=cmap3,interpolation='None')
    plt.imshow(failed_cities,cmap=cmap4,interpolation='None')
    count =  len(settlement_positions[1])
    plt.title("settlement influence and cropped cells, #cities: "+str(count)+", t="+str(t))    
    fig.savefig(picture_directory+"influence_cropped_b_%d.png"%(t,),dpi=300,interpolation=None)
    plt.close(fig)
    
def show_network(t):
    population = np.load(data_directory+"population_%d.npy"%(t,))
    settlement_positions = np.load(data_directory+"settlement_positions_%d.npy"%(t,))
    adjacency = np.load(data_directory+"adjacency_%d.npy"%(t,))
    centrality = np.load(data_directory+"centrality_%d.npy"%(t,))
    #bca = np.load(data_directory+"bca_%d.npy"%(t,))
    x = settlement_positions[0]+0.5
    y = settlement_positions[1]+0.5
    labels = np.arange(len(settlement_positions[0]))
    fig, ax = plt.subplots(figsize=(5,5))
    plt.imshow(ocean)
    alive = population!=0
    x_alive = x[alive]
    y_alive = y[alive]
    dead = population==0
    x_dead = x[dead]
    y_dead = y[dead]
    plt.xlim(0,bca.shape[0])
    plt.ylim(0,bca.shape[1])
    ax.scatter(y_alive, x_alive,marker='+',s=1)
    ax.scatter(y_dead, x_dead,marker='+',color='r',s=1)
    # ax.scatter(y[(centrality>1) & (centrality<4)],x[(centrality>1) & (centrality<4)],color='g')
    """
    for i,txt in enumerate(labels):
        ax.annotate(txt, (y[i],x[i]),size=3)
    """
    generator = (i for i,x in np.ndenumerate(adjacency) if adjacency[i]==1)
    for i,j in generator:
        plt.plot([y[i],y[j]],[x[i],x[j]], color="k",linewidth=0.5,alpha=0.2)
    
    plt.xlim(0,bca.shape[1])
    plt.ylim(bca.shape[0],0)
    
    plt.title("trade network, t="+str(t))
    fig.savefig(picture_directory+"trade_network_%d.png"%(t,),dpi=300,interpolation=None)
    plt.close(fig)    
    
   
def show_pop_gradient(t):
    pop_gradient = np.load(data_directory+"pop_gradient_%d.npy"%(t,))
    fig = plt.figure()
    plt.imshow(pop_gradient) #*(np.isfinite(bca))
    plt.colorbar()
    plt.title("population gradient, t="+str(t))
    fig.savefig(picture_directory+"pop_grad_%d.png"%(t,),dpi=200)
    plt.close(fig)   

################################################################################

# define functions recording time evolution of certain variables

def write_rain_evo(t):
    step = np.load(data_directory+"rain_%d.npy"%(t,))    
    rain_evo[0][t-1] = np.nanmean(step)
    rain_evo[1][t-1] = np.percentile(step[np.isfinite(step)],90)
    rain_evo[2][t-1] = np.percentile(step[np.isfinite(step)],10)
    
def write_npp_evo(t):
    step = np.load(data_directory+"npp_%d.npy"%(t,))
    npp_evo[0][t-1] = np.nanmean(step)
    npp_evo[1][t-1] = np.percentile(step[np.isfinite(step)],90)
    npp_evo[2][t-1] = np.percentile(step[np.isfinite(step)],10)
    
def write_forest_evo(t):
    step = np.load(data_directory+"forest_%d.npy"%(t,))
    forest_evo[2][t-1] = len(step[step == 3])
    forest_evo[1][t-1] = len(step[step == 2])
    forest_evo[0][t-1] = len(step[step == 1])
    
def write_wf_evo(t):
    step = np.load(data_directory+"waterflow_%d.npy"%(t,))
    wf_evo[0][t-1] = np.nanmean(step)
    wf_evo[1][t-1] = np.percentile(step[np.isfinite(step)],90)
    wf_evo[2][t-1] = np.percentile(step[np.isfinite(step)],10)
    
def write_ag_evo(t):
    step = np.load(data_directory+"AG_%d.npy"%(t,))
    ag_evo[0][t-1] = np.nanmean(step)
    ag_evo[1][t-1] = np.percentile(step[np.isfinite(step)],90)
    ag_evo[2][t-1] = np.percentile(step[np.isfinite(step)],10)
    
def write_es_evo(t):
    step = np.load(data_directory+"ES_%d.npy"%(t,))
    es_evo[0][t-1] = np.nanmean(step)
    es_evo[1][t-1] = np.percentile(step[np.isfinite(step)],90)
    es_evo[2][t-1] = np.percentile(step[np.isfinite(step)],10)
    
def write_bca_evo(t):
    step = np.load(data_directory+"bca_%d.npy"%(t,))
    bca_evo[0][t-1] = np.nanmean(step[step>0]) # include only positive values of bca
    bca_evo[1][t-1] = np.percentile((step[step>0]),90)
    bca_evo[2][t-1] = np.percentile((step[step>0]),10)
    
def write_cells_in_influence_evo(t):
    step = np.load(data_directory+"number_cells_in_influence_%d.npy"%(t,))
    cells_in_influence_evo[t-1,:len(step)] = step
    
def write_cropped_cells_evo(t):
    step = np.load(data_directory+"number_cropped_cells_%d.npy"%(t,))
    cropped_cells_evo[t-1,:len(step)] += step
    
def write_crop_yield_evo(t):
    step = np.load(data_directory+"crop_yield_%d.npy"%(t,))
    crop_yield_evo[t-1,:len(step)] += step
    
def write_eco_benefit_evo(t):
    step = np.load(data_directory+"eco_benefit_pc_%d.npy"%(t,))
    eco_benefit_evo[t-1,:len(step)] += step

def write_abnd_sown_evo(t):
    step = np.load(data_directory+"abnd_sown_%d.npy"%(t,))
    abnd_sown_evo[t-1,:len(step)] = step
    
def write_real_income_pc_evo(t):
    step = np.load(data_directory+"real_income_pc_%d.npy"%(t,))
    real_income_pc_evo[t-1,:len(step)] = step
    
def write_population_evo(t):
    step = np.load(data_directory+"population_%d.npy"%(t,))
    population_evo[t-1,:len(step)] = step
 
def write_death_rate_evo(t):
    step = np.load(data_directory+"death_rate_%d.npy"%(t,))
    death_rate_evo[t-1,:len(step)] = step
    
def write_out_mig_evo(t):
    step = np.load(data_directory+"out_mig_%d.npy"%(t,))
    out_mig_evo[t-1,:len(step)] = step
    
def write_comp_size_evo(t):
    step = np.load(data_directory+"comp_size_%d.npy"%(t,))
    comp_size_evo[t-1,:len(step)] = step

def write_centrality_evo(t):
    step = np.load(data_directory+"centrality_%d.npy"%(t,))
    centrality_evo[t-1,:len(step)] = step
    
def write_trade_income_evo(t):
    step = np.load(data_directory+"trade_income_%d.npy"%(t,))
    trade_income_evo[t-1,:len(step)] = step
     
def write_soil_deg_evo(t):
    step = np.load(data_directory+"soil_deg_%d.npy"%(t,))
    soil_deg_evo[0][t-1] = np.nanmean(step[np.isfinite(bca)])
    soil_deg_evo[1][t-1] = np.percentile(step[np.isfinite(bca)],90)
    soil_deg_evo[2][t-1] = np.percentile(step[np.isfinite(bca)],10)
    
def write_number_tradelinks_evo(t):
    step = np.load(data_directory+"degree_%d.npy"%(t,))
    number_tradelinks_evo[t-1] = sum(step)/2
    
def write_degree_evo(t):
    step = np.load(data_directory+"degree_%d.npy"%(t,))
    degree_evo[t-1,:len(step)] = step
    
def write_number_settlements_evo(t):
    population = np.load(data_directory+"population_%d.npy"%(t,))
    number_settlements_evo[t-1] = np.sum(population!=0)
    number_failed_cities_evo[t-1] = np.sum(population==0)
 
def write_ag_pop_density_evo(t):
    area = 5.10864490603363
    population = np.load(data_directory+"population_%d.npy"%(t,))
    number_cropped_cells = np.load(data_directory+"number_cropped_cells_%d.npy"%(t,))
    a = population/(number_cropped_cells * area)
    ag_pop_density_evo[t-1,:len(population)] = a
################################################################################

### loop over time, saving frames and recording time evolution
for t in xrange(1,N+1):
    print t
    
    show_rain(t) 
    show_npp(t)
    show_forest(t)
    show_wf(t)
    show_ag(t)
    show_es(t)
    
    show_soil_deg(t)
    show_bca(t)
    
    show_bca_influence_cropped(t)
    show_network(t)
    show_pop_gradient(t)
    
    write_rain_evo(t)
    write_npp_evo(t)
    write_forest_evo(t)
    write_wf_evo(t)
    write_ag_evo(t)
    write_es_evo(t)
    write_bca_evo(t)
    write_cells_in_influence_evo(t)
    write_cropped_cells_evo(t)
    write_crop_yield_evo(t)
    write_eco_benefit_evo(t)
    write_abnd_sown_evo(t)
    write_real_income_pc_evo(t)
    write_population_evo(t)
    write_death_rate_evo(t)
    write_out_mig_evo(t)
    write_comp_size_evo(t)
    write_centrality_evo(t)
    write_trade_income_evo(t)
    write_number_tradelinks_evo(t)
    write_degree_evo(t)
    write_number_settlements_evo(t)
    write_soil_deg_evo(t)
    write_ag_pop_density_evo(t)
    
## show last frame:
show_soil_deg(t)
show_bca(t)

show_bca_influence_cropped(t)
show_network(t)
show_pop_gradient(t)
################################################################################

### after looping: plot time evolution
 
def plot_evo_std(variable):
    plt.plot(variable[0])
    plt.fill_between(range(N), variable[1], variable[2], color='grey', alpha='0.5')



#fig = plt.figure()
#plt.plot(range(1,N),population_evo[population_evo>4000],population_evo[population_evo>7500],population_evo[population_evo>9000],'k',alpha=0.1)
#plt.ylim(0,15000)
#plt.title("population evo")
#plt.xlabel("timesteps")
#plt.ylabel("population")
#fig.savefig(picture_directory+'evo_ranks.png',dpi=200)
#plt.close(fig)

###############################################################################
def plot1():
    ### total population, total real income, XXX: crop yield, number of trade links and mean cluster sizes
    fig = plt.figure(figsize=(12,7))
    
    ### total population
    plt.subplot(221)
    plt.plot(np.sum(population_evo,1))
    plt.title("population")
    
    ### total real income
    plt.subplot(222)
    plt.plot(np.nansum(real_income_pc_evo*population_evo,1))
    plt.title("real income")
    
    ### XXX: crop yield
    plt.subplot(223)
    plt.plot(crop_yield_evo,'k',alpha=0.2)
    plt.title("crop yield")
    
    ### number of trade links and mean cluster sizes
    host=plt.subplot(224)
    no2 = host.twinx()
    host.plot(np.sum(comp_size_evo,1)/np.sum(comp_size_evo>0,1),'g')
    no2.plot(number_tradelinks_evo,'k')
    plt.title("total number trade (black) links and mean cluster size (green)")
    
    plt.tight_layout()
    fig.savefig(picture_directory+'1_panel.png',dpi=200)
    plt.close(fig)

def plot2():
    ### number of settlements,cropping: abandoned/sown, soil degradation, forest state
    fig = plt.figure(figsize=(12,7))
    
    ### number of settlements
    plt.subplot(221)
    plt.plot(number_settlements_evo)
    plt.title("number of settlements")
    
    ### cropping: abandoned/sown
    plt.subplot(222)
    plt.plot(abnd_sown_evo[:,0],"r")
    plt.plot(abnd_sown_evo[:,1],"g")
    plt.title("abandoned/sown")
    
    ### soil degradation
    plt.subplot(223)
    plt.plot(soil_deg_evo[0])
    plt.title("soil degradation")
    
    ### forest state
    plt.subplot(224)
    plt.stackplot(np.arange(N),forest_evo[0],forest_evo[1],forest_evo[2],colors=['#FF9900', '#66FF33','#336600'])
    plt.title("forest state")
    
    plt.tight_layout()
    fig.savefig(picture_directory+'2_panel.png',dpi=200)
    plt.close(fig)

def plot3():
### number of settlements, cropping: abandoned/sown, soil degradation, total real income
    fig = plt.figure(figsize=(12,7))
    crop_yield_evo[np.isnan(crop_yield_evo)]=0
    
    ### number of settlements
    plt.subplot(221)
    plt.plot(crop_yield_evo)
    plt.title("crop yield")
    
    ### ecoserv benefit 
    plt.subplot(222)
    plt.plot(eco_benefit_evo)
    plt.title("eco benefit")
    
    ### soil degradation
    plt.subplot(223)
    plt.plot(trade_income_evo)
    plt.title("trade strength")
    
    ### total real income
    plt.subplot(224)
    plt.stackplot(np.arange(N),np.nanmean(crop_yield_evo,1)*1.1,np.nanmean(eco_benefit_evo,1)*10,np.nanmean(trade_income_evo,1)*6000)
    plt.title("total real income")
    
    plt.tight_layout()
    fig.savefig(picture_directory+'3_panel.png',dpi=200)
    plt.close(fig)

def plot_ecosystem():    
    ### 
    fig = plt.figure(figsize=(12,7))
    
    ### npp
    plt.subplot(231)
    plot_evo_std(npp_evo)
    plt.title("npp")
    
    ### waterflow
    plt.subplot(232)
    plot_evo_std(wf_evo)
    plt.title("waterflow")
    
    ### forest evolution
    plt.subplot(233)
    plt.stackplot(np.arange(N),forest_evo[0],forest_evo[1],forest_evo[2],colors=['#FF9900', '#66FF33','#336600'])
    plt.title('forest state')
    
    ### soil degradation
    plt.subplot(234)
    plot_evo_std(soil_deg_evo)
    plt.title("soil degradation")
    
    ###  agricultural productivity
    plt.subplot(235)
    plot_evo_std(ag_evo)
    plt.title("agricultural productivity")
        
    ### ecosystem services
    plt.subplot(236)
    plot_evo_std(es_evo)
    plt.title("ecosystem services")
    
    plt.tight_layout()
    fig.savefig(picture_directory+'4_ecosystem.png',dpi=200)
    plt.close(fig)
    
def plot_agriculture():
    # abandoned/sown
    # cropped_cells
    # agri benefit
    fig = plt.figure(figsize=(12,7))
        
    ### abandoned/sown
    plt.subplot(221)
    plt.plot(abnd_sown_evo[:,0],"r")
    plt.plot(abnd_sown_evo[:,1],"g")
    plt.title("abandoned / sown")
    
    ### cropped_cells
    plt.subplot(222)
#    number_cropped_cells_evo = np.zeros((3,N))
#    number_cropped_cells_evo[0,:] = np.nanmean(cropped_cells_evo,axis=1)
#    number_cropped_cells_evo[1,:] = np.std(cropped_cells_evo)
#    number_cropped_cells_evo[2,:] = np.std(cropped_cells_evo)
#    plot_evo_std(number_cropped_cells_evo)
    plt.plot(cropped_cells_evo,'k.',alpha=0.2)
    plt.title("number of cropped cells")

    
    ### crop yield evo
    plt.subplot(223)
    plt.plot(crop_yield_evo,'k.',alpha=0.2)
    plt.title("crop yield")
    
    ### 
    plt.subplot(224)
    plt.title("?")
    
    plt.tight_layout()
    fig.savefig(picture_directory+'5_agriculture.png',dpi=200)
    plt.close(fig)
    
def plot_trade():
    fig = plt.figure(figsize=(12,7))

    # trade links
    plt.subplot(221)
    plt.plot(number_tradelinks_evo)
    plt.title("number of tradelinks")
    
    # centrality
    plt.subplot(222)
    plt.plot(centrality_evo,'k.',alpha=0.2)
    plt.title("centrality of traders")
    
    # comp sizes
    plt.subplot(223)
    plt.plot(comp_size_evo,'k.',alpha=0.2)
    plt.title("component size")
    
    # trade strength
    plt.subplot(224)
    plt.plot(trade_income_evo,color='#F3456E',linestyle='',marker='.',alpha=0.2)
    plt.title("trade strength")
    
    plt.tight_layout()
    fig.savefig(picture_directory+'6_trade.png',dpi=200)
    plt.close(fig)



# ecoserv benefit

def plot_income():
    ### real income per capita evolution
    fig = plt.figure(figsize=(12,7))
    plt.subplot(221)
    plt.plot(crop_yield_evo*1.1,color='#A57A00',linestyle='',marker='.',alpha=0.2)
    plt.title("crop yield")
    
    plt.subplot(222)
    plt.plot(eco_benefit_evo*10.,color='#0EB800',linestyle='',marker='.',alpha=0.2)
    plt.title("ecosystem benefit")
    
    plt.subplot(223)
    plt.plot(trade_income_evo*6000.,color='#F3456E',linestyle='',marker='.',alpha=0.2)
    plt.title("trade strength")
    
    plt.subplot(224)
    plt.plot(crop_yield_evo*1.1+eco_benefit_evo*10.+trade_income_evo*6000.,'k.',alpha=0.2)
    plt.title("total real income")
    
    fig.savefig(picture_directory+'7_evo_real_income.png',dpi=200)
    plt.close(fig)

def plot_population(figsize=(12,7)):
    ### population evolution
    fig = plt.figure()
    plt.subplot(211)
    plt.plot(population_evo,'k',alpha=0.1)
    plt.ylim(0,15000)
    plt.title("population evo")
    plt.ylabel("population of settlements")
    
    plt.subplot(212)
    plt.plot(np.sum(population_evo,1))
    plt.xlabel("timesteps")
    plt.ylabel("total population")
    fig.savefig(picture_directory+'8_population.png',dpi=200)
    plt.close(fig)
    
def plot_migration():
    # number of settlements and failed cities
    fig = plt.figure(figsize=(12,7))
    plt.plot(number_settlements_evo)
    plt.plot(number_failed_cities_evo,'r')
    plt.title("number of settlements/failed cities")
    fig.savefig(picture_directory+'9_migration.png',dpi=200)
    plt.close(fig)
    
plot1()
plot2()
plot3()
plot_ecosystem()
plot_agriculture()
plot_trade()
plot_income()
plot_population()
plot_migration()
