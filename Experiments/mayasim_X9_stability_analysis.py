"""
Experiment to test the influence of drought events.
Drought events start once the civilisation has reached
a 'complex society' state (after 200 years) and vary 
in length and severity from 0 to 100 years and 0 to 100% 
less precipitation.

Therefore, starting point is at t = 200 where the model has
reached a complex society state in all previous studies.
We also use parameters for income from trade, agriculture and
ecosystem services, that have previously proven to lead to
some influence of precipitation variability on the state of the
system.

Also, we vary the parameter for income from trade so see, if
there is a certain parameter value, that results in 
stable complex society for some drought events and collapse for others.
"""
import getpass
import itertools as it
import sys
import pandas as pd
from pathlib import Path
import numpy as np

from pymofa.experiment_handling import experiment_handling as eh
from mayasim.model.ModelCore import ModelCore as Model


def run_function(d_start=200, d_length=20, d_severity=50.,
                 r_bca=0.2, r_es=0.0002, r_trade=6000,
                 population_control=False,
                 n=30, crop_income_mode='sum',
                 better_ess=True,
                 kill_cropless=False, steps=500,
                 settlement_output_path=0,
                 test=False):
    """
    Set up the Model for different Parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values and Parameters, as well as the time
    development of aggregated variables for each run.

    Parameters:
    -----------
    d_start : int
        starting point of drought in model time
    d_length : int
        length of drought in timesteps
    d_severity : float
        severity of drought (decrease in rainfall in percent)
    r_bca : float > 0
        the prefactor for income from agriculture
    r_es : float
        the prefactor for income from ecosystem services
    r_trade : float
        the prefactor for income from trade
    population_control : boolean
        determines whether the population grows
        unbounded or if population growth decreases
        with income per capita and population density.
    n : int > 0
        initial number of settlements on the map
    crop_income_mode : string
        defines the mode of crop income calculation.
        possible values are 'sum' and 'mean'
    better_ess : bool
        switch to use forest as proxy for income from eco
        system services from net primary productivity.
    kill_cropless: bool
        Switch to determine whether or not to kill cities
        without cropped cells.
    filename: string
        path to save the results to.
    """

    # initialize the Model

    m = Model(n=n, output_data_location=settlement_output_path, debug=test)
    if settlement_output_path == 0:
        m.output_geographic_data = False
        m.output_settlement_data = False

    m.population_control = population_control
    m.crop_income_mode = crop_income_mode
    m.better_ess = better_ess
    m.r_bca_sum = r_bca
    m.r_es_sum = r_es
    m.r_trade = r_trade
    m.kill_cities_without_crops = kill_cropless

    m.precipitation_modulation = False
    m.drought_times = [[d_start, d_start + d_length]]
    m.drought_severity = d_severity

    # run Model

    if test:
        m.run(3)
    else:
        m.run(steps)

    # Retrieve results.

    # This is a pandas dataframe

    micro_output = m.get_trajectory()
    micro_output.index.name = 'tsteps'

    data = {'final_population': [sum(m.population)],
            'final_trade_links': [sum(m.degree) / 2.],
            'final_max_cluster_size': [m.max_cluster_size]}

    macro_output = pd.DataFrame(data)
    macro_output.index.name = 'ind'

    # and save them to the path indicated by 'filename'

    return 1, [micro_output, macro_output]


def run_experiment(argv):
    """
    Take arv input variables and run sub_experiment accordingly.
    This happens in five steps:
    1)  parse input arguments to set switches
        for [test],
    2)  set output folders according to switches,
    3)  generate parameter combinations,
    4)  define names and dictionaries of callables to apply to sub_experiment
        data for post processing,
    5)  run computation and/or post processing and/or plotting
        depending on execution on cluster or locally or depending on
        experimentation mode.

    Parameters
    ----------
    argv: list[N]
        List of parameters from terminal input

    Returns
    -------
    rt: int
        some return value to show whether sub_experiment succeeded
        return 1 if sucessfull.
    """

    # Parse switches from input
    if len(argv) > 1:
        test = int(argv[1])
    else:
        test = True
    if len(argv) > 2:
        mode = int(argv[2])
    else:
        mode = None
    if len(argv) > 3:
        job_id = int(argv[3])
    else:
        job_id = 1
    if len(argv) > 4:
        max_id = int(argv[4])
    else:
        max_id = 1

    # Generate paths according to switches and user name

    test_folder = ['', 'test_output/'][int(test)]
    experiment_folder = 'X9_stability_analysis/'
    raw = 'raw_data/'
    res = 'results/'

    if getpass.getuser() == "kolb":
        save_path_raw = f"/p/tmp/kolb/Mayasim/output_data/{test_folder}{experiment_folder}{raw}"
        save_path_res = f"/home/kolb/Mayasim/output_data/{test_folder}{experiment_folder}{res}"
    elif getpass.getuser() == "jakob":
        save_path_raw = f"/home/jakob/Project_MayaSim/Python/output_data/{test_folder}{experiment_folder}{raw}"
        save_path_res = f"/home/jakob/Project_MayaSim/Python/output_data/{test_folder}{experiment_folder}{res}"
    else:
        save_path_res = f'./{res}'
        save_path_raw = f'./{raw}'

    # Generate parameter combinations and set up 'index' dictionary,
    # indicating their possition in the Index of the postprocessed results.

    d_start = [200]
    r_bca, r_es = [0.2], [0.0002]
    population_control = [False]
    n, crop_income_mode, better_ess, kill_cropless, steps = [30], ['sum'], [True], [False], [500]
    output_path = [0]

    if test == 0:
        d_length = list(range(0, 105, 5))
        d_severity = list(range(0, 105, 5))
        r_trade = list(range(4000, 11000, 2000))
        test = [False]
    else:
        d_length = [20]
        d_severity = [0., 60.]
        r_trade = [6000, 8000]
        test = [True]

    # Order of the parameters in the resulting tuples have to match the one indicated in
    # interface of run func, since they will be put in as *param_combs[index]
    param_combs = list(it.product(d_start, d_length, d_severity, r_bca, r_es,
                                  r_trade, population_control, n, crop_income_mode,
                                  better_ess, kill_cropless, steps, output_path, test))

    print(f'computing results for {len(param_combs)} parameter combinations')

    # In this experiment, I use the job_id variable from an array job to split the
    # parameter combinations into equally sized junks.
    # This makes it easier for the queing algorithm to allocate its resources (which means,
    # it will give you more of it ;)
    # this also means, that the total number of jobs you run, must be a divider of the
    # number of parameter combinations that you run.
    if len(param_combs) % max_id != 0:
        print(f'number of jobs ({len(param_combs)}) has to be multiple of max_id ({max_id})!!')
        exit(-1)

    sample_size = 20 if not test else 2

    # Define names and callables for post processing

    def mean(d_start, d_length, d_severity,
             r_bca, r_es, r_trade,
             population_control,
             n, crop_income_mode,
             better_ess, kill_cropless, steps,
             settlement_output_path, test):

        from pymofa.safehdfstore import SafeHDFStore

        print(settlement_output_path, type(settlement_output_path))

        query = f'd_start={d_start} & d_length={d_length} & d_severity={d_severity} ' \
                f'& r_bca={r_bca} & r_es={r_es} & r_trade={r_trade} ' \
                f'& population_control={population_control} & n={n} ' \
                f'& crop_income_mode={crop_income_mode} & better_ess={better_ess} ' \
                f'& kill_cropless={kill_cropless} & steps={steps} ' \
                f'& settlement_output_path={settlement_output_path} & test={test}'

        with SafeHDFStore(compute_handle.path_raw) as store:
            trj = store.select("dat_0", where=query)

        df_out = trj.groupby(level='tsteps').mean()

        return 1, df_out

    def sem(d_start, d_length, d_severity,
             r_bca, r_es, r_trade,
             population_control,
             n, crop_income_mode,
             better_ess, kill_cropless, steps,
             settlement_output_path, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = f'd_start={d_start} & d_length={d_length} & d_severity={d_severity} ' \
                f'& r_bca={r_bca} & r_es={r_es} & r_trade={r_trade} ' \
                f'& population_control={population_control} & n={n} ' \
                f'& crop_income_mode={crop_income_mode} & better_ess={better_ess} ' \
                f'& kill_cropless={kill_cropless} & steps={steps} ' \
                f'& settlement_output_path={settlement_output_path} & test={test}'

        with SafeHDFStore(compute_handle.path_raw) as store:
            trj = store.select("dat_0", where=query)

        df_out = trj.groupby(level='tsteps').sem()

        return 1, df_out

    def collect_final_states(d_start, d_length, d_severity,
                             r_bca, r_es, r_trade,
                             population_control,
                             n, crop_income_mode,
                             better_ess, kill_cropless, steps,
                             settlement_output_path, test):

        from pymofa.safehdfstore import SafeHDFStore

        query = f'd_start={d_start} & d_length={d_length} & d_severity={d_severity} ' \
                f'& r_bca={r_bca} & r_es={r_es} & r_trade={r_trade} ' \
                f'& population_control={population_control} & n={n} ' \
                f'& crop_income_mode={crop_income_mode} & better_ess={better_ess} ' \
                f'& kill_cropless={kill_cropless} & steps={steps} ' \
                f'& settlement_output_path={settlement_output_path} & test={test}'

        with SafeHDFStore(compute_handle.path_raw) as store:
            trj = store.select("dat_1", where=query)

        return 1, trj

    # Create dummy runfunc output to pass its shape to experiment handle

    try:
        if not Path(save_path_raw).exists():
            Path(save_path_raw).mkdir(parents=True, exist_ok=True)
        rf_output = pd.read_pickle(save_path_raw + 'rfof.pkl')
    except FileNotFoundError:
        params = list(param_combs[0])
        params[-1] = True
        rf_output = run_function(*params)[1]

    # Run computation and post processing.

    # devide parameter combination into equally sized chunks.
    cl = int(len(param_combs)/max_id)
    i = (job_id-1)*cl
    j = job_id*cl

    # initialize computation and post processing handles
    compute_handle = eh(run_func=run_function,
                        runfunc_output=rf_output,
                        sample_size=sample_size,
                        parameter_combinations=param_combs[i:j],
                        path_raw=save_path_raw,
                        )
    pp1_handle = eh(run_func=mean,
                    runfunc_output=rf_output,
                    sample_size=1,
                    parameter_combinations=param_combs,
                    path_raw=save_path_res + '/mean.h5',
                    )
    pp2_handle = eh(run_func=sem,
                    runfunc_output=rf_output,
                    sample_size=1,
                    parameter_combinations=param_combs,
                    path_raw=save_path_res + '/sem.h5',
                    )
    pp3_handle = eh(run_func=collect_final_states,
                    runfunc_output=rf_output,
                    sample_size=1,
                    parameter_combinations=param_combs,
                    path_raw=save_path_res + '/final_states.h5',
                    )

    if mode == 1:
        compute_handle.compute()
        return 0
    if mode == 2:
        pp1_handle.compute()
        pp2_handle.compute()
        pp3_handle.compute()

    return 1


# The definition of the run_function makes it easier to test the experiment with pytest.
if __name__ == '__main__':

    run_experiment(sys.argv)
