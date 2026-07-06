import os
from data_import.load_data import load_data
from data_import.generate_objects import generate_objects
from simulation.anemos import Anemos
from config import G, directories, RUN_CONFIG
from datetime import datetime as dt
from output.results_manage_files import save_config_file, concatenate_results_df
import multiprocessing as mp
import timeit
import logging, coloredlogs
import tqdm
from output.output_functions import log_error


def run_anemos_scenarios():
    ''' Run multiple scenarios of Anemos in series'''
    for scenario in G.SCENARIO_SIMULATION:
        if type(scenario) == str:
            run_anemos(scenario)
        elif type(scenario) == tuple:
            run_anemos(scenario=scenario[0],
                       simulation_time_id=scenario[1],
                       iter_start=scenario[2])
        else:
            raise Exception('G.SCENARIO_SIMULATION items type not supported')


def run_anemos(scenario, simulation_time_id=None, iter_start=None):
    '''
    Set up the simulation and run multiple iterations. If RUN_CONFIG is set to 0, run iterations in series; if set to
    1000, run iterations in parallel.'''

    # Keep track of run time
    time_start = timeit.default_timer()
    # Find dame of the simulation
    # scenario_id = G.SCENARIO_SIMULATION + '_'
    scenario_id = scenario + '_'
    if G.INPUT_SIM_ID == 1:
        simulation_run_id = scenario_id + input('Simulation run id: ')
        iteration_start = int(input('Iteration to start from: '))
    elif G.INPUT_SIM_ID == 0 and simulation_time_id==None:
        # PAPER_DESIGN: experiment runner sets ANEMOS_RUN_TAG for deterministic, cell-named
        # output folders (e.g. default_run_expA_R0_static); otherwise fall back to a timestamp.
        run_tag = os.environ.get('ANEMOS_RUN_TAG')
        simulation_run_id = scenario_id + (run_tag if run_tag else dt.today().strftime('%Y%m%d_%H%M%S'))
        # Allow a batch campaign to append a new, non-overlapping iteration range to an
        # existing deterministic output folder. The default remains a fresh run at zero.
        iteration_start = int(os.environ.get('ANEMOS_ITERATION_START', '0'))
    elif G.INPUT_SIM_ID == 0 and simulation_time_id!=None:
        simulation_run_id = scenario_id + simulation_time_id
        iteration_start = iter_start
        # Check that simulation already exists

    else:
        raise Exception('G.INPUT_SIM_ID value not supported')

    print('Initializing simulation '+simulation_run_id)
    # Create a folder for the simulation results
    create_results_folder(simulation_run_id)
    # Save config file
    save_config_file(simulation_run_id, iteration_start)
    # Load data
    data = load_data(scenario)

    if RUN_CONFIG.MODE == 0:
        run_iterations_in_series(data, simulation_run_id, iteration_start)
    elif RUN_CONFIG.MODE == 1000:
        run_iterations_in_parallel(data, simulation_run_id, iteration_start)
    else:
        raise Exception('RUN_CONFIG value not supported')

    # Concatenate results obtained in simulation
    concatenate_results_df(simulation_run_id)

    # Print run time info
    time_run = round((timeit.default_timer() - time_start) / 60)
    print('\nSimulation ', simulation_run_id, ' ended || Run time: ', time_run, 'minutes')


def create_results_folder(folder_name):
    ''' Generate a folder in the results directory with the desired name '''
    directory_results = directories.output
    directory_new_folder = os.path.join(directory_results, folder_name)
    file_exists = os.path.exists(directory_new_folder)
    if file_exists == 0:
        log_error('A new folder was generated for the results')
        os.mkdir(directory_new_folder)
    else:
        log_error('Adding results to existing folder')


def run_iterations_in_series(data, simulation_run_id, iteration_start):
    ''' Run simulation iterations in series'''
    # Set correct logging level
    set_log()
    for iteration_n in range(iteration_start, iteration_start + G.SIM_ITERATIONS):
        run_iteration(iteration_n, data, simulation_run_id)


def custom_error_callback(error):
    print('An error has occurred', error)


def run_iterations_in_parallel(data, simulation_run_id, iteration_start):
    ''' Run simulation iterations in parallel '''
    print('using', G.PARALLEL_RUN_N_CORES, 'out of',  mp.cpu_count(), 'available cores')
    # Initialize pool with correct log settings
    pool = mp.Pool(G.PARALLEL_RUN_N_CORES, initializer=set_log)
    async_results = []
    for iteration_n in range(iteration_start, iteration_start + G.SIM_ITERATIONS):
        async_results.append(pool.apply_async(
            run_iteration,
            args=(iteration_n, data, simulation_run_id, iteration_start),
            error_callback=custom_error_callback))
    pool.close()
    # Wait for all operation to end
    pool.join()
    # Async worker exceptions otherwise only reach the print-only callback, allowing a cell
    # to exit zero and concatenate an incomplete result set. Re-raise in the parent so the
    # batch runner records the cell as failed.
    for result in async_results:
        result.get()


def run_iteration(iteration_n, data, simulation_run_id, iteration_start=0):
    ''' Initialize and run one single iteration of the simulation '''
    # PAPER_DESIGN sec. 4.4: optional deterministic seeding so paired experiment cells
    # (e.g. static vs predicted NR at the same rung/iteration) see the same random stream.
    base_seed = os.environ.get('ANEMOS_BASE_SEED')
    if base_seed is not None:
        import random as _random
        import numpy as _np
        seed = int(base_seed) + iteration_n
        _random.seed(seed)
        _np.random.seed(seed % (2 ** 32))
    # Generate objects or import from pickle
    obj = generate_objects(data, iteration_n)
    # Generate and run simulation iteration
    simulation = Anemos(obj=obj, data=data, sim_run_id=simulation_run_id, sim_iteration=iteration_n,
                        iteration_start=iteration_start)
    simulation.run()


def set_log():
    log_format = '%(message)s'  # '%(levelname)s '
    # Pyomo/appsi can redirect logging streams during solve(). coloredlogs installs
    # handlers whose stream property cannot always be reassigned, which raises:
    #   AttributeError: property 'stream' of 'StandardErrorHandler' object has no setter
    # Use plain stdlib logging by default; colored console output is cosmetic and can
    # be explicitly re-enabled for environments known to support it.
    if os.environ.get('ANEMOS_USE_COLOREDLOGS') not in ('1', 'true', 'True'):
        level = logging.INFO if RUN_CONFIG.MODE == 0 else logging.ERROR
        logging.basicConfig(level=level, format=log_format, force=True)
        return

    coloredlogs.DEFAULT_LEVEL_STYLES = {
        'debug': {'color': 'green', 'bold': False, 'bright': True},
        'info': {'color': 'white', 'bold': False, 'bright': True},
        'warning': {'color': 'yellow', 'bold': False, 'fair': True},
        'error': {'color': 'red', 'bold': False, 'fair': True}
    }
    coloredlogs.DEFAULT_FIELD_STYLES = {'levelname': {'color': 'black', 'bold': False, 'bright': True}}

    # Set logging level
    if RUN_CONFIG.MODE == 0:
        # logging.basicConfig(level=logging.DEBUG)
        coloredlogs.install(level='INFO', fmt=log_format)

    elif RUN_CONFIG.MODE == 1000:
        # logging.basicConfig(level=logging.ERROR)
        coloredlogs.install(level='ERROR', fmt=log_format)
    else:
        raise Exception('RUN_CONFIG.MODE not supported')
