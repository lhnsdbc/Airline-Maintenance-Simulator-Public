import glob
import logging
import os
from config import directories, RESULTS, RUN_CONFIG, INPUT_FILES
import pandas as pd
from output.output_functions import write_csv_from_dataframe, log_error, log_warning


def save_config_file(simulation, iteration_start):
    ''' Copy the current config file as a text file in the output folder'''
    # Find directory config file
    path_config = os.path.join(directories.anemos, 'config.py')
    # Directory of simulation file
    path_output = os.path.join(directories.output, simulation)
    path_output = os.path.join(path_output, 'config_'+str(iteration_start)+'.txt')

    with open(path_config) as f:
        data = f.read()

    with open(path_output, mode='w') as f:
        f.write(data)



def concatenate_results_df_for_dashboard():
    ''' Generate the output files to be used in the dashboard by concatenating the results from the selected
    simulations'''
    simulations = RUN_CONFIG.RESULTS_DASHBOARD_OUTPUT
    print('Generating input for the results dashboard: ', str(simulations))
    # Find list of output directories for each requested simulation
    paths_output = []
    for simulation in simulations:
        directory_simulation_output = os.path.join(directories.output, simulation)
        paths_output.append(directory_simulation_output)

    # List of results files identifiers
    results_file_types = RESULTS.FILES_TO_CONCATENATE_IDENTIFIERS

    for file_type in results_file_types:
        # Find list of directories of relevant file type
        results_directories = []
        for path_output in paths_output:
            # For each result file type and each simulation find complete file
            name_pattern = os.path.join(path_output, file_type+'*')
            results_directories_simulation = glob.glob(name_pattern)
            # Add found file to list of files of file type
            results_directories = results_directories + results_directories_simulation

        # Concatenate the dfs found at given directories:
        df_dashboard = concatenate_df_from_directories(results_directories)

        # Find name of df
        df_dashboard_filename = 'dashboard'+'_'+file_type

        # If it's flights df, add costs columns
        if file_type == RESULTS.FILE_NAMES['flights']:
            df_dashboard = compute_flights_cost(df_dashboard)

        # Save complete df
        write_csv_from_dataframe(filename=df_dashboard_filename, dataframe=df_dashboard, simulation_id='dashboard')

    print('Dashboard input generated successfully')

def concatenate_results_df(simulation):
    ''' Generate files that concatenate the results obtained from different simulation iterations within the same
    simulation run '''
    # Find directory of results
    path_output = os.path.join(directories.output, simulation)
    # List of output file types to concatenate
    results_file_types = RESULTS.FILES_TO_CONCATENATE_IDENTIFIERS

    for file_type in results_file_types:
        # For each result file type find list of corresponding files
        name_pattern = os.path.join(path_output, file_type + '*')
        results_directories = glob.glob(name_pattern)

        # Concatenate the dfs found at given directories:
        df_complete = concatenate_df_from_directories(results_directories)

        # Find name of df
        df_complete_filename = find_name_complete_file(file_type, simulation)
        # Save complete df
        write_csv_from_dataframe(filename=df_complete_filename, dataframe=df_complete, simulation_id=simulation)

    print('Complete simulation results generated successfully')

def find_name_complete_file(file_type, simulation_id):
    return 'complete_'+file_type + '_' + simulation_id

def concatenate_df_from_directories(directories_df):
    '''Given a list of directories where dictionaries to concate are located, returns the concatenated df'''
    # Initialize list of dataframes to concatenate
    df_to_concatenate = []
    # Import dataframes and add to list of df to concatenate
    for directory in directories_df:
        try:
            df = pd.read_csv(directory, decimal='.', parse_dates=True)
            df_to_concatenate.append(df)
        except:
            log_warning('The directory could not be opened: ' + directory)

    # Concatenate dfs in one df
    if df_to_concatenate != []:
        df_concatenated = pd.concat(df_to_concatenate)
    else:
        log_error('No dataframe to concatenate was found')
        df_concatenated = pd.DataFrame({'empty_col' : []})
    return df_concatenated


def compute_flights_cost(df_flights):

    # Make delay columns into int
    df_flights['delay_arrival'] = df_flights['delay_arrival'].astype(int)
    df_flights['delay_departure'] = df_flights['delay_departure'].astype(int)

    # Df future value
    directory_file = os.path.join(directories.input, INPUT_FILES.FUTURE_VALUE + '.xlsx')
    df_future_value = pd.read_excel(directory_file, decimal='.', parse_dates=True)
    # Only keep relevant columns
    df_future_value = df_future_value[['Delay', 'ICA_non_elite']]

    costs = RESULTS.COSTS_CARRIER
    # CANCELLATION
    df_flights['cost_cancellation'] = 0
    df_flights['cost_cancellation'] = df_flights['cost_cancellation'].mask(df_flights['execution_state'].str.contains('cancelled'),
                                                                           costs['cancellation'])
    # FOOD - now disregarded
    df_flights['cost_food'] = 0
    # df_flights['cost_food'] = df_flights['cost_food'].mask((df_flights['execution_state'] == 'executed')
    #                                                        & (df_flights['delay_departure']>costs['food_time']),
    #                                                        costs['food']*costs['passengers'])

    # COMPENSATION
    df_flights['cost_compensation'] = 0
    # Compensation 3 hours
    df_flights['cost_compensation'] = df_flights['cost_compensation'].mask((df_flights['execution_state'] == 'executed')
                                                                           & (df_flights['delay_arrival']>=3*60)
                                                                           & (df_flights['delay_arrival']<4*60),
                                                                           costs['compensation_delay_3hr']*costs['passengers']*costs['claim_rate'])
    # Compensation 4 hours
    df_flights['cost_compensation'] = df_flights['cost_compensation'].mask((df_flights['execution_state'] == 'executed')
                                                                           & (df_flights['delay_arrival']>=4*60),
                                                                           costs['compensation_delay_4hr']*costs['passengers']*costs['claim_rate'])
    # FUTURE VALUE
    df_flights = pd.merge(df_flights, df_future_value, how='left', left_on='delay_arrival', right_on='Delay')
    # Flights with delay higher than the highest in table should be assigned the maximum value
    df_flights['ICA_non_elite'] = df_flights['ICA_non_elite'].mask((df_flights['execution_state'] == 'executed')
                                                                   & (df_flights['delay_arrival']>max(df_future_value['Delay'])),
                                                                   max(df_future_value['ICA_non_elite']))

    df_flights['cost_future_value'] = 0
    df_flights['cost_future_value'] = df_flights['cost_future_value'].mask((df_flights['execution_state'] == 'executed')
                                                                           & (df_flights['delay_arrival'] > 0),
                                                                           df_flights['ICA_non_elite']*costs['passengers'])

    # TOTAL COST
    df_flights['cost_total'] = df_flights['cost_cancellation'] + \
                               df_flights['cost_compensation'] + df_flights['cost_future_value']
    df_flights['cost_total'] = df_flights['cost_total'].astype(int)

    # Drop unnecessary columns
    df_flights = df_flights.drop(columns=['Delay', 'ICA_non_elite'])

    # Save obtained file
    # write_csv_from_dataframe(filename=df_flight_filename, dataframe=df_flights, simulation_id='dashboard')

    return df_flights


def change_simulation_id_in_results_files(id_old, id_new):
    ''' Change the id of a simulation from id_old to id_new in all directory, filenames and files'''
    # Old and new directory of results
    directory_old = os.path.join(directories.output, id_old)
    directory_new = os.path.join(directories.output, id_new)
    # If directory already exists, raise exception. Otherwise, generate it
    file_exists = os.path.exists(directory_new)
    if file_exists:
        raise Exception('Directory already exists')
    os.mkdir(directory_new)

    # Find list of files in old directory
    file_names_pattern = os.path.join(directory_old, '*.csv')
    all_files_old = glob.glob(file_names_pattern)

    # Open each directory and change the old id to the new one wherever found
    for directory in all_files_old:
        try:
            df = pd.read_csv(directory, decimal='.', parse_dates=True)
        except:
            log_error('Directory',directory,'could not be opened')
            continue

        for col in df.columns:
            try:
                df[col] = df[col].str.replace(id_old, id_new)
            except:
                pass
        # Save new file in new directory
        filename_old = os.path.basename(directory)
        filename_new = filename_old.replace(id_old, id_new)
        filename_new = filename_new.replace('.csv', '')
        write_csv_from_dataframe(filename=filename_new, dataframe=df, simulation_id=id_new)

    print('All files generated')












