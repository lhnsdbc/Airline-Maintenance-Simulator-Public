import logging

import pandas as pd
from data_import.input_output import load_csv, load_excel , write_pickle, read_pickle #, write_csv
from config import INPUT_FILES, M
import logging
from datetime import timedelta
from config import RUN_CONFIG, G, P, directories
from datetime import datetime as dt
from datetime import timedelta
from itertools import groupby, product
import numpy as np
from matplotlib import pyplot as plt
import time
from math import floor, ceil
from distfit import distfit
from tqdm import tqdm
import sys
import os
import glob
from statistics import mean
from scipy import stats
from collections import Counter

class EmptyClass:
    pass


def find_distributions_maintenance_faults():
    ##### IMPORT AND PROCESS DATA #####
    # Import raw dataframes or process ones
    if RUN_CONFIG.MODE == 12:
        # Import Maintenix files and split it into single dataframes
        df_mtx = split_full_maintenix_data_file()
        # Process dataframes
        df_tasks = generate_tasks_dataframes(df_mtx)
        # Save to pickle
        write_pickle(df_tasks, 'distributions_maintenance_dataframes')
    ##### IMPORT PICKLE #####
    elif RUN_CONFIG.MODE == 13:
        df_tasks = read_pickle('distributions_maintenance_dataframes')

    else:
        raise Exception('Run config mode not supported')

    ##### FIND DISTRIBUTIONS DEFERRED DEFECTS #####
    DDs, DDs_tasks = generate_deferred_defect_df(df_tasks['DDs'])
    DDs_distributions = find_distributions_deferred_defects(df_tasks, DDs)

    ##### FIND DISTRIBUTIONS NON-ROUTINES #####
    NR_distributions = find_distributions_non_routines(df_tasks)

    ##### SAVE PICKLES #####
    write_pickle(NR_distributions, 'distributions_NR')
    write_pickle(DDs_distributions, 'distributions_DD')
    write_pickle(DDs_tasks, 'tasks_DD')

    plt.show()
    print('hello maintenance')


# =================================================================================#
# IMPORT RAW DATAFRAMES
# =================================================================================#
def split_full_maintenix_data_file():
    ''' Import the Maintenix data regarding faults, non-routines, and ad-hoc tasks and separates it into multiple
    dataframes'''
    # Dataframe of column and df names
    data_mtx_columns = load_csv(INPUT_FILES.ENGINEERING_NON_RECURRING_COL_NAMES)
    first_col_name = data_mtx_columns['data_first_column_name'][0]

    # Initialize dictionary of dataframes
    df_mtx = {}

    # Each engineering non recurring file must be opened separately and concatenated to previous ones
    files_path = os.path.join(directories.input, INPUT_FILES.ENGINEERING_NON_RECURRING_FOLDER)
    files_path = os.path.join(files_path, INPUT_FILES.ENGINEERING_NON_RECURRING+'*')
    files_list = glob.glob(files_path)

    # Progress bar
    progress = tqdm(files_list, file=sys.stdout)
    for file in progress:
        # Progress bar description
        progress.set_description('Engineering_non_recurring files imported')
        data_mtx_full = load_csv(filename=None, directory_full=file)
        df_full_length = len(data_mtx_full)
        # Dynamic end of current considered df
        df_end = len(data_mtx_full)

        for i in range(len(data_mtx_columns)-1, -1, -1):
            # Find characteristic of new df
            df_new_name_col = data_mtx_columns['data_first_column_name'][i]
            df_new_name_df = data_mtx_columns['data_name'][i]

            # Find index of dataframe where extracted df starts
            if df_new_name_col!=first_col_name:
                df_start = data_mtx_full[data_mtx_full[first_col_name]==df_new_name_col].index.to_list()
            else:
                df_start = [-1]
            # check that only one index found
            if len(df_start)!=1:
                raise Exception('Column name must be found only once')
            df_start = df_start[0]

            # Import dataframe
            df_new = load_csv(filename=None, directory_full=file, decimal=',', skiprows=df_start+1,
                              skipfooter=df_full_length - df_end)

            # Add to dictionary of dataframes or concatenate to existing df
            try:
                df_mtx[df_new_name_df] = pd.concat([df_mtx[df_new_name_df], df_new], ignore_index=True)
            except:
                df_mtx[df_new_name_df] = df_new

            # Update end of df
            df_end = df_start

    return df_mtx


# =================================================================================#
# PROCESS DATAFRAMES
# =================================================================================#

def generate_tasks_dataframes(df_mtx):
    # Dataframe of all the work packages in the considered time period
    WP = process_wp_df(df_mtx['wp_hangar_full'])
    # Initialize dataframe with tasks info
    tasks = tasks_initialize(df_mtx['production'])
    # Add due date of tasks
    tasks = tasks_add_due_date(tasks, df_mtx['date_due'])
    # Add arrival date
    tasks = tasks_add_arrival_date(tasks, df_mtx['date_arrival'])
    # Add labor info
    tasks = tasks_add_labor(tasks, df_mtx['labor'])
    # Add faults info
    tasks = tasks_add_faults_info(tasks, df_mtx['faults'])

    # Add parts info
    tasks = tasks_add_ready_date(tasks, df_mtx['parts'])
    # Add AOG info
    tasks = tasks_add_AOG_info(tasks, WP)

    # Find start and end window of the considered data
    window_start, window_end = find_data_window(tasks)
    # Separate deferred defects from non-routines
    DDs, NR = separate_DDs_NR(tasks)
    # Find total labor and duration for DD df
    DDs = find_total_labor_and_duration_DDs(DDs)
    # Filter NR and WP dfs by work location and non-AOG slots
    NR, WP = filter_for_NR_elaboration(NR, WP)
    # Find total labor and duration for NR df
    NR = find_total_labor_in_workpackage_NR(NR)

    dataframes_tasks = {'DDs':DDs,
                        'NR':NR,
                        'wp_hangar':WP,
                        'window_start':window_start,
                        'window_end':window_end}

    return dataframes_tasks

def tasks_initialize(df_production):
    ''' Initialize a dataframe containing tasks info, based on the production dataframe given as input'''
    tasks = df_production[['Tasks_Barcode', 'Tasks_Task name', 'TaskInf_Type', 'TaskInf_Task Definition',
                                  'WorkPackage_BarCode', 'WorkPackage_No', 'WorkPackage', 'WorkPackage_ActualStartDate',
                                  'WorkPackage_ActualEndDate', 'WorkPackage_WorkLocation', 'TaskInf_Work Type',
                                  'WorkPackage_Aircraft', 'WorkPackage_Aircraft_Fleet', 'WorkPackage_WorkType',
                                  'Tasks_Estimated Duration'
                                  ]]
    # Remove duplicate tasks if assigned to multiple workpacakges than only the last one should stay
    tasks = tasks.sort_values('WorkPackage_ActualStartDate')
    tasks = tasks.drop_duplicates(['Tasks_Barcode', 'WorkPackage_BarCode'], keep='last')

    # Remove bureaucratic tasks (fall back procedures and pre-flight)
    tasks['Tasks_Task name'] = tasks['Tasks_Task name'].fillna('')
    tasks = tasks[~tasks['Tasks_Task name'].str.contains('DECL_FB')]
    tasks = tasks[~tasks['Tasks_Task name'].str.contains('Pre-flight')]

    return tasks

def tasks_add_due_date(tasks, df_date_due):
    '''
    Given a dataframe with info about tasks, add their due date, when available.
    If multuple due dates are available, the latest one is considered.
    If a due date extension was given, the extended due date is condiered.
    '''
    # Remove due dates that are too much into the future ex: 2250
    df_date_due = df_date_due[df_date_due['Due.Date'] < '2050-01-01']
    # If extension to due date given, modify due date
    df_date_due['Due.Date'] = df_date_due['Due.Date'].mask(pd.isnull(df_date_due['Due.Date Extend']) == 0,
                                                           df_date_due['Due.Date Extend'])
    # Keep last registered due date
    df_date_due = df_date_due.sort_values(['Tasks_Barcode', 'Due.Date'])
    df_date_due = df_date_due.drop_duplicates('Tasks_Barcode', keep='last')
    tasks = pd.merge(tasks, df_date_due[['Tasks_Barcode','Due.Date']], how='left', on=['Tasks_Barcode'])

    return tasks


def tasks_add_arrival_date(tasks, df_arrival):
    ''' Given a dataframe with info about tasks, add their arrival date, when available. '''
    # Drop duplicates on task arrival
    df_arrival = df_arrival.drop(columns={'Tasks_WorkPackageNumber', 'HistTask_Note'})
    df_arrival = df_arrival.drop_duplicates()
    # Merge arrival date
    tasks = pd.merge(tasks, df_arrival, how='left', on='Tasks_Barcode')
    return tasks

def tasks_add_labor(tasks, df_labor):
    ''' Given a dataframe with info about tasks, add their scheduled and actual labor.'''
    # Rename columns
    df_labor = df_labor.rename(index=str, columns = {'LaborReq.Sched. Hours': 'Labor_sched',
                                                     'LaborReq.Actual Hrs': 'Labor_act'})
    # Remove labor rows with no scheduled and actual labor set to zero
    df_labor = df_labor[(df_labor['Labor_sched'] != 0) | (df_labor['Labor_act'] != 0)]

    # Merge tasks and labor dataframes
    tasks = pd.merge(tasks,df_labor, how='left',left_on=['Tasks_Barcode','WorkPackage_No'] ,
                                                right_on=['Tasks_Barcode', 'Tasks_WorkPackageNumber'])

    # Remove labour rows which are not valid for the current workpackage. Add slack to account for not accurate data
    tasks = tasks[tasks['Details.Start Time'] >= tasks['WorkPackage_ActualStartDate']
                  - timedelta(minutes=M.SLACK_INCLUDE_TASK)]
    tasks = tasks[tasks['Details.End Time'] <= tasks['WorkPackage_ActualEndDate']
                  + timedelta(minutes=M.SLACK_INCLUDE_TASK)]

    # Drop unnecessary columns
    tasks = tasks.drop(columns=['Tasks_WorkPackageNumber'])# 'Details.Start Time', 'Details.End Time', ])

    return tasks

def tasks_add_faults_info(tasks, faults):
    ''' Given a dataframe with info about tasks, add faults info to faults '''
    faults = faults.drop_duplicates()
    tasks = pd.merge(tasks, faults, how='left', left_on=['Tasks_Barcode','WorkPackage_No'] ,
                                                right_on=['Tasks_Barcode', 'Tasks_WorkPackageNumber'])

    # If fault found on date column not available, substitute it with the arrival date
    tasks['Fault_FoundOnDate'] = tasks['Fault_FoundOnDate'].mask(pd.isnull(tasks['Fault_FoundOnDate']),
                                                                 tasks['HistTask_Date'])

    tasks = tasks.drop(columns=['Fault_Name', 'Tasks_WorkPackageNumber', 'Fault Found During Task', 'Fault_Severity',
                                'HistTask_Date'])

    return tasks


def find_data_window(tasks):
    ''' Find the first and last dates of wp start time'''
    wp_dates = tasks['WorkPackage_ActualStartDate'].to_list()
    wp_start = min(wp_dates).date()
    wp_end = max(wp_dates).date()
    return wp_start, wp_end

def separate_DDs_NR(tasks):
    ''' Separate deferred defects and ad hoc tasks, and non-routines into separate dataframes '''
    # DDs and Ad-hoc tasks are grouped together. DDs are recognized because they have a deferral class.
    DDs = tasks[(pd.isnull(tasks['Due.Date'])==0) | (tasks['TaskInf_Type']=='ADHOC')]

    # NR are corrective maintenance tasks without a deferral class.
    NR = tasks[(pd.isnull(tasks['Due.Date'])) & (tasks['TaskInf_Type']=='CORR')]

    # Print warning if not all rows considered
    len_total = len(DDs) + len(NR)
    if len(tasks) != len_total:
        logging.warning('Some task rows are not considered or considered multiple times')

    return DDs, NR

def find_total_labor_and_duration_DDs(DDs):
    '''
    Find the duration and labor for each task as follows:
    - Labor (scheduled and actual): sum of the labor associated to a tasks
    - Duration: Max duration found for each task.
    '''
    # Labor scheduled and actual are summed for each task
    DDs['Labor_sched_total'] = DDs['Labor_sched'].groupby(DDs['Tasks_Barcode']).transform('sum')
    DDs['Labor_act_total'] = DDs['Labor_act'].groupby(DDs['Tasks_Barcode']).transform('sum')

    # Duration is estimated as the maximum found duration for that task
    DDs['Duration_max'] = DDs['Tasks_Estimated Duration'].groupby(DDs['Tasks_Barcode']).transform('max')

    # Only keep relevant columns
    DDs = DDs[['Tasks_Barcode','Tasks_Task name', 'TaskInf_Type', 'Labor_sched_total', 'Labor_act_total',
              'Duration_max', 'Fault_DeferralClass', 'Fault_FoundOnDate','Due.Date','TaskInf_Work Type',
              'WorkPackage_Aircraft', 'WorkPackage_Aircraft_Fleet', 'WorkPackage_ActualStartDate',
               'WorkPackage_ActualEndDate']]

    # Drop duplicates. There should be one row per task
    DDs = DDs.drop_duplicates()

    # Rename columns
    DDs = DDs.rename(index=str, columns={'Tasks_Barcode': 'task_barcode',
                                         'Tasks_Task name': 'task_name',
                                         'TaskInf_Type': 'task_type',
                                         'Labor_sched_total': 'labor_sched',
                                         'Labor_act_total': 'labor_act',
                                         'Duration_max': 'duration',
                                         'Fault_DeferralClass': 'deferral_class',
                                         'Fault_FoundOnDate': 'date_fault_found',
                                         'Due.Date': 'date_due',
                                         'TaskInf_Work Type': 'task_work_type',
                                         'WorkPackage_Aircraft': 'ac_regis',
                                         'WorkPackage_Aircraft_Fleet': 'ac_fleet'
                                         })


    return DDs


def filter_for_NR_elaboration(NR, WP):
    ''' Non routines are only considered when not executed in an AOG slot, and in the hangar  '''
    # Remove workpackages corresponding to AG slots
    WP = WP[pd.isnull(WP['AOG_slot'])]
    # Only keep work packages executed in the hangar, but corresponding to work type different from A and C type
    WP = WP[WP['work_location'].str.contains('/H')]
    WP = WP[(WP['wp_work_type']!='A') & (WP['wp_work_type']!='C')]

    # Only keep NR that correspond to list of work packages in WP dataframe
    NR = NR[NR['WorkPackage_BarCode'].isin(WP['wp_barcode'].to_list())]

    # Drop columns regardin AOG slots in both dataframes and ready date for NR
    NR = NR.drop(columns=['date_ready', 'AOG_slot', 'AOG_date_start', 'AOG_date_end', 'AOG_duration'])
    WP = WP.drop(columns=['AOG_slot', 'AOG_date_start', 'AOG_date_end', 'AOG_duration'])

    return NR, WP


def find_total_labor_in_workpackage_NR(NR):
    ''' Compute the non-routine labor hours per work package. Labor hours are computed as a sum of all the ACTUAL
    labor hours of all the included non-routines (NR). NR with actual labor set to zero are disregarded. Only work
    packages with worktype H are considered. '''

    # Exclude NR when the associated actual labor hours is equal to zero
    NR = NR[NR['Labor_act']!=0]

    # Sum all labor executed in one workpackage
    NR['wp_labor_total'] = NR['Labor_act'].groupby(NR['WorkPackage_BarCode']).transform('sum')

    # Only keep relevant columns
    NR = NR[['WorkPackage_BarCode', 'WorkPackage','WorkPackage_WorkLocation' , 'wp_labor_total',
              'WorkPackage_Aircraft', 'WorkPackage_Aircraft_Fleet']]

    # Drop duplicates. There should be one row per work package
    NR = NR.drop_duplicates('WorkPackage_BarCode')

    # Rename columns
    NR = NR.rename(index=str, columns={'WorkPackage_BarCode': 'wp_barcode',
                                       'WorkPackage': 'wp_name',
                                       'WorkPackage_WorkLocation': 'work_location',
                                       'WorkPackage_Aircraft': 'ac_regis',
                                       'WorkPackage_Aircraft_Fleet': 'ac_fleet'})

    return NR



def process_wp_df(wp):
    ''' Reduces the wp dataframe and change names to columns'''

    # Keep relevant columns
    wp = wp[['WorkPackage_BarCode', 'WorkPackage', 'WorkPackage_WorkLocation',
             'WorkPackage_Aircraft', 'WorkPackage_Aircraft_Fleet', 'WorkPackage_WorkType',
             'WorkPackage_ActualStartDate', 'WorkPackage_ActualEndDate']]

    # There should not be duplicates. Use this line to make sure
    wp = wp.drop_duplicates('WorkPackage_BarCode', keep='first')

    # Rename columns
    wp = wp.rename(index=str, columns={'WorkPackage_BarCode': 'wp_barcode',
                                       'WorkPackage': 'wp_name',
                                       'WorkPackage_WorkLocation': 'work_location',
                                       'WorkPackage_Aircraft': 'ac_regis',
                                       'WorkPackage_Aircraft_Fleet': 'ac_fleet',
                                       'WorkPackage_WorkType': 'wp_work_type',
                                       'WorkPackage_ActualStartDate': 'wp_date_start',
                                       'WorkPackage_ActualEndDate': 'wp_date_end'})

    # Find which work packages correspond to AG slots
    wp = find_AOG_work_packages(wp)

    return wp


def find_AOG_work_packages(df_wp):
    ''' Find AOG work packages based on overlap with AOG maintenance slot and on wp name'''
    ##### AOG WORK PACKAGES FOR OVERLAP WITH AOG SLOTS #####
    # Import list of AOG slots
    df_aog = import_df_AOG_mtop()
    # Reduce list to match work package data
    data_start = min(df_wp['wp_date_start'].to_list())
    data_end = max(df_wp['wp_date_end'].to_list())
    df_aog = df_aog[(df_aog['date_start'] >= data_start - timedelta(hours=4)) &
                    (df_aog['date_end'] <= data_end + timedelta(hours=4))]

    # Look for AOG slot corresponding to the considered WP
    df_wp = df_wp.reset_index(drop=True)
    for wp_index, wp_row in df_wp.iterrows():
        slots_match = find_slot_match_for_wp(wp_row, df_aog)
        df_wp.loc[wp_index, 'AOG_slot'] = slots_match


    df_aog = df_aog.rename(columns={'date_start': 'AOG_date_start',
                                    'date_end': 'AOG_date_end',
                                    'slot_duration': 'AOG_duration',})
    df_wp = pd.merge(df_wp, df_aog[['slot_id', 'AOG_date_start', 'AOG_date_end', 'AOG_duration']],
                     left_on='AOG_slot', right_on='slot_id', how='left')
    df_wp = df_wp.drop(columns=['slot_id'])

    ##### AOG WORK PACKAGES BASED ON WP NAME #####
    # If work package includes 'aog' in its name, also mark as a AOG
    df_wp['AOG_slot'] = df_wp['AOG_slot'].mask((pd.isnull(df_wp['AOG_slot'])) &
                                               (df_wp['wp_name'].str.lower().str.contains('aog')),
                                               'AOG')

    return df_wp

def find_slot_match_for_wp(wp, df_aog):
    wp_regis = wp['ac_regis'].replace('-','')

    # To find a match, there must be the same registration, and it must be true that
    # (StartA <= EndB)  and  (EndA >= StartB), considering a buffer
    buffer = timedelta(hours=1)
    df_aog_overlap = df_aog[(df_aog['ac_regis'] == wp_regis) &
                            (df_aog['date_start'] - buffer <= wp['wp_date_end'] + buffer) &
                            (df_aog['date_end'] + buffer >= wp['wp_date_start'] - buffer)]

    # Flag for match found for the work package
    if len(df_aog_overlap) == 0:
        aog_slot_found = np.nan

    else:
        aog_slot_found = df_aog_overlap['slot_id'].to_list()[0]
        # Raise warning if more than one slot found
        if len(df_aog_overlap)>1:
            logging.warning(str(len(df_aog_overlap)) + ' slots found for one work package')

    return aog_slot_found


def tasks_add_AOG_info(tasks, wp):
    ''' Find if task was executed in an AG slot '''

    # Merge AOG slot info into tasks df
    tasks = pd.merge(tasks, wp[['wp_barcode', 'AOG_slot', 'AOG_date_start', 'AOG_date_end', 'AOG_duration']],
                     left_on='WorkPackage_BarCode', right_on='wp_barcode', how='left' )
    tasks = tasks.drop(columns=['wp_barcode'])

    return tasks


def tasks_add_ready_date(tasks, parts):
    ''' Determine the ready date of the considered tasks based on their parts requests.
    When a part request is found, the ready date corresponds to the first occurring between the part arrial day and
    the task start date.'''

    # Rename column
    parts = parts.rename(columns={'Request Master ID': 'Request_Id'})

    # Only keep status of available part
    parts = parts[(parts['Request_Status'] == 'AVAIL') |
                  (parts['Request_Status'] == 'RESERVE') |
                  (parts['Request_Status']== 'AWAITING ISSUE')]

    # Sort by requests by date
    parts = parts.sort_values(['Tasks_Barcode', 'Request_Id', 'Request_Date'])

    # Only keep first available status
    parts = parts.drop_duplicates('Request_Id', keep='first')
    # If more requests for one task, keep the one that is available later
    parts = parts.sort_values(by=['Tasks_Barcode', 'Request_Id'])
    parts = parts.drop_duplicates(['Tasks_Barcode'], keep='last')

    # Merge ready dates into tasks df
    tasks = pd.merge(tasks, parts[['Tasks_Barcode', 'Request_Date']], on='Tasks_Barcode', how='left')

    # When a part request exists, set the ready date to the earliest between the task start time and part arrival time
    tasks['date_ready'] = np.nan
    tasks['date_ready'] = tasks['date_ready'].mask(tasks['Details.Start Time'] < tasks['Request_Date'],
                                                   tasks['Details.Start Time'])
    tasks['date_ready'] = tasks['date_ready'].mask(tasks['Details.Start Time'] > tasks['Request_Date'],
                                                   tasks['Request_Date'])

    # Remove request date column
    tasks = tasks.drop(columns='Request_Date')

    return tasks


# UNUSED: not called anywhere in the active pipeline (legacy alternate of find_AOG_work_packages).
def find_AOG_work_packages_2(df_tasks):

    # Import list of AOG slots
    df_aog = import_df_AOG_mtop()
    # Reduce list to match work package data
    data_start = min(df_tasks['WorkPackage_ActualStartDate'].to_list())
    data_end = max(df_tasks['WorkPackage_ActualEndDate'].to_list())
    df_aog = df_aog[(df_aog['date_start'] >= data_start - timedelta(hours=4)) &
                    (df_aog['date_end'] <= data_end + timedelta(hours=4))]

    # Look for AOG slot corresponding to the considered WP
    # df_tasks = df_tasks.reset_index(drop=True)
    for wp_index, wp_row in df_tasks.iterrows():
        slots_match = find_slot_match_for_wp_2(wp_row, df_aog)
        df_tasks.loc[wp_index, 'AOG_slot'] = slots_match

    df_aog = df_aog.rename(columns={'date_start': 'AOG_date_start',
                                    'date_end': 'AOG_date_end',
                                    'slot_duration': 'AOG_duration',
                                    'aog_type': 'AOG_type'})

    df_tasks = pd.merge(df_tasks, df_aog[['slot_id', 'AOG_date_start', 'AOG_date_end', 'AOG_duration', 'AOG_type']],
                        left_on='AOG_slot', right_on='slot_id', how='left')
    df_tasks = df_tasks.drop(columns=['slot_id'])

    return df_tasks

def find_slot_match_for_wp_2(wp, df_aog):
    wp_regis = wp['WorkPackage_Aircraft'].replace('-','')

    # To find a match, there must be the same registration, and it must be true that
    # (StartA <= EndB)  and  (EndA >= StartB), considering a buffer
    buffer = timedelta(hours=1)
    df_aog_overlap = df_aog[(df_aog['ac_regis'] == wp_regis) &
                            (df_aog['date_start'] - buffer <= wp['WorkPackage_ActualEndDate'] + buffer) &
                            (df_aog['date_end'] + buffer >= wp['WorkPackage_ActualStartDate'] - buffer)]

    # Flag for match found for the work package
    if len(df_aog_overlap) == 0:
        aog_slot_found = np.nan

    else:
        aog_slot_found = df_aog_overlap['slot_id'].to_list()[0]
        # Raise warning if more than one slot found
        if len(df_aog_overlap)>1:
            logging.warning(str(len(df_aog_overlap)) + ' slots found for one work package')

    return aog_slot_found


# =================================================================================#
# DISTRIBUTIONS NON-ROUTINES
# =================================================================================#

def find_distributions_non_routines(df_tasks):

    # Fit NR labor hours
    NR_distributions = NR_fit_distributions(df_tasks['NR'])

    # Find probability of having non routines
    NR_distributions = NR_probability(df_tasks, NR_distributions)

    # Plot the fitted distribution
    plot_fitted_distributions(NR_distributions,'labor_fitted', 'labor_empirical', title='NR labor', x_label='[hours]')

    return NR_distributions


def NR_probability(df_tasks, NR_distributions):
    ''' Find the probability that a non-routine is found in a work package. '''
    # Dataframe of NR labor per workpackage
    df_nr = df_tasks['NR']
    # Dataframe of all hangar workpackages
    df_wp = df_tasks['wp_hangar']

    # Find probability of NR in the workpackage for each fleet type
    fleets = list(set(df_nr['ac_fleet'].to_list()))
    for fleet in fleets:
        # Find data regarding fleet
        df_nr_fleet = df_nr[df_nr['ac_fleet']==fleet]
        df_wp_fleet = df_wp[df_wp['ac_fleet']==fleet]
        # Count wp with and without NR
        wp_with_nr = len(df_nr_fleet)
        wp_all = len(df_wp_fleet)
        # Find probability
        probability_non_routines = wp_with_nr/wp_all
        fleet_dict = next(d for d in NR_distributions if d['fleet']==fleet)
        fleet_dict['probability_NR'] = probability_non_routines

    return NR_distributions

def NR_fit_distributions(df_nr):

    NR_distributions = []
    fleets = list(set(df_nr['ac_fleet'].to_list()))
    for fleet in fleets:
        # Empirical labor data
        data_labor = df_nr[df_nr['ac_fleet'] == fleet]['wp_labor_total'].to_list()
        # Find name
        fit_labor_name = 'NR labor ' + fleet
        fitted_labor = fit_distribution(data_labor, fit_labor_name, bins_width=M.BIN_SIZE_NR)

        fleet_data = {'fleet': fleet,
                      'labor_fitted': fitted_labor,
                      'labor_empirical': data_labor}

        NR_distributions.append(fleet_data)

    return NR_distributions

# =================================================================================#
# DISTRIBUTIONS DEFERRED DEFECTS
# =================================================================================#
def find_distributions_deferred_defects(df_tasks, DDs):
    # Generate dataframe with DDs arrival info
    df_arrivals = DDs_arrivals_dataframe(df_tasks, DDs)
    # Find inter-arrival time
    df_arrivals = DDs_find_inter_arrival_time(df_arrivals)

    # Find the probability of inter arrival times and DD counts
    DDs_distributions = DDs_fit_distributions(df_arrivals)

    # Plot results
    plot_empirical_probabilities(DDs_distributions, 'inter_arrival_time_probabilities', 'inter_arrival_time_empirical',
                                 title='DDs inter-arrival time', x_label='[DD count]', max_allowed=M.MAX_DD_INTER_ARRIVAL)
    plot_empirical_probabilities(DDs_distributions, 'dd_count_probabilities', 'dd_count_empirical',
                                 title='DDs count per day', x_label='[days]',  max_allowed=M.MAX_DD_COUNT)

    # # Plot DDs empirical and fitted distributions
    # plot_fitted_distributions(DDs_distributions, 'inter_arrival_time_fitted', 'inter_arrival_time_empirical',
    #                           title='DD inter-arrival time', x_label='[days]')
    # plot_fitted_distributions(DDs_distributions, 'dd_count_fitted', 'dd_count_empirical',
    #                           title='DD arrival count per day', x_label='[DD count]')

    return DDs_distributions


def DDs_arrivals_dataframe(df_tasks, DDs):
    '''
    Generate dataframe with info about deferred defects arrivals
    - Find the dates when some DD arrived. Only consider the dates within the considered workpackage window
    - Count how many DDs arrived on the found dates
    '''

    df_dd = DDs #df_tasks['DDs']
    # Add column of arrival date (no time)
    df_dd['date_arrival'] = df_dd['date_fault_found'].dt.date

    # Count the tasks arriving each day
    df_arrivals = df_dd.groupby(['date_arrival','ac_regis'])['date_arrival'].count()
    df_arrivals = df_arrivals.reset_index(name='tasks_arrived_count')

    # Remove dates outside of the considered window
    window_start = df_tasks['window_start']
    window_end = df_tasks['window_end']
    df_arrivals = df_arrivals[(df_arrivals['date_arrival'] >= window_start)
                              & (df_arrivals['date_arrival'] <= window_end)]

    # Merge fleet type
    df_aircraft = df_dd[['ac_regis', 'ac_fleet']].drop_duplicates()
    df_arrivals = pd.merge(df_arrivals, df_aircraft, how='left', on='ac_regis')

    return df_arrivals

def DDs_find_inter_arrival_time(df_arrivals):
    ''' Find the inter-arrival time [days] between DDs arrivals'''
    # Order by aircraft and arrival date
    df_arrivals = df_arrivals.sort_values(['ac_fleet', 'ac_regis', 'date_arrival'])
    # Find previous arrival date
    df_arrivals['date_prev_arrival'] = df_arrivals['date_arrival'].shift(1)

    # Remove first value for each aircraft
    df_arrivals['first_arrival_per_ac'] = df_arrivals['date_arrival'].groupby(df_arrivals['ac_regis']).transform(min)
    df_arrivals = df_arrivals[df_arrivals['date_arrival'] != df_arrivals['first_arrival_per_ac']]
    df_arrivals = df_arrivals.drop(columns='first_arrival_per_ac')

    # Compute inter-arrival time and make it into an integer [days]
    df_arrivals['inter_arrival_time'] = df_arrivals['date_arrival'] - df_arrivals['date_prev_arrival']
    df_arrivals['inter_arrival_time'] = round(df_arrivals['inter_arrival_time'].dt.total_seconds()/3600/24)
    df_arrivals['inter_arrival_time'] = df_arrivals['inter_arrival_time'].astype(int)

    return df_arrivals

def DDs_fit_distributions(df_arrivals):
    DDs_distributions = []
    fleets = list(set(df_arrivals['ac_fleet'].to_list()))
    for fleet in fleets:
        data_inter_arrival_time = df_arrivals[df_arrivals['ac_fleet'] == fleet]['inter_arrival_time'].to_list()
        data_dd_count = df_arrivals[df_arrivals['ac_fleet'] == fleet]['tasks_arrived_count'].to_list()

        # Fit DD arrivals count
        dd_count_probability = compute_discrete_probability(data_dd_count, max_value=M.MAX_DD_COUNT)
        dd_inter_arrival_time_probability = compute_discrete_probability(data_inter_arrival_time,
                                                                         max_value=M.MAX_DD_INTER_ARRIVAL)

        fleet_data = {'fleet': fleet,
                      'inter_arrival_time_probabilities': dd_inter_arrival_time_probability,
                      'inter_arrival_time_empirical': data_inter_arrival_time,
                      'dd_count_probabilities': dd_count_probability,
                      'dd_count_empirical': data_dd_count}

        DDs_distributions.append(fleet_data)

    return DDs_distributions

def compute_discrete_probability(data, max_value):
    ''' Compute the empirical probability of observing a variable assuming a certain value '''
    # Sort values
    data.sort()
    # Remove two high values
    data = [dt for dt in data if dt <= max_value]
    # Generate dictionary with items count
    probability_dict = Counter(data)
    # Make count into probability
    probability_dict = {key: count/len(data) for key, count in probability_dict.items()}
    return probability_dict

def plot_empirical_probabilities(data_list, key_probabilities, key_empirical, max_allowed=None, title='', x_label=''):
    # Order data by fleet name for consistent order
    data_list.sort(key=lambda x: x['fleet'])

    n_rows = len(data_list)
    fig, axs = plt.subplots(n_rows, sharex='col', sharey='col')
    plt.suptitle(title)

    for i in range(n_rows):
        # Set title to subfigure
        fleet = data_list[i]['fleet']
        axs[i].set_title(fleet)

        # Find data
        data_probabilities = data_list[i][key_probabilities]
        data_empirical = data_list[i][key_empirical]


        # Histogram
        data_max = ceil(max(data_empirical))
        bins = range(data_max+2)
        # bins = [x-0.5 for x in range(data_max+1)]
        _, _, bars = axs[i].hist(data_empirical, bins=bins, density=True, label='Historical', align='left')
        axs[i].set_xticks(range(data_max+1))
        axs[i].set_yticks([0.2 * x for x in range(5)])

        # Probability over bars
        bar_labels = []
        for bar_key in range(data_max+1):
            if bar_key not in data_probabilities:
                bar_labels.append(str(0))
            else:
                bar_labels.append(str(round(data_probabilities[bar_key],3)))
        axs[i].bar_label(bars, labels=bar_labels, color='tab:orange')

        # Add legend color for probability over bars
        axs[i].hist([50], bins=bins, density=True, label='Computed Weight', align='left')


        # Max allowed
        if max_allowed!=None:
            axs[i].axvline(max_allowed, color='r', linestyle='dashed', linewidth=1, label='Max allowed')

        axs[i].grid(axis='y')

    axs[0].legend()
    axs[n_rows - 1].set(xlabel=x_label)

def generate_deferred_defect_df(df):
    ''' Filter the DDs dataframe and generate a new dataframe for tasks simulation '''
    ##### FILTER DF AND ADJUST MISSING DATA #####
    # Remove DDs for which the work type is not either P or H. Tasks with nan field are also removed under the
    # assumption that they are tasks executable during the turnaround operations
    df = df[(df['task_work_type']=='P') | (df['task_work_type']=='H') | (df['task_work_type']=='PH')]

    # Remove items without due date
    df = df[(pd.isnull(df['date_fault_found'])==0) & (pd.isnull(df['date_due'])==0)]

    # Only keep MEL, NSRE and ADHOC items
    df = df[(df['task_type']=='ADHOC') |
            (df['deferral_class'].str.lower().str.contains('mel')) |
            (df['deferral_class'].str.lower().str.contains('nsre'))]

    # Remove the substitution of the emergency medical kit from the list of tasks (MEL A)
    df = df[~((df['deferral_class'] == 'MEL A') &
          ((df['task_name'].str.lower().str.contains('med')) |
          df['task_name'].str.lower().str.contains('aid') |
          df['task_name'].str.lower().str.contains('emk') |
          df['task_name'].str.lower().str.contains('fak') |
          df['task_name'].str.lower().str.contains('eq')|
          df['task_name'].str.lower().str.contains('doctor')|
          df['task_name'].str.lower().str.contains('eq')|
          df['task_name'].str.lower().str.contains('emergency kit')|
          df['task_name'].str.lower().str.contains('precaution')|
          df['task_name'].str.lower().str.contains('glucose')|
          df['task_name'].str.lower().str.contains('f.a.k')))]

    # When scheduled labor not available, use actual labor instead
    df['labor_sched'] = df['labor_sched'].mask(df['labor_sched']==0, df['labor_act'])

    # Exclude tasks with too long associated labor
    df = df[(df['labor_sched'] < M.EXCLUDE_DD_MAX_LABOR) & (df['labor_act'] < M.EXCLUDE_DD_MAX_LABOR)]

    # Exclude tasks if actual labor much higher than scheduled
    df = df[~((df['labor_sched'] < df['labor_act']/M.EXCLUDE_DD_LABOR_SCHED_ACT_MULTIPL_FACTOR) &
              (df['labor_sched'] > M.EXCLUDE_DD_LABOR_SCHED_ACT_MIN))]

    # Compute deferral days based on arrival and due date
    df['deferral_days'] = (df['date_due'].dt.date - df['date_fault_found'].dt.date).dt.days.astype(int)
    # Remove white spaces from deferral class
    df['deferral_class'] = df['deferral_class'].str.replace(' ','')
    # For MEL and NSRE items with assigned deferral time, change the found value
    for deferral_class, deferral_time in M.DEFERRAL_CLASSES.items():
        df['deferral_days'] = df['deferral_days'].mask(df['deferral_class']==deferral_class, deferral_time)

    # Remove tasks with deferral time higher than selected and set to zero (unless MEL A)
    df = df[df['deferral_days'] <= M.MAX_DEFERRAL_DAYS_DD]
    df = df[(df['deferral_days'] != 0) | (df['deferral_class']=='MELA')]

    ##### GENERATE NEW DATAFRAME FOR DDs SIMULATION #####
    df_sim = df.copy()
    # Drop unnecessary columns
    df_sim = df_sim.drop(columns=['date_fault_found', 'date_due', 'ac_regis'])

    return df, df_sim

# def test_function(x): # TODO remove
#     from scipy.stats import nbinom, poisson, geom
#     x = pd.Series(x)
#     mean = x.mean()
#     var = x.var()
#     likelihoods = {}
#     log_likelihoods = {}
#
#     # Negative Binomial
#     p = 1 - mean / var
#     r = (1 - p) * mean / p
#     likelihoods['nbinom'] = x.map(lambda val: nbinom.pmf(val, r, p)).prod()
#     log_likelihoods['nbinom'] = x.map(lambda val: nbinom.logpmf(val, r, p)).sum()
#
#     # Poisson
#     lambda_ = mean
#     likelihoods['poisson'] = x.map(lambda val: poisson.pmf(val, lambda_)).prod()
#     log_likelihoods['poisson'] = x.map(lambda val: poisson.logpmf(val, lambda_)).sum()
#
#     # Geometric
#     p = 1 / mean
#     likelihoods['geometric'] = x.map(lambda val: geom.pmf(val, p)).prod()
#     log_likelihoods['geometric'] = x.map(lambda val: geom.logpmf(val, p)).sum()
#
#     best_fit = max(likelihoods, key=lambda x: likelihoods[x])
#     print("Best fit:", best_fit)
#     print("Likelihood:", likelihoods[best_fit])
#
#     best_fit2 = max(log_likelihoods, key=lambda x: log_likelihoods[x])
#     print("Best fit:", best_fit2)
#     print("Likelihood:", log_likelihoods[best_fit2])
#     breakpoint()

# =================================================================================#
# AOG SLOTS DISTRIBUTIONS
# =================================================================================#
def find_distributions_maintenance_AOG():

    if M.AOG_DATA_SOURCE == 0:
        # Import dataframe of mtop slots - data up until 2021
        df_aog = import_df_AOG_mtop()
    elif M.AOG_DATA_SOURCE == 1:
        # Import dataframe of AG slots - 2022 data
        df_aog = import_df_AOG_bluelagoon() # TODO old dataframe from BlueLagoon data
    else:
        raise Exception('M.AOG_DATA_SOURCE value not supported')

    # Find inter arrival time between AOG slots
    df_aog = find_inter_arrival_AOG(df_aog)

    # Fit distributions for each fleet type
    AOG_distributions = AOG_fit_distributions(df_aog)
    # Test if data from different fleet can be assumed as one distribution
    KS_results_AOG = AOG_do_KStest(AOG_distributions)

    # Plot distributions
    plot_fitted_distributions(AOG_distributions, 'time_between_AOG_fitted', 'time_between_AOG_empirical',
                              title='AOG slots inter-arrival time', x_label='[days]', bins_width=2)
    plot_fitted_distributions(AOG_distributions, 'AOG_duration_fitted', 'AOG_duration_empirical',
                              title='AOG slots duration', x_label='[hours]', bins_width=2)

    # Save results
    write_pickle(AOG_distributions, 'distributions_AOG')

    plt.show()


def import_df_AOG_mtop():
    ''' Import Mtop maintenance slots dataframe and filter out inactive slots '''
    df_mtop = load_csv(INPUT_FILES.MTOP_SLOTS)

    # only keep last mutation of slot
    df_mtop = df_mtop.sort_values(['SEQNO','MUTDATETIME'])
    df_mtop = df_mtop.drop_duplicates(['SEQNO'], keep='last')

    # Remove slots not assigned to any registration
    df_mtop = df_mtop[df_mtop['ACREGIS']!='DUMMY']

    # Only keep the active slots
    df_mtop = df_mtop[df_mtop['MSTATUS'] == 'A']

    # Remove slots in the future
    df_mtop = df_mtop[df_mtop['STDATETIME']<=dt.today()]


    # Only keep selected slots
    string_slot_names = '|'.join(st for st in M.AOG_INCLUDED_SLOTS_TYPES)
    df_mtop = df_mtop[(df_mtop['KINDMNT'].str.contains(string_slot_names, regex=True))]

    # Add column with fleet type
    df_mtop['ac_fleet'] = ''
    df_mtop['ac_fleet'] = df_mtop['ac_fleet'].mask(((df_mtop['ACSUBTYP'] == '772') | (df_mtop['ACSUBTYP'] == '77W')),'777')
    df_mtop['ac_fleet'] = df_mtop['ac_fleet'].mask(((df_mtop['ACSUBTYP'] == '789') | (df_mtop['ACSUBTYP'] == '781')),'787')
    df_mtop['ac_fleet'] = df_mtop['ac_fleet'].mask(((df_mtop['ACSUBTYP'] == '333') | (df_mtop['ACSUBTYP'] == '332')),'A330')

    # Add slot duration column in hours
    df_mtop['slot_duration'] = df_mtop['ENDATETIME'] - df_mtop['STDATETIME']
    df_mtop['slot_duration'] = df_mtop['slot_duration'].dt.total_seconds() / 3600

    # Reduce dataframe columns
    df_mtop = df_mtop[['SEQNO', 'ACREGIS', 'ac_fleet', 'STDATETIME', 'ENDATETIME',
                     'slot_duration', 'STATION', 'KINDMNT']]
    # Rename columns
    df_mtop = df_mtop.rename(index=str, columns={'SEQNO': 'slot_id',
                                                 'ACREGIS': 'ac_regis',
                                                 'STDATETIME': 'date_start',
                                                 'ENDATETIME': 'date_end',
                                                 'STATION': 'airport',
                                                 'KINDMNT': 'aog_type'})
    return df_mtop





def import_df_AOG_bluelagoon(): # TODO this funciton is no longer used. This data is recent data from Bluelagoon, now Mtop data used
    ''' Import AOG slot dataframe and filter out inactive slots '''
    df_AOG = load_csv(INPUT_FILES.AOG_SLOTS)

    # Drop slot with a deactivation time
    df_AOG = df_AOG[pd.isnull(df_AOG['DeactivationTimeUtc'])]

    # Remove slots without registration
    df_AOG = df_AOG[pd.isnull(df_AOG['Registration'])==0]

    # Only keep selected slots
    string_slot_names = '|'.join(st for st in M.AOG_INCLUDED_SLOTS_TYPES)
    df_AOG = df_AOG[(df_AOG['TypeCode'].str.contains(string_slot_names, regex=True))]

    # Add column with fleet type
    df_AOG['ac_fleet'] = ''
    df_AOG['ac_fleet'] = df_AOG['ac_fleet'].mask(((df_AOG['IataType']=='772')|(df_AOG['IataType']=='77W')),'777')
    df_AOG['ac_fleet'] = df_AOG['ac_fleet'].mask(((df_AOG['IataType']=='789')|(df_AOG['IataType']=='781')),'787')
    df_AOG['ac_fleet'] = df_AOG['ac_fleet'].mask(((df_AOG['IataType']=='333')|(df_AOG['IataType']=='332')),'A330')

    # Add slot duration column in hours
    df_AOG['slot_duration'] = df_AOG['PlannedEndTimeUtc'] - df_AOG['PlannedStartTimeUtc']
    df_AOG['slot_duration'] = df_AOG['slot_duration'].dt.total_seconds() / 3600
    # df_AOG['slot_duration'] = df_AOG['slot_duration'].astype(int)

    # Remove data points that last longer than the maximum considered duration (only few outliers removed)
    # df_AOG = df_AOG[df_AOG['slot_duration']/24 < M.MAX_DURATION_AOG]

    # Reduce dataframe columns
    df_AOG = df_AOG[['Id', 'Registration', 'ac_fleet', 'PlannedStartTimeUtc', 'PlannedEndTimeUtc',
                     'slot_duration', 'Airport', 'TypeCode']]
    # Rename columns
    df_AOG = df_AOG.rename(index=str, columns={'Id': 'slot_id',
                                               'Registration': 'ac_regis',
                                               'PlannedStartTimeUtc': 'date_start',
                                               'PlannedEndTimeUtc': 'date_end',
                                               'Airport': 'airport',
                                               'TypeCode': 'aog_type'})

    return df_AOG

def find_inter_arrival_AOG(df_AOG):
    ''' Find inter-arrival time between AOG slots '''

    # Find end date of previous AOG slot
    df_AOG = df_AOG.sort_values(['ac_fleet','ac_regis','date_start'])
    df_AOG['date_prev_slot_end'] = df_AOG['date_end'].shift(1)

    # Remove first value for each aircraft
    df_AOG['first_slot_for_ac'] = df_AOG['date_start'].groupby(df_AOG['ac_regis']).transform(min)
    df_AOG = df_AOG[df_AOG['date_start'] != df_AOG['first_slot_for_ac']]
    df_AOG = df_AOG.drop(columns='first_slot_for_ac')

    # Find between arrivals in hours
    df_AOG['time_between_slots'] = df_AOG['date_start'] - df_AOG['date_prev_slot_end']
    df_AOG['time_between_slots'] = df_AOG['time_between_slots'].dt.total_seconds()/3600/24
    # df_AOG['time_between_slots'] = df_AOG['time_between_slots'].astype(int)
    df_AOG = df_AOG.drop(columns='date_prev_slot_end')

    return df_AOG


def AOG_fit_distributions(df_AOG):
    AOG_distributions = []
    fleets = list(set(df_AOG['ac_fleet'].to_list()))
    for fleet in fleets:
        data_inter_arrival_AOG = df_AOG[df_AOG['ac_fleet']==fleet]['time_between_slots'].to_list()
        data_AOG_duration = df_AOG[df_AOG['ac_fleet']==fleet]['slot_duration'].to_list()

        fit_name_inter_arrival_AOG = 'Time between slots '+fleet
        fitted_inter_arrival_AOG = fit_distribution(data_inter_arrival_AOG, fit_name_inter_arrival_AOG,
                                                    bins_width=M.BIN_SIZE_AOG_INTER_ARRIVAL, distr='expon')

        fit_name_AOG_duration = 'AOG duration '+fleet
        fitted_AOG_duration = fit_distribution(data_AOG_duration, fit_name_AOG_duration,
                                               bins_width=M.BIN_SIZE_AOG_DURATION)

        fleet_data = {'fleet': fleet,
                      'time_between_AOG_fitted': fitted_inter_arrival_AOG,
                      'time_between_AOG_empirical': data_inter_arrival_AOG,
                      'AOG_duration_fitted': fitted_AOG_duration,
                      'AOG_duration_empirical': data_AOG_duration}
        AOG_distributions.append(fleet_data)

    return AOG_distributions


def AOG_do_KStest(AOG_distributions):
    KS_results = {}

    # Find combinations of fleets to compare
    combinations = []
    fleets = [x['fleet'] for x in AOG_distributions]
    for fleet1 in fleets:
        for fleet2 in fleets:
            combination = set([fleet1, fleet2])
            if fleet1!=fleet2 and combination not in combinations:
                combinations.append(combination)
    combinations = [list(x) for x in combinations]

    data_to_test_keys = ['time_between_AOG_empirical', 'AOG_duration_empirical']
    for key in data_to_test_keys:
        for combination in combinations:
            dict1 = next(x for x in AOG_distributions if x['fleet'] == combination[0])
            dict2 = next(x for x in AOG_distributions if x['fleet'] == combination[1])

            test_id = key + '-' + dict1['fleet'] + '-' + dict2['fleet']
            KS = stats.kstest(dict1[key], dict2[key])
            KS_results[test_id] = KS

    return KS_results

def fit_distribution(data, name, bins_width=1, distr='popular', method='parametric'):
    ''' Gven a list of historical data, fits an empirical distirbution and returns a distfit object'''
    # Initialize distfit
    dist = distfit(todf=True, distr=distr, method=method)

    # Find bins to group empirically observed delays
    min_data = min(data)
    max_data = max(data)
    n_bins = round((max_data - min_data + 1)/bins_width)
    dist.bins = n_bins

    # Determine best-fitting probability distribution for data
    dist.fit_transform(np.array(data))

    #Print summary and plot obtained fitting
    print('\n\n################# DISRUPTION LEVEL: ', name, ' #################')
    print(dist.summary)
    dist.plot()
    dist.plot_summary()

    return dist

def plot_fitted_distributions(data_list, key_fitted, key_empirical, title='', x_label='', bins_width=1):
    '''
    Plot histograms based on historical data and fitted distribution.
    :param data_list: List of dictionaries: {'fleet': string,
                                            key_fitted: distfit object,
                                            key_empirical: list of empirical data}

    :param title: title for the figure
    :param x_label: unit for the x axis of the plot
    '''
    # Order data by fleet name for consistent order
    data_list.sort(key=lambda x: x['fleet'])

    # ''' Plot histograms based on historical data and fitted distribution.'''
    n_rows = len(data_list)
    fig, axs = plt.subplots(n_rows, sharex='col', sharey='col')
    plt.suptitle(title)


    for i in range(n_rows):
        # Set title to subfigure
        fleet = data_list[i]['fleet']
        fitted_distr_name = data_list[i][key_fitted].model['name']
        axs[i].set_title(fleet+ ' - ' +fitted_distr_name)

        # Find data
        data_fitted = data_list[i][key_fitted]
        data_empirical = data_list[i][key_empirical]
        data_min = floor(min(data_empirical))
        data_max = ceil(max(data_empirical))

        # Histogram
        bins = range(data_min, data_max, bins_width)
        axs[i].hist(data_empirical, bins=bins, density=True, label='Historical',histtype='step')

        # Historical mean
        # axs[i].axvline(mean(data_empirical), color='r', linestyle='dashed', linewidth=1)

        # Fitted distribution
        x = np.linspace(data_min, data_max ,(data_max-data_min)*10)
        # x = np.array(range(data_min, data_max))
        y = data_fitted.model['distr'].pdf(x,
                                           *data_fitted.model['arg'],
                                           loc=data_fitted.model['loc'],
                                           scale=data_fitted.model['scale'])
        axs[i].plot(x, y, label='Fitted')
        axs[i].grid()


    axs[0].legend()
    axs[n_rows-1].set(xlabel=x_label)

