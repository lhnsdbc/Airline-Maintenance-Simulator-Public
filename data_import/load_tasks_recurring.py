from data_import.input_output import load_csv
import numpy as np
import pandas as pd
from config import G, INPUT_FILES

def load_requirements(df_aircraft):
    # Import real maintenance requirements or dummy ones
    if G.REQUIREMENTS_TYPE == 0:
        file_req = INPUT_FILES.ENGINEERING_REQUIREMENTS_INTERVAL
        file_decimal = ','
    elif G.REQUIREMENTS_TYPE == 1:
        file_req = INPUT_FILES.ENGINEERING_REQUIREMENTS_INTERVAL_DUMMY
        file_decimal = '.'
    else:
        raise Exception('REQUIREMENT_TYPE not accepted')

    requirements = load_csv(file_req, low_memory=False, decimal=file_decimal)

    # If real requirements used, process the file
    if G.REQUIREMENTS_TYPE == 0:
        # Import dataframe of blocks
        blocks = load_csv(INPUT_FILES.ENGINEERING_BLOCKS_INTERVAL, low_memory=False, decimal=',')
        blocks = blocks_adjust_columns(blocks, requirements)
        # blocks = process_real_requirements(blocks, df_aircraft)
        # Concatenate requirements and blocks
        requirements = pd.concat([requirements,blocks])
        # Elaborate full dataframe
        requirements = process_real_requirements(requirements, df_aircraft)

    elif G.REQUIREMENTS_TYPE == 1:
        requirements['Requirement_Assembly'] = requirements['Requirement_Assembly'].apply(str)
        requirements['Assembly_short'] = requirements['Assembly_short'].apply(str)

    return requirements


def blocks_adjust_columns(blocks, requirements):
    ''' This function make the blocks dataframe equevalent to the requirement dataframe to allow elaboration through
    the same functions'''
    # Change names of columns to match names or requirements columns
    columns_block = [cl for cl in blocks.columns if 'Block' in cl]
    columns_new_names = []
    for column in columns_block:
        column_new_name = column.replace('Block', 'Requirement')
        columns_new_names.append(column_new_name)
    blocks = blocks.rename(columns = dict(zip(columns_block, columns_new_names)))

    # Add empty columns when needed
    blocks['Requirement_Description'] = 'Block'
    columns_to_add = [cl for cl in requirements.columns if cl not in blocks.columns]
    for column in columns_to_add:
        blocks[column] = np.nan

    return blocks

def process_real_requirements(requirements, df_aircraft):
    # Change data type to some columns
    requirements = change_types(requirements)
    # Only keep ICA fleet requirements, maintenance lower than A-checks and last revisions
    requirements = requirements_filter_rows(requirements)
    # Elaborate data on aircraft types
    requirements = requirements_aircraft_types(requirements, df_aircraft)
    # Determine the requirement duration and workforce
    requirements = requirements_labor_and_duration(requirements)
    # Convert the calendar requirement columns
    requirements = convert_calendar_days(requirements)
    # Rename columns
    requirements = rename_columns(requirements)
    # Adjust for engineering requirements with 0 as interval
    requirements = remove_requirement_for_interval(requirements)
    # Remove specific requirements (ex:disinfestation)
    requirements = remove_specific_requirement_codes(requirements)
    # Add assembly short name
    requirements = add_assembly_short_name(requirements)
    # Remove specific requirements
    requirements = remove_specific_requirements(requirements)

    return requirements

def change_types(Engineering_Req):
    ''' Change types to some columns that would cause issues otherwise'''
    # Make assembly column into string, if not already applied
    Engineering_Req['Requirement_Assembly'] = Engineering_Req['Requirement_Assembly'].apply(str)
    # Make some columns float
    columns_float = ['JobCard_Total_Sched_Hours', 'JobCard_EstimatedDuration',
                     'SchedRules_RepeatInterval_AcHours', 'SchedRules_RepeatInterval_AcCycles',
                     'SchedRules_RepeatInterval_CalDay',
                     'SchedRules_RepeatInterval_CalHour', 'SchedRules_RepeatInterval_CalWeek',
                     'SchedRules_RepeatInterval_CalMonth',
                     'SchedRules_RepeatInterval_CalYear', 'SchedRules_RepeatInterval_CalLastDayMonth']
    Engineering_Req[columns_float] = Engineering_Req[columns_float].astype(float)

    return Engineering_Req

def requirements_filter_rows(Engineering_Req):
    '''Reduce the dataframe to the selected long-haul fleet and tasks lower than A-checks.'''

    # Remove old revisions
    Engineering_Req['Max_Revision'] = Engineering_Req.groupby(['Requirement_Assembly', 'Requirement_Code'])[
        'Requirement_Rev_Num'].transform('max')
    Engineering_Req = Engineering_Req[Engineering_Req['Requirement_Rev_Num'] == Engineering_Req['Max_Revision']]
    Engineering_Req = Engineering_Req.drop(columns=['Max_Revision'])

    # Reset index and order elements
    Engineering_Req = Engineering_Req.sort_values(by=['Requirement_Assembly', 'Requirement_Code',
                                                      'Requirement_Aircraft_RegistrationCode'],ignore_index=True)

    return Engineering_Req


def requirements_aircraft_types(Engineering_Req, df_aircraft):
    '''
    Filter out requirements that are not assigned to a minimum requested number of aircraft within the subtype:
    - Find the subtype of each registration
    - Determine to how many aircraft per subtype a requirement is assigned
    - Requirements are assigned to a subtype only if assigned to a minimum number of aircraft within that subtype
    - Filter requirement df by keeping one row per JIC
    '''
    ##### REGISTRATIONS IN REQUIREMENTS DF #####
    # Rename registration column
    Engineering_Req = Engineering_Req.rename(columns={'Requirement_Aircraft_RegistrationCode': 'Registration'})
    # Remove dash from registration columns
    Engineering_Req['Registration'] = Engineering_Req['Registration'].str.replace('-','')
    # Remove rows assigned to aircraft not in use
    Engineering_Req = Engineering_Req[Engineering_Req['Registration'].isin(df_aircraft['AircraftRegistrationFull'])]
    # Find registration part that corresponds to aircraft subtype
    Engineering_Req['Regis_reduced'] = Engineering_Req['Registration'].str[0:4]

    ##### AIRCAFT SUBTYPES INFO #####
    # Reduce original df
    aircraft_reduced = df_aircraft[['AircraftRegistrationFull','AircraftTypeCodeIATA']]
    aircraft_reduced = aircraft_reduced.rename(columns={'AircraftRegistrationFull': 'Subtype_regis',
                                                       'AircraftTypeCodeIATA': 'Subtype'})
    # Find part of registration that corresponds to ac subtype
    aircraft_reduced['Subtype_regis'] = aircraft_reduced['Subtype_regis'].str[0:4]
    # Count the aircraft in subtype
    aircraft_reduced['Subtype_count'] = aircraft_reduced['Subtype'].groupby(aircraft_reduced['Subtype']).transform('count')
    # Only keep one row per subtype
    aircraft_reduced = aircraft_reduced.drop_duplicates()

    ##### MERGE SUBTYPE AND SUBTYPE COUNT TO THE REQUIREMENTS DF
    Engineering_Req = pd.merge(Engineering_Req, aircraft_reduced, how='left',
                               left_on='Regis_reduced', right_on='Subtype_regis')
    Engineering_Req = Engineering_Req.drop(columns=['Regis_reduced', 'Subtype_regis'])

    ##### LIST OF REGISTRATIONS AND SUBTYPES ASSIGNED TO AN AIRCRAFT #####
    # Generate reduced df with one row per requirement-aircraft
    requirements_reduced = Engineering_Req.drop_duplicates(['Requirement_Code', 'Registration'])
    # Count how many ac of same subtype are assigned the requirement
    requirements_reduced['Req_subtype_count'] = requirements_reduced['Subtype']\
                                            .groupby([requirements_reduced['Subtype'], requirements_reduced['Requirement_Code']])\
                                            .transform('count')
    # Only keep relevant columns and drop duplicated rows
    requirements_reduced = requirements_reduced[['Requirement_Code', 'Subtype', 'Requirement_Assembly',
                                                 'Subtype_count', 'Req_subtype_count']]
    requirements_reduced = requirements_reduced.drop_duplicates()
    # Only keep requirements rows for which the requirement is assigned a minimum number of times witing the ac subtype
    requirements_reduced = requirements_reduced[requirements_reduced['Req_subtype_count']
                                                >= requirements_reduced['Subtype_count'] * G.REQUIREMENTS_FRACTION_OF_AIRCRAFT]


    # Find dataframe with list of subtypes to which requirement is assigned
    requirements_subtypes = requirements_reduced.groupby(['Requirement_Code', 'Requirement_Assembly'])\
                            ['Subtype'].apply(list).reset_index()
    requirements_subtypes = requirements_subtypes.rename(index=str, columns={'Subtype':'Subtypes'})

    # Drop unnecessary columns and duplicates from reduced df
    requirements_reduced = requirements_reduced.drop(columns=['Req_subtype_count', 'Subtype_count', 'Subtype'])
    requirements_reduced = requirements_reduced.drop_duplicates()

    # Merge list of subtypes
    requirements_reduced = pd.merge(requirements_reduced, requirements_subtypes, how='left',
                                    on=['Requirement_Code', 'Requirement_Assembly'])


    ##### MERGE RESULTS TO ORIGNAL DATAFRAME #####
    # Only keep selecred requirements
    Engineering_Req = pd.merge(Engineering_Req, requirements_reduced, how='right',
                               on=['Requirement_Code', 'Requirement_Assembly'])
    # Drop unnecessary columns
    Engineering_Req = Engineering_Req.drop(columns=['Registration', 'Subtype', 'Subtype_count'])
    # Only keep one row per JIC
    Engineering_Req = Engineering_Req.drop_duplicates(['Requirement_Code', 'Requirement_Assembly','JobCard_Code'])

    return Engineering_Req


def requirements_labor_and_duration(Engineering_Req):
    '''
    Determine the duration and workforce requirements for each requirements based on the underlying jobcards
    '''

    # Scheduled labor as sum of job cards labor
    Engineering_Req['Requirement_total_labor'] = Engineering_Req.groupby(['Requirement_Assembly', 'Requirement_Code']) \
                                                ['JobCard_Total_Sched_Hours'].transform(sum)

    # Duration as max duration
    Engineering_Req['Requirement_duration'] = Engineering_Req.groupby(['Requirement_Assembly', 'Requirement_Code']) \
        ['JobCard_EstimatedDuration'].transform(max)

    # Only keep one row per requirement
    Engineering_Req = Engineering_Req.drop_duplicates(['Requirement_Assembly','Requirement_Code'])

    # Remove rows for which labor hours are set to zero
    Engineering_Req = Engineering_Req[Engineering_Req['Requirement_total_labor']>0]

    # Remove rows not necessary anymore
    Engineering_Req = Engineering_Req.drop(columns=['Requirement_EstimatedDuration', 'Requirement_Total_Sched_Hours',
                                                    'JobCard_Total_Sched_Hours', 'JobCard_EstimatedDuration',
                                                    'JobCard_Total_Sched_Hours', 'JobCard_EstimatedDuration',
                                                    'JobCard', 'JobCard_Code'
                                                    ])

    return Engineering_Req


def convert_calendar_days(Engineering_Req):
    # Replace NaN values in Calday column
    Engineering_Req['SchedRules_RepeatInterval_CalDay'] = Engineering_Req['SchedRules_RepeatInterval_CalDay']. \
        mask(Engineering_Req['SchedRules_RepeatInterval_CalDay'].isna(), 0)

    # Convert hours to caldays
    Engineering_Req['SchedRules_RepeatInterval_CalDay'] = Engineering_Req['SchedRules_RepeatInterval_CalDay']. \
        mask(~Engineering_Req['SchedRules_RepeatInterval_CalHour'].isna(),
             Engineering_Req['SchedRules_RepeatInterval_CalDay'] + Engineering_Req[
                 'SchedRules_RepeatInterval_CalHour'] * (1 / 24))

    # Convert week to caldays
    Engineering_Req['SchedRules_RepeatInterval_CalDay'] = Engineering_Req['SchedRules_RepeatInterval_CalDay']. \
        mask(~Engineering_Req['SchedRules_RepeatInterval_CalWeek'].isna(),
             Engineering_Req['SchedRules_RepeatInterval_CalDay'] + Engineering_Req[
                 'SchedRules_RepeatInterval_CalWeek'] * 7)

    # Convert year to caldays (365 days)
    Engineering_Req['SchedRules_RepeatInterval_CalDay'] = Engineering_Req['SchedRules_RepeatInterval_CalDay']. \
        mask(~Engineering_Req['SchedRules_RepeatInterval_CalYear'].isna(),
             Engineering_Req['SchedRules_RepeatInterval_CalDay'] + Engineering_Req[
                 'SchedRules_RepeatInterval_CalYear'] * 365)

    # Convert month to caldays
    Engineering_Req['SchedRules_RepeatInterval_CalDay'] = Engineering_Req['SchedRules_RepeatInterval_CalDay']. \
        mask(~Engineering_Req['SchedRules_RepeatInterval_CalMonth'].isna(),
             Engineering_Req['SchedRules_RepeatInterval_CalDay'] + Engineering_Req[
                 'SchedRules_RepeatInterval_CalMonth'] * 30.436875)

    # Convert last day of the month to caldays
    Engineering_Req['SchedRules_RepeatInterval_CalDay'] = Engineering_Req['SchedRules_RepeatInterval_CalDay']. \
        mask(~Engineering_Req['SchedRules_RepeatInterval_CalLastDayMonth'].isna(),
             Engineering_Req['SchedRules_RepeatInterval_CalDay'] + Engineering_Req[
                 'SchedRules_RepeatInterval_CalLastDayMonth'] * 30.436875)

    # if there is no calendar requirement change to None
    Engineering_Req['SchedRules_RepeatInterval_CalDay'] = Engineering_Req['SchedRules_RepeatInterval_CalDay']. \
        mask(Engineering_Req['SchedRules_RepeatInterval_CalDay'] == 0, np.nan)

    # Drop the remaining cal interval columns
    Engineering_Req = Engineering_Req.drop(columns=['SchedRules_RepeatInterval_CalHour',
                                                    'SchedRules_RepeatInterval_CalWeek',
                                                    'SchedRules_RepeatInterval_CalYear',
                                                    'SchedRules_RepeatInterval_CalMonth',
                                                    'SchedRules_RepeatInterval_CalLastDayMonth'])

    return Engineering_Req



def remove_requirement_for_interval(Engineering_Req):
    """
    Remove requirements for which an interval is :
    - not available
    - longer than a certain value
    - shorter than a certain value
    """
    # Remove requirements with no interval
    Engineering_Req = Engineering_Req[~((pd.isnull(Engineering_Req['FH_task']))
                                      & (pd.isnull(Engineering_Req['FC_task']))
                                      & (pd.isnull(Engineering_Req['Cal_task'])))]

    # Find most stringent interval and only evaluate that it for excluding tasks
    utilization = next(util for util in G.AIRCRAFT_UTILIZATION if util['season']=='summer')
    Engineering_Req['CD_from_FH'] = Engineering_Req['FH_task']/utilization['FH']
    Engineering_Req['CD_from_FC'] = Engineering_Req['FC_task']/utilization['FC']
    Engineering_Req['interval_min'] = Engineering_Req[['CD_from_FC', 'CD_from_FH', 'Cal_task']].min(axis=1)
    # Remove requirements with too long interval
    Engineering_Req = Engineering_Req[(Engineering_Req['interval_min'] >= G.REQUIREMENTS_MIN_INTERVAL_ALLOWED) &
                                      (Engineering_Req['interval_min'] <= G.REQUIREMENTS_MAX_INTERVAL_ALLOWED)]


    Engineering_Req = Engineering_Req.reset_index(drop=True)


    return Engineering_Req

def remove_specific_requirement_codes(requirements):
    ''' Exclude requirements in specified list '''
    requirements = requirements[~requirements['Requirement_Code'].isin(G.REQUIREMENTS_EXCLUDE)]

    return requirements

def rename_columns(Engineering_Req):
    # Remove withspaces
    Engineering_Req['Requirement_WorkType'] = Engineering_Req['Requirement_WorkType'].str.strip()
    # Rename columns
    Rename = {'Requirement_Description': 'Description',
              'SchedRules_RepeatInterval_AcHours': 'FH_task',
              'SchedRules_RepeatInterval_AcCycles': 'FC_task',
              'SchedRules_RepeatInterval_CalDay': 'Cal_task',
              'Requirement_Class': 'Class',
              'Requirement_Subclass': ' Subclass',
              'JobCard_Total_Sched_Hours': 'Workforce',
              'JobCard_EstimatedDuration': 'Duration'
              }
    Engineering_Req = Engineering_Req.rename(columns=Rename)

    return Engineering_Req


def add_assembly_short_name(Engineering_Req):
    ''' Add reduce name for the aircraft types '''
    Engineering_Req['Assembly_short'] = ''
    Engineering_Req['Assembly_short'] = Engineering_Req['Assembly_short'].mask(Engineering_Req['Requirement_Assembly']=='777', '77')
    Engineering_Req['Assembly_short'] = Engineering_Req['Assembly_short'].mask(Engineering_Req['Requirement_Assembly']=='787', '78')
    Engineering_Req['Assembly_short'] = Engineering_Req['Assembly_short'].mask(Engineering_Req['Requirement_Assembly']=='A330', '33')

    return Engineering_Req

def remove_specific_requirements(requirements):
    ''' Remove some requirements that should not be in the list '''
    # Fill nan values with empty cell
    requirements['Description'] = requirements['Description'].fillna('')

    # Remove requirments that are one-off return to operations before or after parking for a long time
    requirements = requirements[~requirements['Description'].str.lower().str.contains('return to operation')]
    requirements = requirements[~requirements['Description'].str.lower().str.contains('storage')]

    return requirements
