from data_import.input_output import load_csv, write_pickle, read_pickle, \
    dataframe_from_blueLagoon_query, detect_date_columns, load_excel
from data_import.load_tasks_recurring import load_requirements
from config import G, INPUT_FILES,directories, MODULES
import pandas as pd
from datetime import timedelta, datetime
import logging
import os
from output.output_functions import log_info, log_warning, log_error


class Data:
    ''' Class that includes all dataframes of input data '''
    def __init__(self, scenario_id):
        # Empty list to which warnings can be added
        self.warnings = []
        # Scenario
        self.scenario = self.__load_scenario(scenario_id)

        # Network
        self.aircraft = self.__load_aircraft()
        self.aircraft_additional = self.__load_aircraft_additional()
        self.flights_duration = self.__load_flights_durations() # Processed distribution but needed for rotations import
        self.rotations = self.__load_rotations()
        self.airports = self.__load_airports()
        self.subtypes = self.__load_subtypes()

        # Maintenance
        self.slots_AChecks_historical = self.__load_AChecks_historical()
        self.slotsNorm = self.__load_slotsNorm()
        self.requirements = load_requirements(self.aircraft)
        self.DDs = self.__load_DDs()

        # Preprocessed distributions
        self.TAT = self.__load_TAT()
        self.disruptions_hub = self.__load_disruptions_hub()
        self.disruptions_outstations = self.__load_disruptions_outstations()
        self.distributions_AOG = self.__load_distributions_AOG()
        self.distributions_NR = self.__load_distributions_NR()
        self.distributions_DDs = self.__load_distributions_DDs()

    @staticmethod
    def __colum_to_list_of_strings(column):
        # Convert empty values to empty string
        column = column.mask(pd.isnull(column), '')
        # Make column into string and make a list
        column = column.apply(str)
        # Make sure there are no white spaces
        column = column.str.replace(' ', '')
        column = column.str.split(',')
        # Remove empty strings from list
        column = column.apply(lambda row:[el for el in row if el != ''])
        return column

    def __load_scenario(self, scenario_id):
        ''' Load the right scenario to be simulated'''
        # Import excel
        scenarios = load_excel(INPUT_FILES.SCENARIOS_SIMULATION)
        # Filter chosen scenario
        scenario = scenarios[scenarios['Id']==scenario_id]
        # scenario = scenarios[scenarios['Id']==G.SCENARIO_SIMULATION]

        # Check that valid scenario found
        if len(scenario) == 0:
            raise Exception('Simulation scenario not found')
        elif len(scenario) > 1:
            raise Exception('More than one simulation scenario with the same id')

        # Make relevant columns into strings
        scenario['Id'] = scenario['Id'].apply(str)
        scenario['Rotations_start'] = scenario['Rotations_start'].apply(str)
        scenario['Slotsnorm_scenario'] = scenario['Slotsnorm_scenario'].apply(str)
        if pd.isnull(scenario['AOG_distr'].iloc[0])==0:
            scenario['AOG_distr'] = scenario['AOG_distr'].apply(str)

        # Make relevant columns into lists of strings
        col_to_list_of_str = ['Aircraft_types', 'Aircraft_remove', 'Aircraft_additional']
        for col in col_to_list_of_str:
            scenario[col] = self.__colum_to_list_of_strings(scenario[col])

        # Simulation start in UTC time
        simulation_start = G.TIMEZONE_UTC.localize(datetime.strptime(scenario['Rotations_start'].iloc[0], '%Y-%m-%d'))

        # Parameters for maintenance module version 2. Note that the module must be correctly selected in the config
        # file, if any of these parameters are specified
        maint_sched_constr_clean = scenario['maint_sched_constr_clean'].iloc[0]
        maint_sched_constr_wp_anticipation = scenario['maint_sched_constr_wp_anticipation'].iloc[0]
        clean_target = scenario['clean_target'].iloc[0]
        wp_anticipation_target = scenario['wp_anticipation_target'].iloc[0]
        # Parameters must be specified for maintenance scheduler in version 2
        if MODULES.MAINTENANCE_SCHEDULE == 2 and \
            (pd.isnull(maint_sched_constr_clean) or pd.isnull(maint_sched_constr_wp_anticipation)):
            raise Exception('Health parameters must be specified when MODULES.MAINTENANCE_SCHEDULE==2 is set in config')

        mas_clean_target = int(clean_target) if pd.isnull(clean_target) == 0 else 0
        mas_wp_anticipation = int(wp_anticipation_target) if pd.isnull(wp_anticipation_target) == 0 else 0

        scenario_dict = {'Id': scenario['Id'].iloc[0],
                         'Rotations_start': scenario['Rotations_start'].iloc[0],
                         'Slotsnorm_scenario': scenario['Slotsnorm_scenario'].iloc[0],
                         'Aircraft_types': scenario['Aircraft_types'].iloc[0],
                         'Aircraft_remove': scenario['Aircraft_remove'].iloc[0],
                         'Aircraft_additional': scenario['Aircraft_additional'].iloc[0],
                         'Reserves_per_day': self.__reserves_override(scenario['Reserves_per_day'].iloc[0]),
                         'AOG_distr': scenario['AOG_distr'].iloc[0],
                         'Simulation_start': simulation_start,
                         'MAS_constr_clean': maint_sched_constr_clean,
                         'MAS_constr_wp_anticipation': maint_sched_constr_wp_anticipation,
                         'MAS_clean_target': mas_clean_target,
                         'MAS_wp_anticipation': mas_wp_anticipation}

        return scenario_dict

    @staticmethod
    def __reserves_override(default_value):
        ''' PAPER_DESIGN Exp. E: allow the experiment runner to override Reserves_per_day via
        the ANEMOS_RESERVES_PER_DAY environment variable (subprocess-isolated). '''
        import os
        override = os.environ.get('ANEMOS_RESERVES_PER_DAY')
        return int(override) if override is not None else default_value

    def __load_aircraft(self):
        ''' Returns dataframe with aircraft registrations data '''
        df = load_csv(INPUT_FILES.AIRCRAFT_REGISTRATIONS, encoding='ISO-8859-1')

        # [FIX] Force columns to string before filtering/processing
        df['AircraftTypeCodeIATA'] = df['AircraftTypeCodeIATA'].astype(str)
        # [FIX] Force SubfleetDetailCode to string to prevent TypeError in concatenation
        if 'SubfleetDetailCode' in df.columns:
            df['SubfleetDetailCode'] = df['SubfleetDetailCode'].astype(str)

        # Only keep ICA aircraft
        df = df[df['AircraftTypeCodeIATA'].isin(G.ICA_SUBTYPES)]
        # Keep one row per aircraft
        df = df.sort_values(['AircraftRegistrationFull', 'RegistrationStartDate'], ignore_index=True)
        df = df.drop_duplicates('AircraftRegistrationFull',keep='last', ignore_index=True)
        # Remove aircraft not in use anymore
        df = df[pd.isnull(df['RegistrationEndDate'])]
        return df

    def __load_aircraft_additional(self):
        ''' Returns dataframe with data on additional simulated aircraft'''
        df = load_csv(INPUT_FILES.AIRCRAFT_ADDITIONAL_REGISTRATIONS)
        # Make type column into string type
        df['AircraftTypeCodeIATA'] = df['AircraftTypeCodeIATA'].astype(str)
        # [FIX] Force SubfleetDetailCode to string here as well
        if 'SubfleetDetailCode' in df.columns:
            df['SubfleetDetailCode'] = df['SubfleetDetailCode'].astype(str)
        return df


    def __load_airports(self):
        ''' Returns dataframe with airport data '''
        df_airports = load_csv(INPUT_FILES.AIRPORTS)
        df_tz = load_csv(INPUT_FILES.TIMEZONES)
        df_airports_coordinates = load_csv(INPUT_FILES.AIRPORTS_COORDINATES)

        # Only keep one row per airport in both dataframes
        df_airports = self.df_drop_duplicates_keep_newest(df_airports,['IcaoAirportCode','IataAirportCode'],'DateUntil')
        df_tz = self.df_drop_duplicates_keep_newest(df_tz,'AirportCode')

        # Only keep some relevant columns in the coordinates dataframe
        df_airports_coordinates = df_airports_coordinates[['AirportICAO', 'Latitude', 'Longitude']]

        # Merge the dataframes
        df = pd.merge(df_airports, df_tz, left_on='IataAirportCode', right_on='AirportCode', how='left')
        df = pd.merge(df, df_airports_coordinates, left_on='IcaoAirportCode', right_on='AirportICAO', how='left')

        # drop double columns
        df = df.drop(['AirportCode','AirportICAO'],axis=1)
        return df


    def __load_rotations(self):
        '''
        Returns dataframe with rotations included in the schedule to simulate. The schedule can be imported in the
        form of a query to BlueLagoon or from a file saved locally. Whenever a new query is run, a local file is
        generated with the final schedule. Note that if the query was run before, the previously generated schedules
        are not overwritten.
        '''
        if G.SCHEDULE_IMPORT_FROM == 0:
            df_rotations = self.__load_rotations_from_BlueLagoon()
        elif G.SCHEDULE_IMPORT_FROM == 1:
            df_rotations = self.__load_rotations_from_local_file()
        else:
            raise Exception('G.SCHEDULE_IMPORT_TYPE value not supported')

        return df_rotations


    def __find_schedule_file_name(self):
        ''' Given the config file, returns the name of a the desired schedule '''
        # Find start date of the schedule after transforming it into datetime to ensure unique format
        schedule_start = datetime.strptime(self.scenario['Rotations_start'], "%Y-%m-%d")
        schedule_start_text = schedule_start.strftime("%Y-%m-%d")
        schedule_file_name = 'schedule_'+schedule_start_text+'_'+str(G.ROTATIONS_WEEKS)+'weeks'
        return schedule_file_name


    def __load_rotations_from_local_file(self):
        ''' Load a schedule from local pickle file '''
        # Find name of file
        schedule_file_name = self.__find_schedule_file_name()
        # Import file. If file not found, raise exception
        try:
            df_schedule = read_pickle(filename=schedule_file_name, directory=directories.schedules)

        except:
            raise Exception('Schedule not found in local files')

        return df_schedule

    @staticmethod
    def __ask_input_flight_changes(flight_to_fix, change):
        '''Requests input for a flight departure (change=='dep'), arrival(change=='arr'), or id (change=='id')'''

        # Find string to print and column to change
        text_flight = 'Flight ' + flight_to_fix
        if change == 'dep':
            string_to_print = '\n' + text_flight + " departure UTC (N/YYYY-mm-dd HH:MM)"
        elif change == 'arr':
            text_flight_empty = ''.ljust(len(text_flight))
            string_to_print = text_flight_empty + " arrival UTC   (YYYY-mm-dd HH:MM)"
        elif change == 'id':
            text_flight_empty = ''.ljust(len(text_flight))
            string_to_print = text_flight_empty + " new id "
        else:
            raise Exception('arr_dep argument not supported')

        # Request input
        input_ok = False
        while input_ok == False:
            # Ask for input
            flight_change_input = input(string_to_print)

            # If 'n' or 'N' is input, don't do anything
            if flight_change_input.lower() == 'n':
                return False
            # If id asked, return id
            elif change == 'id':
                return flight_change_input

            # If other input
            try:
                flight_change_input = datetime.strptime(flight_change_input, '%Y-%m-%d %H:%M')
                flight_change_input = G.TIMEZONE_UTC.localize(flight_change_input)
                input_ok = True
            except:
                print('The input was invalid, try again!')

        return flight_change_input

    @staticmethod
    def __repeat_y_n_input(message):
        y_n_input = input(message).lower()
        # Keep asking if wrong input
        while y_n_input != 'y' and y_n_input != 'n':
            print('Input not valid, try again!')
            y_n_input = input(message).lower()

        # Make yes/no into True/False
        if y_n_input == 'y':
            t_f_input = True
        else:
            t_f_input = False

        return t_f_input

    def __fix_rotations_user_input(self, df_merge):
        ''' When inconsistent rotations are found because the flights are scheduled to overlap or the rotation is
        longer than regular rotations, the user is asked to fix the input.'''
        # ROTATIONS THAT ARE TOO LONG
        df_merge['rot_too_long'] = (df_merge['LegNumber'] == df_merge['NumberOfLegs']) & \
                                   ((df_merge['ScheduledArrivalTimeAtHovUtc'] - df_merge['RotationHeadStdUtc']) >
                                    timedelta(days=2))
        rotations_too_long = df_merge[df_merge['rot_too_long']]['RotationId'].to_list()

        # ROTATIONS WITH FLIGHTS DEPARTING BEFORE THE PREVIOUS FLIGHT IS SCHEDULED TO ARRIVE
        # Order flights within rotations
        df_merge = df_merge.sort_values(['RotationId', 'LegNumber'])
        # Find rotations with overlapping scheduled flights
        df_merge['next_departure'] = df_merge.groupby(['RotationId'])['ScheduledDepartureTimeAtHovUtc'].shift(-1)
        # Check if flight arrives after scheduled departure of next flight
        df_merge['overlapping_flight'] = df_merge['next_departure'] < df_merge['ScheduledArrivalTimeAtHovUtc']
        rotations_overlapping_flights = df_merge[df_merge['overlapping_flight']]['RotationId'].to_list()

        # Drop columns not used anymore
        df_merge = df_merge.drop(columns=['rot_too_long', 'next_departure', 'overlapping_flight'])

        # ASK FOR USER INPUT TO FIX ROTATIONS
        # Print list of found problematic rotations
        if rotations_too_long != [] or rotations_overlapping_flights != []:
            print('##### SOME INCONSISTENT ROTATIONS WHERE FOUND #####')
        if rotations_too_long != []:
            print('Long rotations: ', rotations_too_long)
        if rotations_overlapping_flights != []:
            print('Rotations with overlapping flights: ', rotations_overlapping_flights)

        rotations_to_fix = list(set(rotations_too_long + rotations_overlapping_flights))
        for rot_to_fix in rotations_to_fix:
            # Find df to print for used
            df_to_print = df_merge[df_merge['RotationId'] == rot_to_fix]
            df_to_print = df_to_print[['RotationId', 'FlightLegId', 'LegNumber','NumberOfLegs', 'DepartureAirport',
                                       'ArrivalAirport', 'ScheduledDepartureTimeAtHovUtc',
                                       'ScheduledArrivalTimeAtHovUtc', 'ActualBlockTimeDuration', 'AircraftType']]

            print('\n\n##### ROTATION', rot_to_fix, '#####\n')
            print(df_to_print.to_string(index=False))
            # Check if user wants to modify rotation
            modify_rotation = self.__repeat_y_n_input('\nDo you want to modify rotation ' + rot_to_fix + '? (Y/N)')
            if modify_rotation:
                flights_to_fix = df_to_print['FlightLegId'].to_list()
                for flight_to_fix in flights_to_fix:

                    # DEPARTURE TIME
                    flight_dep_new = self.__ask_input_flight_changes(flight_to_fix, 'dep')
                    if flight_dep_new != False:
                        df_merge['ScheduledDepartureTimeAtHovUtc'] = df_merge['ScheduledDepartureTimeAtHovUtc'].mask(
                            df_merge['FlightLegId'] == flight_to_fix, flight_dep_new)

                    # ARRIVAL TIME
                    flight_arr_new = self.__ask_input_flight_changes(flight_to_fix, 'arr')
                    if flight_dep_new != False:
                        df_merge['ScheduledArrivalTimeAtHovUtc'] = df_merge['ScheduledArrivalTimeAtHovUtc'].mask(
                            df_merge['FlightLegId'] == flight_to_fix, flight_arr_new)

                    # FLIGHT LEG ID
                    flight_id_new = self.__ask_input_flight_changes(flight_to_fix, 'id')
                    if flight_dep_new != False:
                        df_merge['FlightLegId'] = df_merge['FlightLegId'].mask(
                            df_merge['FlightLegId'] == flight_to_fix, flight_id_new)

        return df_merge

    def __load_rotations_from_BlueLagoon(self):
        ''' Load schedule from BlueLagoon database and if the rotations was not loaded before, save the schedule as
        a pickle file for future use'''
        # Find end date for schedule
        date_start = datetime.strptime(self.scenario['Rotations_start'], "%Y-%m-%d")
        date_end = date_start + timedelta(weeks=G.ROTATIONS_WEEKS)
        date_end = date_end.strftime("%Y-%m-%d")

        ##### DEFINE QUERIES #####
        query_rotations = '''SELECT [RotationId]
                                  ,[LegNumber]
                                  ,[FlightLegUtcId]
                                  ,[NumberOfLegs]
                                  ,[RotationHeadStdUtc]
                                  ,[RotationHeadStdLocal]
                                  ,[LegCancelled]
                                  ,[RotationCancelled]
                                  ,[AircraftOwner]
                                  ,[AircraftType]
                                  ,[FlightGroup]
                                  ,[FlightCancelled]
                                  ,[DepartureAirport]
                                  ,[ArrivalAirport]
                                  ,[ScheduledDepartureTimeUtc]
                                  ,[ScheduledDepartureTimeLocal]
                                  ,[ScheduledArrivalTimeUtc]
                                  ,[ScheduledArrivalTimeLocal]
                                  FROM [BlueLagoonMart].[dbo].[FlightRotations]

                                  where RotationHeadStdUtc >= ' ''' + self.scenario['Rotations_start'] + \
                          ''' ' and RotationHeadStdUtc<= ' ''' + date_end + \
                          ''' ' and FlightGroup = 'ICA' and AircraftOwner = 'KL' '''

        query_flight_legs = ''' SELECT [FlightLegUtcLegId]
                                       ,[FlightLegUtcId]
                                       ,[RotationHeadLegScheduledDepartureTimeUtc]
                                       ,[AircraftOwner]
                                       ,[AircraftType]
                                       ,[FlightGroup]
                                       ,[FlightCancelled]
                                       ,[FlightCancellationTimeUtc]
                                       ,[DepartureAirport]
                                       ,[ArrivalAirport]
                                       ,[ScheduledDepartureTimeUtc]
                                       ,[ScheduledArrivalTimeUtc]
                                       ,[ScheduledBlockTimeDuration]
                                       ,[ActualBlockTimeDuration]
                                       ,[ScheduledDepartureTimeAtD03Utc]
                                       ,[AircraftTypeAtD03]
                                       ,[ScheduledDepartureTimeAtHovUtc]
                                       ,[ScheduledArrivalTimeAtHovUtc]

                                       FROM [BlueLagoonMart].[dbo].[FlightLegs]
                                        where   RotationHeadLegScheduledDepartureTimeUtc >= ' ''' + self.scenario['Rotations_start'] + \
                            ''' ' and RotationHeadLegScheduledDepartureTimeUtc <= ' ''' + date_end + \
                            ''' ' and FlightGroup = 'ICA' and AircraftOwner = 'KL' '''

        ##### IMPORT AND MERGE DATAFRAMES #####
        # Import df renaming column [FlightLegUtcId] to [FlightLegId] due to change in database column name during development
        df_rotations = dataframe_from_blueLagoon_query(query_rotations,
                                                       columns_rename={'FlightLegUtcId': 'FlightLegId'})
        df_flights = dataframe_from_blueLagoon_query(query_flight_legs,
                                                     columns_rename={'FlightLegUtcId': 'FlightLegId'},
                                                     # {'FlightLegUtcLegId': 'FlightLegId'},
                                                     columns_exclude_utc_localization=['FlightLegUtcLegId',
                                                                                       'FlightLegUtcId'])

        # Merge the two dataframes on the flights ID
        df_merge = pd.merge(df_rotations[['RotationId', 'FlightLegId', 'RotationHeadStdUtc', 'LegNumber', 'NumberOfLegs']],
                            df_flights,
                            on=['FlightLegId'], how='left', suffixes=('', ''))


        ##### REMOVE CANCELLED ROTATIONS #####
        # Remove flights for which data in df_flight not found
        df_merge = df_merge[pd.isnull(df_merge['FlightLegUtcLegId'])==0]

        # Remove flights that were cancelled before the rotations are passed to the OCT
        date_start = G.TIMEZONE_UTC.localize(date_start)
        date_include_rotations = date_start - timedelta(days=G.EXCLUDE_ROTATIONS_CANCELLED_BEFORE)
        df_merge = df_merge[(pd.isnull(df_merge['FlightCancellationTimeUtc'])) \
                            | (df_merge['FlightCancellationTimeUtc'] > date_include_rotations)]


        ##### FIX DEPARTURE TIME WHEN NOT DIRECTLY AVAILABLE #####
        # When departure time at hov not available, use final scheduled time
        df_merge['ScheduledDepartureTimeAtHovUtc'] = df_merge['ScheduledDepartureTimeAtHovUtc'].mask(
            pd.isnull(df_merge['ScheduledDepartureTimeAtHovUtc']),
            df_merge['ScheduledDepartureTimeUtc'])
        df_merge['ScheduledArrivalTimeAtHovUtc'] = df_merge['ScheduledArrivalTimeAtHovUtc'].mask(
            pd.isnull(df_merge['ScheduledArrivalTimeAtHovUtc']),
            df_merge['ScheduledArrivalTimeUtc'])

        # For flights for which block time is not available, use computed flight time
        if len(df_merge[pd.isnull(df_merge['ScheduledArrivalTimeAtHovUtc'])]) > 0:
            flights_duration = self.flights_duration.copy()
            flights_duration['Airport1'] = flights_duration['OrigDestAirports'].str[0]
            flights_duration['Airport2'] = flights_duration['OrigDestAirports'].str[1]
            flights_duration = flights_duration.rename(columns={'ActualBlockTimeDuration': 'BlockTimeAvg'})
            flights_duration = flights_duration.drop(columns=['OrigDestAirports'])
            # Merge block time assuming airport1 and airport2 can be either the departure or arrival airport
            df_merge = pd.merge(df_merge, flights_duration, how='left',
                                left_on=['DepartureAirport', 'ArrivalAirport'],
                                right_on=['Airport1', 'Airport2'])
            df_merge = df_merge.drop(columns=['Airport1', 'Airport2'])

            df_merge = pd.merge(df_merge, flights_duration, how='left', suffixes=('', '2'),
                                left_on=['DepartureAirport', 'ArrivalAirport'],
                                right_on=['Airport2', 'Airport1'])
            df_merge['BlockTimeAvg'] = df_merge['BlockTimeAvg'].mask(pd.isnull(df_merge['BlockTimeAvg']),
                                                                     df_merge['BlockTimeAvg2'])
            # Mask the missing acutal block time with the found average block time
            df_merge['ActualBlockTimeDuration'] = df_merge['ActualBlockTimeDuration'].mask(
                pd.isnull(df_merge['ActualBlockTimeDuration']),
                df_merge['BlockTimeAvg'])
            # Drop unnecessary column
            df_merge = df_merge.drop(columns=['BlockTimeAvg', 'BlockTimeAvg2', 'Airport1', 'Airport2'])

        # If arrival time not available, compute it from departure time
        df_merge['ScheduledArrivalTimeAtHovUtc'] = df_merge['ScheduledArrivalTimeAtHovUtc'].mask(
            pd.isnull(df_merge['ScheduledArrivalTimeAtHovUtc']),
            df_merge['ScheduledDepartureTimeAtHovUtc'] + pd.to_timedelta(df_merge['ActualBlockTimeDuration'], 'min'))

        ##### HEAD START AND SCHEDULED DEPARTURE OF FIRST FLIGHT MUST BE THE SAME #####
        df_merge['RotationHeadStdUtc'] = df_merge.sort_values(['RotationId', 'LegNumber'])\
        ['ScheduledDepartureTimeAtHovUtc'].groupby(df_merge['RotationId']).transform('first')

        ##### ASK FOR USED INPUT IF INCONSISTENT ROTATIONS ARE FOUND #####
        df_merge = self.__fix_rotations_user_input(df_merge)

        ##### CALCULATE COLUMNS FOR SIMULATION #####
        # Remove dates from IDs
        columns_remove_date = ['RotationId', 'FlightLegId']
        for column in columns_remove_date:
            df_merge = self.df_remove_first_charachters_from_column(df_merge, column)

        # Add weekday and change datetime to time
        columns_weekday = ['RotationHeadStdUtc', 'ScheduledDepartureTimeAtHovUtc', 'ScheduledArrivalTimeAtHovUtc']
        for column in columns_weekday:
            df_merge = self.add_weekday(df_merge, column)
            df_merge[column] = df_merge[column].dt.time

        ##### SAVE SCHEDULE #####
        schedule_file_name = self.__find_schedule_file_name()
        # Check if file exists. if yes, ask if file should be overwritten.
        schedule_already_saved = os.path.exists(os.path.join(directories.schedules,schedule_file_name))
        if schedule_already_saved == False:
            write_pickle(filename=schedule_file_name, struct=df_merge, directory=directories.schedules)
        else:
            should_overwrite = self.__repeat_y_n_input('Schedule already saved. Overwrite it? (Y/N)')
            if should_overwrite:
                write_pickle(filename=schedule_file_name, struct=df_merge, directory=directories.schedules)
                print('The new schedule was saved')
            elif should_overwrite == False:
                print('The new schedule was NOT saved')

        return df_merge


    def __load_subtypes(self):
        '''Returns dataframes with details on aircraft subtypes'''
        df_subtypes = load_csv(INPUT_FILES.AIRCRAFT_TYPE_DETAILS)
        df_TAT = load_csv(INPUT_FILES.TURNAROUND)

        # [FIX] Force column to string before merge (to match types)
        df_subtypes['AircraftTypeCodeIATA'] = df_subtypes['AircraftTypeCodeIATA'].astype(str)
        df_TAT['AircraftTypeCodeIATA'] = df_TAT['AircraftTypeCodeIATA'].astype(str)

        # [FIX] Force SubfleetDetailCode to string here too!
        if 'SubfleetDetailCode' in df_subtypes.columns:
            df_subtypes['SubfleetDetailCode'] = df_subtypes['SubfleetDetailCode'].astype(str)

        df = pd.merge(df_subtypes, df_TAT, on='AircraftTypeCodeIATA', how='left')

        # Only keep ICA fleet data
        df = df[df['AircraftTypeCodeIATA'].isin(G.ICA_SUBTYPES)]
        return df

    def __load_disruptions_hub(self):
        ''' Returns list of dictionaries containing information about disruption events '''
        disruptions_hub = read_pickle(INPUT_FILES.DISRUPTIONS_HUB)
        return disruptions_hub

    def __load_disruptions_outstations(self):
        ''' Returns dictionary containing information about disruption events '''
        disruptions_outstations = read_pickle(INPUT_FILES.DELAYS_OUTSTATIONS)
        return disruptions_outstations

    def __load_distributions_AOG(self):
        ''' Returns dictionary containing information about AOG slots '''
        if pd.isnull(self.scenario['AOG_distr']):
            distributions_AOG = read_pickle(INPUT_FILES.DISRTRIBUTIONS_AOG)
        else:
            distributions_AOG = read_pickle(self.scenario['AOG_distr'], directory=directories.aog_distributions)
        return distributions_AOG

    def __load_distributions_NR(self):
        ''' Returns dictionary containing information about Non-Routine labor in maintenance slots '''
        distributions_NR = read_pickle(INPUT_FILES.DISTRIBUTIONS_NR)
        return distributions_NR

    def __load_distributions_DDs(self):
        ''' Returns dictionary containing information about Deferred Defects inter arrival time and count '''
        distributions_DDs=  read_pickle(INPUT_FILES.DISTRIBUTIONS_DD)
        return distributions_DDs

    def __load_flights_durations(self):
        ''' Returns dataframe containing flights duration '''
        flights_duration = read_pickle(INPUT_FILES.FLIGHT_DURATION)
        return flights_duration

    def __load_TAT(self):
        ''' Returns dataframe containing TAT at different airport cathegories '''
        TAT = read_pickle(INPUT_FILES.TAT)
        return TAT

    def __load_AChecks_historical(self):
        AChecks = load_csv(INPUT_FILES.A_CHECKS)
        AChecks = AChecks.rename(columns={'WorkPackage_WorkType': 'WORKTYPE',
                                          'WorkPackage': 'WP',
                                          'WorkPackage_Aircraft': 'ACREGIS',
                                          'WorkPackage_ActualStartDate': 'STDATE_INNIT',
                                          'WorkPackage_Aircraft_Fleet': 'ACSUBTYP'})
        AChecks['ACREGIS'] = AChecks['ACREGIS'].str.replace('-', '')
        AChecks = AChecks.sort_values(by='STDATE_INNIT')
        return AChecks

    def __load_slotsNorm(self):
        ''' Import dataframe of norm slots. Only the slots for subtypes specified in in the chosen scenario are kept.'''

        ###### NOTE: Slots_norm file must EXCLUDE towing time, and times must be given in LOCAL TIME ######

        # Import excel
        slotsNorm = load_excel(INPUT_FILES.SLOTS_NORM_SCENARIOS)
        # Filter the selected variant
        slotsNorm = slotsNorm[slotsNorm['Variant'] == self.scenario['Slotsnorm_scenario']]
        # Make allowed registrations into a list of strings
        slotsNorm['Regis_allowed'] = slotsNorm['Regis_allowed'].apply(str)
        slotsNorm['Regis_allowed'] = slotsNorm['Regis_allowed'].str.split(',')
        # Only keep requested subtypes
        slotsNorm['Subtypes'] = slotsNorm['Subtypes'].apply(str)
        slotsNorm = slotsNorm[(slotsNorm['Subtypes'].isin(self.scenario['Aircraft_types'])) |
                              (slotsNorm['Subtypes'].str[:2].isin(self.scenario['Aircraft_types']))]

        # [FIX] Force time columns to be datetime.time objects instead of strings
        slotsNorm['Time_start'] = pd.to_datetime(slotsNorm['Time_start'].astype(str)).dt.time
        slotsNorm['Time_end'] = pd.to_datetime(slotsNorm['Time_end'].astype(str)).dt.time

        return slotsNorm

    def __load_DDs(self):
        DDs = read_pickle(INPUT_FILES.TASKS_DDS)
        return DDs

    # UNUSED: empty stub, never implemented or called (legacy from old repo).
    def __load_blocks(self):
        pass # TODO


    @staticmethod
    def df_drop_duplicates_keep_newest(df,column,based_on=None):
        # Sort dataframe
        if based_on!=None:
            df.sort_values(by=based_on, ascending=False, inplace=True)
        # Keep one row
        df.drop_duplicates(subset=column, keep='first', inplace=True)
        # Order df
        df.sort_values(by=column, ascending=True, inplace=True)
        return df

    @staticmethod
    def df_remove_first_charachters_from_column(df,column,n=11):
        '''Remove the first n charachters from the selected column'''
        df[column] = df[column].str[n:]
        return df

    @staticmethod
    def add_weekday(df, column):
        column_new = column+'Weekday'
        df[column_new] = df[column].dt.weekday
        return df


def load_data(scenario_id):
    # If complete preprocessing, import data from csv files
    if G.PREPROCESSING == 1:
        # Load data in dataframes
        data = Data(scenario_id)
        # Save the dataframes as Pickles
        write_pickle(data, 'data')
        log_info('Dataframes generated correctly')

    # Otherwise import pickles
    else:
        data = read_pickle(INPUT_FILES.PICKLE_DATA)
        log_info('Dataframes imported')
    for wr in data.warnings:
        log_warning(wr)

    return data
