from config import G, directories
from classes.classes_aircraft import Aircraft, AcSubtype
from classes.classes_operations import Rotation_norm, Flight_norm, Airport, Rotation, ReserveSlot
from classes.classes_maintenance import SlotNorm, Slot, Requirement, Task
from data_import.load_data import write_pickle, read_pickle
from datetime import datetime as dt, timedelta
import math
import os
import pandas as pd
from output.output_functions import log_info, log_warning, log_error



def join_list_in_text(list):
    ''' Unify a list of strings into one string separated by the or (|) operator'''
    text = '|'.join(r"{}".format(str(x)) for x in list)
    return text

# def df_filter_rows_based_on_text(df, column, text, contains=1):
#     '''
#     Filter rows based on if a certain columns contains or does not contain a string
#     :param df: dataframe from which rows should be removed
#     :param column: Column in which text should be checked
#     :param text: string or list of strings that must appear in column
#     :param contains: 1 to only keep rows that contain text, 0 to only keep rows that do NOT contain test
#     :return: modified df
#     '''
#     # If list of strings given, put them together with or statement
#     if isinstance(text, str) == 0:
#         text = join_list_in_text(text)
#     # Remove rows
#     if contains == 1:
#         df = df[df[column].str.contains(text)]
#     elif contains == 0:
#         df = df[~df[column].str.contains(text)]
#     else:
#         raise Exception('Invalid value for contains key')
#     return df
def df_filter_rows_based_on_text(df, column, text, contains=1):
    '''
    Filter rows based on if a certain columns contains or does not contain a string
    :param df: dataframe from which rows should be removed
    :param column: Column in which text should be checked
    :param text: string or list of strings that must appear in column
    :param contains: 1 to only keep rows that contain text, 0 to only keep rows that do NOT contain test
    :return: modified df
    '''
    # [FIX] Force the column to be string type so .str accessor works
    df[column] = df[column].astype(str)

    # If list of strings given, put them together with or statement
    if isinstance(text, str) == 0:
        text = join_list_in_text(text)
    # Remove rows
    if contains == 1:
        df = df[df[column].str.contains(text)]
    elif contains == 0:
        df = df[~df[column].str.contains(text)]
    else:
        raise Exception('Invalid value for contains key')
    return df

class Objects:
    def __init__(self, data, iteration_n):
        # General
        self.airports = self.__generate_airports(data)
        self.aircraft = self.__generate_aircraft(data)
        self.subtypes = self.__generate_subtypes(data)

        # Network
        self.schedule, self.flights = self.__generate_schedule(data, iteration_n)
        self.reserve_slots = self.__generate_reserve_slots(data)

        # Maintenance
        self.task_hangar_preparation = self.__generate_task_hangar_preparation()
        self.slotsNorm = self.__generate_slotsNorm(data)
        self.slots_TO, slots_A = self.__generate_slots_from_norm_slots(data)
        self.slots_AChecks = self.__choose_AChecks(data, slots_A)
        self.slots_LM = self.__generate_slots_LM(data)
        self.requirements = self.__generate_requirements(data)

    def __generate_aircraft(self, data):
        ''' Generate aircraft instances from pandas dataframes of real and additional simulated aircraft'''
        # Reduce aircraft list
        data.aircraft = df_filter_rows_based_on_text(data.aircraft, 'AircraftTypeCodeIATA', data.scenario['Aircraft_types'])

        # Remove some aircraft if requested in the data
        if data.scenario['Aircraft_remove'] != []:
            data.aircraft = df_filter_rows_based_on_text(data.aircraft, 'AircraftRegistrationFull',
                                                         data.scenario['Aircraft_remove'], contains=0)

        # Check if any additional aircraft must be included
        if data.scenario['Aircraft_additional'] != []:
            # Reduce list of additional aircraft. Keep aircraft matching the selected aircraft type and registrations
            data.aircraft_additional = df_filter_rows_based_on_text(data.aircraft_additional, 'AircraftTypeCodeIATA',
                                                                    data.scenario['Aircraft_types'])
            data.aircraft_additional = df_filter_rows_based_on_text(data.aircraft_additional,
                                                                    'AircraftRegistrationFull',
                                                                    data.scenario['Aircraft_additional'])
            # Concatenate two dataframes of aircraft
            df_aircraft_full = pd.concat([data.aircraft, data.aircraft_additional])
        else:
            df_aircraft_full = data.aircraft

        # Generate Aircraft objects
        aircraft_list = []
        # Find initial coordinates
        home_base = next((ap for ap in self.airports if ap.id==G.AIRPORT_BASE), None)
        if home_base == None:
            coordinates = {'latitude': -1000,
                           'longitude': -1000}
        else:
            coordinates = home_base.coordinates
        for index, ac_row in df_aircraft_full.iterrows():
            # Find subtype id
            subtype_id = self.subtype_id(ac_row['AircraftTypeCodeIATA'], ac_row['SubfleetDetailCode'])
            aircraft = Aircraft(id=ac_row['AircraftRegistrationFull'],
                                subtype=subtype_id,
                                asia_tail=ac_row['AsiaTail'],
                                coordinates=coordinates)
            aircraft_list.append(aircraft)

        return aircraft_list

    @staticmethod
    def subtype_id(type, detail):
        id = str(type) + '|' + detail
        return id

    def __generate_subtypes(self, data):
        '''Generate subtypes instances from pandas dataframe'''
        subtypes = []
        # Only create subtypes used in simulation
        df = df_filter_rows_based_on_text(data.subtypes, 'AircraftTypeCodeIATA', data.scenario['Aircraft_types'])

        for index, st_row in df.iterrows():
            id = self.subtype_id(st_row['AircraftTypeCodeIATA'], st_row['SubfleetDetailCode'])
            subtype = AcSubtype(id=id,
                                IATA=st_row['AircraftTypeCodeIATA'],
                                detail_code=st_row['SubfleetDetailCode'],
                                name=st_row['AircraftTypeName'],
                                DT=st_row['DeparturePrepTime'],
                                TAT=st_row['TurnAroundTime'],
                                AT=st_row['ArrivalPrepTime'])
            subtypes.append(subtype)
        return subtypes


    def __generate_schedule(self, data, iteration_n):
        '''Generate Flights and Rotations instances from dataframes'''
        # Generate Flight and Rotation objects for general schedule
        schedule = []
        flights = []

        def add_weekday_to_id(id, weekday):
            '''Generate a unique ID for rotations and flights by adding the weekday at the end of the original ID'''
            id_new = str(id)  + '|D'+ str(weekday) + '|'
            return id_new

        def find_orig_dest(airport_dep,airport_arr):
            '''Given the departure and arrival airport of a flight, generate the tuple (orig,dest)
            where the order of origin and destination is alphabethical'''
            orig_dest = tuple(sorted([airport_dep, airport_arr]))
            return orig_dest

        # Only keep rotations executed with selected aircraft subtypes and that include
        # flights executed by different aircraft types
        if G.ONLY_ROTATIONS_SELECTED_SUBTYPES == 1:

            # Generate warning if some rotations excluded because no aircraft type is assigned
            rot_ac_type_not_assigned_df = data.rotations[pd.isnull(data.rotations['AircraftType'])]
            rot_ac_type_not_assigned_df['rotation_unique_id'] = rot_ac_type_not_assigned_df['RotationId'] + \
                                                                rot_ac_type_not_assigned_df['RotationHeadStdUtcWeekday'].astype(str)
            rot_ac_type_not_assigned = list(set(rot_ac_type_not_assigned_df['rotation_unique_id'].to_list()))

            if rot_ac_type_not_assigned != [] and iteration_n==0:
                log_error('Rotations ' + str(rot_ac_type_not_assigned) +
                          ' were excluded because no aircraft type was assigned to them', print_error=False)
            elif rot_ac_type_not_assigned != [] and iteration_n!=0:
                log_warning('Rotations ' + str(rot_ac_type_not_assigned) + \
                            ' were excluded because no aircraft type was assigned to them')

            rotations_to_keep = data.rotations.groupby(['RotationId', 'RotationHeadStdUtcWeekday']).filter(
                lambda x: (x['AircraftType'].iloc[0] == x['AircraftType']).all())
            rotations_to_keep = df_filter_rows_based_on_text(rotations_to_keep, 'AircraftType',
                                                             data.scenario['Aircraft_types'])

        else:
            rotations_to_keep = data.rotations


        for index, flight_row in rotations_to_keep.iterrows():
            # Check if rotation already exists
            rotation = next((rt for rt in schedule if rt.id_general == flight_row['RotationId']
                             and rt.weekday_dep == flight_row['RotationHeadStdUtcWeekday']), None)

            # If rotation does not exist, create it
            if rotation == None:
                # Generate rotation and flight ID
                rot_id = add_weekday_to_id(flight_row['RotationId'], flight_row['RotationHeadStdUtcWeekday'])
                # Generate rotation instance
                rotation = Rotation_norm(id=rot_id,
                                         id_general=flight_row['RotationId'],
                                         n_legs=flight_row['NumberOfLegs'],
                                         time_dep=flight_row['RotationHeadStdUtc'],
                                         weekday_dep=flight_row['RotationHeadStdUtcWeekday'],
                                         subtypes=flight_row['AircraftType']
                                         )
                # Add rotation to schedule
                schedule.append(rotation)

            # If flight is first flight, check that departure time of rotation and flight match
            if flight_row['LegNumber']==1:
                assert (flight_row['RotationHeadStdUtc'] == flight_row['ScheduledDepartureTimeAtHovUtc']) \
                        and (flight_row['RotationHeadStdUtcWeekday'] == flight_row['ScheduledDepartureTimeAtHovUtcWeekday'])

            # Generate flight ID
            flight_id = add_weekday_to_id(flight_row['FlightLegId'], flight_row['ScheduledDepartureTimeAtHovUtcWeekday'])
            # Generate origin destination tuple
            orig_dest = find_orig_dest(flight_row['DepartureAirport'],flight_row['ArrivalAirport'])
            # Find flight duration
            try:
                flight_block_time = data.flights_duration[data.flights_duration['OrigDestAirports']==orig_dest]['ActualBlockTimeDuration'].iloc[0]
            except:
                # If flight duration is not present in currently used distributions, take block time from the actual flight
                flight_block_time = flight_row['ActualBlockTimeDuration']
                # If block time is not present, raise Exception
                if pd.isnull(flight_block_time):
                    raise Exception('Block time for flight '+ flight_row['FlightLegId']+' was not found.')
                # If block time found, throw a warning
                log_warning('The block time for flight '+ flight_row['FlightLegId'] +
                                ' was not found in the computed flight durations, so the actual duration was used.')

            # Generate scheduled flight instance
            time_dep_val = flight_row['ScheduledDepartureTimeAtHovUtc']
            time_arr_val = flight_row['ScheduledArrivalTimeAtHovUtc']

            # Ensure they are time objects
            time_dep = time_dep_val.time() if hasattr(time_dep_val, 'time') else time_dep_val
            time_arr = time_arr_val.time() if hasattr(time_arr_val, 'time') else time_arr_val
            flight = Flight_norm(id=flight_id,
                                 id_general=flight_row['FlightLegId'],
                                 time_dep=flight_row['ScheduledDepartureTimeAtHovUtc'],
                                 time_arr=flight_row['ScheduledArrivalTimeAtHovUtc'],
                                 weekday_dep=flight_row['ScheduledDepartureTimeAtHovUtcWeekday'],
                                 weekday_arr=flight_row['ScheduledArrivalTimeAtHovUtcWeekday'],
                                 rotation=rotation.id,
                                 leg_number=flight_row['LegNumber'],
                                 airport_dep=flight_row['DepartureAirport'],
                                 airport_arr=flight_row['ArrivalAirport'],
                                 block_time=flight_block_time,
                                 subtypes=flight_row['AircraftType']
                                 )

            flights.append(flight)

        # Order rotation by departure time from Monday
        schedule = sorted(schedule, key = lambda x: (x.weekday_dep, x.time_dep))

        return schedule, flights


    def __generate_reserve_slots(self, data):
        ''' Generate the reserve slots for the duration of the simulation '''
        reserve_slots = []
        day = data.scenario['Simulation_start']
        for day_n in range(G.SIM_DURATION):
            for slot_day_count in range(data.scenario['Reserves_per_day']):
                reserve_slot = ReserveSlot(date = day, day_count=slot_day_count)
                reserve_slots.append(reserve_slot)
            day = day + timedelta(days=1)

        return reserve_slots


    def __generate_airports(self, data):
        airports = []
        # Create a list with airports used in the considered rotations
        airports_used = data.rotations['ArrivalAirport'].drop_duplicates().tolist()
        for index, ap_row in data.airports.iterrows():
            # Create airport only if covered in considered rotations
            if ap_row['IataAirportCode'] in airports_used:
                airport = Airport(id=ap_row['IataAirportCode'],
                                  name=ap_row['AirportName'],
                                  country_code=ap_row['CountryCode'],
                                  latitude=ap_row['Latitude'],
                                  longitude=ap_row['Longitude']
                              )
                airports.append(airport)
        return airports

    def find_flightNorm_waypoints(self):
        ''' Generate waypoints for the norm flights'''
        for flight in self.flights:
            flight.find_waypoints()

    def __generate_task_hangar_preparation(self):
        '''
        When an aircraft undergoes maintenance in the hangar, some generic tasks must be executed for getting the
        aircraft ready for maintenance and after it. This task is therefore added to the work package to account for
        these operations.
        '''
        task_hangar_preparation = Task(id='HANGAR_PREPARATION',
                                       durationEst = timedelta(minutes=0),
                                       laborEst = timedelta(minutes=G.TASK_HANGAR_PREPARATION_LABOR),
                                       aircraft = None,
                                       dateArrival = None,
                                       dateReady = None,
                                       dateDue = None,
                                       workType = 'H',
                                       type = 'HANGAR_PREPARATION',
                                       info = 'HANGAR_PREPARATION')
        return task_hangar_preparation

    def __generate_slotsNorm(self, data):
        ''' Generate slotsnorm objects, i.e. slots that repeat themselves every so many days.'''
        slotsNorm_list = []
        for index, slot_row in data.slotsNorm.iterrows():
            # If slot is executed in hangar, the hangar preparation task should be added
            if slot_row['Location'] == 'H':
                task_hangar_preparation = self.task_hangar_preparation
            else:
                task_hangar_preparation = None

            # Generate object
            slot = SlotNorm(
                id=slot_row['Slotnr'],
                ac_subtype=slot_row['Subtypes'],
                time_start=slot_row['Time_start'],
                time_end=slot_row['Time_end'],
                day_start=slot_row['Day_start'],
                day_end=slot_row['Day_end'],
                cycle_duration=slot_row['Cycle_duration'],
                type=slot_row['Slot_type'],
                location=slot_row['Location'],
                ac_allowed=slot_row['Regis_allowed'],
                task_hangar_preparation=task_hangar_preparation,
                remarks=slot_row['Slot_remarks']
            )
            # Add object to list
            slotsNorm_list.append(slot)

        # Assign day order and initialize drumbeat aircraft turn
        if slotsNorm_list != []:
            # Assign daily order
            self.__assign_slot_daily_order(slotsNorm_list, 'TO')
            self.__assign_slot_daily_order(slotsNorm_list, 'A')
        return slotsNorm_list

    @staticmethod
    def __assign_slot_daily_order(slotsNorm_list, slot_remarks):
        '''Assign a daily order to the given slots'''
        slots_daily_order = [sl for sl in slotsNorm_list if sl.remarks == slot_remarks]
        ac_subtypes_list = list(set([sn.subtype for sn in slots_daily_order]))
        # For every ac subtype, find order per day
        for subtype in ac_subtypes_list:
            for day in range(slots_daily_order[0].cycle_duration):
                daily_slots = [sl for sl in slots_daily_order if sl.day_start == day and subtype in sl.subtype]
                daily_slots.sort(key=lambda x: x.time_start)
                for slot in daily_slots:
                    slot.day_order = daily_slots.index(slot)
                    slot.ac_turn = daily_slots.index(slot)


    @staticmethod
    def generate_slot_id_from_slotnorm(ac_subtype, cycle, day_count, day_order, slot_type, day):
        ''' Generate id for TO slot and Acheck slots from slotnorm '''
        if slot_type == 'TO':
            id = 'TO_' +ac_subtype + 'Cycle' + str(cycle) + 'Day' + str(day_count) +'S' + str(day_order)
        elif slot_type == 'A':
            id = 'A_' + ac_subtype + '_' + day.strftime('%Y-%m-%d') +'_S' + str(day_order)
        else:
            raise Exception('Slot type not supported')
        return id

    def __generate_slots_from_norm_slots(self, data):
        ''' Generate slots for the simulation.

        Note on the start day initialization:
        For slots cycles shorter than a week, the slots are generated so that day 0 of the cycle correspond to the
        simulation start day. When the cycle is longer than a week, slots are generated so that they cycle is
        anticipated to the monday of that week. Since cycles often have a duration that is a multiple of 7,
        this makes sure that they are always initialized to a monday.  '''
        slots_TO = []
        slots_A = []
        # initialize day and slot id
        day = data.scenario['Simulation_start'].astimezone(G.TIMEZONE_LOCAL)
        cycle = 0
        if self.slotsNorm!=[]:
            cycle_duration = self.slotsNorm[0].cycle_duration
        else:
            cycle_duration = 1
        # If cycle is shorter than a week initialize to day zero. Otherwise initialize to the day of the week
        if cycle_duration < 7:
            cycle_dayCount = 0
        else:
            cycle_dayCount = day.weekday()

        # Assign start date to all slotsNorm
        for slot_norm in self.slotsNorm:
            slot_norm.simulation_start_weekday = cycle_dayCount

        for day_n in range(G.SIM_DURATION+G.MAINTENANCE_SCHEDULE_WINDOW):
            # Slots of the day
            slotsNorm_daily = [sn for sn in self.slotsNorm if sn.day_start == cycle_dayCount]
            for slotNorm in slotsNorm_daily:
                # Find start and end date in correct format
                dateStart = day
                dateStart = dateStart.replace(hour=slotNorm.time_start.hour, minute=slotNorm.time_start.minute)
                dateEnd = dateStart + slotNorm.duration
                # Localize in utc
                dateStart = dateStart.astimezone(G.TIMEZONE_UTC)
                dateEnd = dateEnd.astimezone(G.TIMEZONE_UTC)
                slot_id = self.generate_slot_id_from_slotnorm(str(slotNorm.subtype), cycle, cycle_dayCount,
                                                              slotNorm.day_order, slotNorm.remarks, day)
                # If drumbeat logic is used for aircraft-slot assignment, find assignment
                # #TODO change if drumbeat in slot assignment necessary
                if G.SLOTS_ASSIGNMENT_ORDER == 1:
                    ac_drumbeat = slotNorm.find_acTurn()
                    aircraft_hist = ac_drumbeat
                else:
                    aircraft_hist = None

                # Find max labor that can be scheduled in slot
                if slotNorm.remarks == 'A':
                    if data.requirements.empty and '772' in data.scenario['Aircraft_types']:
                        max_labor = timedelta(hours=G.SAC_GNN_SLOT_CAPACITY)
                    else:
                        max_labor = timedelta(hours=G.A_CHECK_LABOR_TOTAL)
                else:
                    max_labor = None

                # Generate slot object
                slot = Slot(
                    id=slot_id,
                    subtype=slotNorm.subtype,
                    dateStart_init=dateStart,
                    dateStart_final=dateStart,
                    dateEnd_init=dateEnd,
                    dateEnd_final=dateEnd,
                    remarks=slotNorm.remarks,
                    duration=slotNorm.duration,
                    cycle=cycle,
                    slotNorm=slotNorm.id,
                    location=slotNorm.location,
                    aircraft_pre_assigned=aircraft_hist,
                    task_hangar_preparation=slotNorm.task_hangar_preparation,
                    labor_max=max_labor
                )

                if slot.remarks == 'TO':
                    slots_TO.append(slot)
                elif slot.remarks == 'A':
                    slots_A.append(slot)
                else:
                    raise Exception('Slot must be either TO or A slot')

            # Update day count for cycle
            if cycle_dayCount == cycle_duration - 1:
                cycle_dayCount = 0
                cycle += 1
            else:
                cycle_dayCount += 1
            # Update day count
            day += timedelta(days=1)

        return slots_TO, slots_A

    @staticmethod
    def __generate_LM_slot_id(aicraft, slot_start):
        return 'LM_' + aicraft.id + '_' + dt.strftime(slot_start, '%Y-%m-%d')

    def __generate_slots_LM(self, data):
        ''' Generate line maintenance slots for all aircraft included in the simulation.
        For each aircraft, one line maintenance slot si create to cover each scheduling window within the simulation'''

        # Interval between calls of the maintenance scheduler
        scheduling_interval = timedelta(days=G.MAINTENANCE_SCHEDULE_INTERVAL)

        # Generate slot until the end of the simulation + the maintenance scheduling window
        day_slot_generation_end = data.scenario['Simulation_start'] + timedelta(days=G.SIM_DURATION)\
                                  + timedelta(days=G.MAINTENANCE_SCHEDULE_WINDOW)

        # Duration and labor that can be scheduled in a slot
        slot_duration = timedelta(hours=G.LM_SLOT_DURATION)
        slot_labor_total = timedelta(hours=G.LM_SLOT_LABOR_TOTAL)
        slot_labor_per_task = timedelta(hours=G.LM_SLOT_LABOR_PER_TASK)

        # List of LM slots
        LM_slots = []

        # Initialize start of slot and slot count
        day_slot_start = data.scenario['Simulation_start']

        while day_slot_start < day_slot_generation_end:
            # End of LM slot
            day_slot_end = day_slot_start + scheduling_interval
            # Generate slot for each aircraft
            for aircraft in self.aircraft:
                slot_id = self.__generate_LM_slot_id(aircraft, day_slot_start)
                slot_end = day_slot_end - timedelta(minutes=1)
                # Generate slot object
                slot = Slot(
                    id=slot_id,
                    subtype=None,
                    dateStart_init=day_slot_start,
                    dateStart_final=day_slot_start,
                    dateEnd_init=slot_end,
                    dateEnd_final=slot_end,
                    remarks='LM',
                    duration=slot_duration,
                    labor_max=slot_labor_total,
                    labor_max_per_task=slot_labor_per_task,
                    location='P',
                    aircraft_pre_assigned=aircraft.id
                )

                # Impose final duration as initial duration
                slot.compute_duration('scheduled')
                LM_slots.append(slot)

            day_slot_start = day_slot_end

        return LM_slots

    def __generate_slots_AChecks_historical(self, data):
        slots_list= []
        return slots_list

    def __choose_AChecks(self, data, slots_A):
        if G.A_CHECKS_IMPORT == 0:
            slots_chosen = slots_A
        elif G.A_CHECKS_IMPORT == 1:
            slots_chosen = self.__generate_slots_AChecks_historical(data)
        else:
            raise Exception('G.ACHECKS_IMPORT value not supported')
        return slots_chosen

    def __generate_requirements(self, data):
        ''' Generate requirements, i.e. recurring tasks '''

        # Find aircraft types from list of types/subtypes
        req_to_generate = data.requirements.copy()
        if req_to_generate.empty and '772' in data.scenario['Aircraft_types']:
            return self.__generate_policy_requirements(data)
        req_to_generate = req_to_generate.explode('Subtypes')
        req_to_generate = req_to_generate[(req_to_generate['Subtypes'].isin(data.scenario['Aircraft_types']))|
                                          (req_to_generate)['Assembly_short'].isin(data.scenario['Aircraft_types'])]
        # Re-aggregate data
        req_subtypes_list = req_to_generate.groupby(['Requirement_Code', 'Requirement_Assembly'])\
                            ['Subtypes'].apply(list).reset_index()
        # Drop unnecessary columns and duplicates from reduced df
        req_to_generate = req_to_generate.drop(columns=['Subtypes'])
        req_to_generate = req_to_generate.drop_duplicates(['Requirement_Code', 'Requirement_Assembly'])

        # Merge list of subtypes
        req_to_generate = pd.merge(req_to_generate, req_subtypes_list, how='left',
                                   on=['Requirement_Code', 'Requirement_Assembly'])


        requirements_list = []
        for index, row_interval in req_to_generate.iterrows():
            requirement = Requirement(
                code=row_interval['Requirement_Code'],
                subtypes=row_interval['Subtypes'],
                ac_type=row_interval['Assembly_short'],
                info=row_interval['Description'],
                workType=row_interval['Requirement_WorkType'],
                flightHours=row_interval['FH_task'],
                flightCycles=row_interval['FC_task'],
                calendarDays=row_interval['Cal_task'],
                laborEst=row_interval['Requirement_total_labor'],
                durationEst=row_interval['Requirement_duration'],
                req_class=row_interval['Class'],
                type='REQUIREMENT'
            )
            requirements_list.append(requirement)

        # Find most stringent interval based on season
        season = self.find_season(data)
        for requirement in requirements_list:
            requirement.find_shortest_interval(season)

        return requirements_list

    def __generate_policy_requirements(self, data):
        """Build the real 777 recurring requirements from the SAC-GNN policy artifact.

        The policy file is the authoritative 777 task catalogue.  Its Aircraft_1..N
        columns are initial days-to-due in the same deterministic aircraft order used
        by the SAC scheduler.
        """
        path_value = G.SAC_GNN_INITIAL_STATUS_CSV_PATH.replace('\\', os.sep).replace('/', os.sep)
        policy_path = path_value if os.path.isabs(path_value) else os.path.join(directories.anemos, path_value)
        if not os.path.exists(policy_path):
            raise FileNotFoundError('777 maintenance policy requirements not found: ' + policy_path)

        policy = pd.read_csv(policy_path)
        required_columns = {'Task_code', 'Interval', 'Labour', 'Skill', 'Panel'}
        missing = required_columns - set(policy.columns)
        if missing:
            raise ValueError('Maintenance policy data missing columns: ' + str(sorted(missing)))

        aircraft_columns = [f'Aircraft_{index}' for index in range(1, len(self.aircraft) + 1)]
        missing_aircraft = [column for column in aircraft_columns if column not in policy.columns]
        if missing_aircraft:
            raise ValueError('Maintenance policy data missing aircraft due-date columns: '
                             + str(missing_aircraft))

        requirements_list = []
        for _, row in policy.iterrows():
            initial_due_days = {
                aircraft.id: float(row[column])
                for aircraft, column in zip(self.aircraft, aircraft_columns)
            }
            requirement = Requirement(
                code=str(row['Task_code']).strip(),
                subtypes=['772'],
                ac_type='77',
                info='Skill=' + str(row['Skill']) + '; Panel=' + str(row['Panel']),
                workType='H',
                flightHours=None,
                flightCycles=None,
                calendarDays=float(row['Interval']),
                laborEst=float(row['Labour']),
                durationEst=0.0,
                req_class='A',
                type='REQUIREMENT',
                initial_due_days_by_aircraft=initial_due_days
            )
            requirement.find_shortest_interval(self.find_season(data))
            requirements_list.append(requirement)

        return requirements_list

    @staticmethod
    def find_season(data):
        check_summerSart = G.SUMMER_PERIOD['start'].replace(year=data.scenario['Simulation_start'].year)
        check_summerEnd = G.SUMMER_PERIOD['end'].replace(year=data.scenario['Simulation_start'].year)
        if data.scenario['Simulation_start'] >= check_summerSart and data.scenario['Simulation_start'] <= check_summerEnd:
            season = 'summer'
        else:
            season = 'winter'
        return season

    # def generate_rotations(self, data):
    #     '''Instantiate rotations for a defined number of weeks.
    #     Note that this function must be called AFTER pickles are imported'''
    #
    #     self.rotations = []
    #     # Find initial date and correspondent weekday
    #     start_date_iter = data.scenario['Simulation_start']
    #     start_weekday = start_date_iter.weekday()
    #     # Find number of times the schedule must be repeated based on the simulation duration
    #     schedule_iterations = math.ceil(G.SIM_DURATION/(7*G.ROTATIONS_WEEKS))
    #     for iter in range(schedule_iterations):
    #         for rot in self.schedule:
    #             # Find how many days after the initial date the flight happens
    #             delta_days = rot.weekday_dep - start_weekday
    #             if delta_days < 0:
    #                 delta_days += 7
    #             # Find date when flight executed
    #             date = (start_date_iter + timedelta(days=rot.weekday_dep)).date()
    #             val = rot.time_dep
    #             time_only = val.time() if hasattr(val, 'time') else val
    #             rotation = Rotation(rot, date)
    #             # rotation = Rotation(rot,date)
    #             self.rotations.append(rotation)
    #
    #         # Increase start date of the schedule by one week
    #         start_date_iter = start_date_iter + timedelta(weeks=1)

    def generate_rotations(self, data):
        '''Instantiate rotations for a defined number of weeks.
        Note that this function must be called AFTER pickles are imported'''

        self.rotations = []
        # Find initial date and correspondent weekday
        start_date_iter = data.scenario['Simulation_start']
        start_weekday = start_date_iter.weekday()
        # Find number of times the schedule must be repeated based on the simulation duration
        schedule_iterations = math.ceil(G.SIM_DURATION / (7 * G.ROTATIONS_WEEKS))
        for iter in range(schedule_iterations):
            for rot in self.schedule:
                if not rot.flights:
                    continue
                # [FIX] Ensure time_dep is a datetime.time object, not a Timestamp
                if hasattr(rot.time_dep, 'time'):
                    rot.time_dep = rot.time_dep.time()

                # Find how many days after the initial date the flight happens
                delta_days = rot.weekday_dep - start_weekday
                if delta_days < 0:
                    delta_days += 7
                # Find date when flight executed
                date = (start_date_iter + timedelta(days=rot.weekday_dep)).date()
                rotation = Rotation(rot, date)
                self.rotations.append(rotation)

            # Increase start date of the schedule by one week
            start_date_iter = start_date_iter + timedelta(weeks=1)




def generate_objects(data, iteration_n):
    # Generate objects
    if G.PREPROCESSING != 0:
        obj = Objects(data, iteration_n)
        # Save the objects as Pickles
        write_pickle(obj, 'objects')
        log_info('Simulation objects generated')
    # Import pickles
    else:
        obj = read_pickle('objects')
        log_info('Simulation objects imported')

    # Connect objects
    cross_reference_instances(obj, iteration_n)
    # Find waypoints for the norm flights
    obj.find_flightNorm_waypoints()
    # Generate rotations
    obj.generate_rotations(data)
    log_info('Rotations generated')

    return obj

def check_rotations(schedule):
    '''
    Check if the generated or imported rotations are consistent, and returns dictionary with rotations that show
    some issue.
    :return {'n_flights': List of rotations that do not include as many flights as they should
             'dep_arr_ams': List of rotations that do not depart or arrive in AMS
             'airport_chain': List of rotations for which the chain of subsequent airport is broken (when a flight
                              departs from a different station than the previous arrival)
             'flights_dep_time': List of rotations that include flights where one flight is scheduled to depart
                                 before the previous one is scheduled to arrive.
             'exclude': list of flights that must be excluded from the schedule because inconsistent.
                        Always includes the first three categories of inconsistent rotations stated above,
                        as well as the flights_dep_time if parameter G.SCHEDULE_EXCLUDE_ROTATIONS_OVERLAPPING_FLIGHTS == True
            }
    '''
    ##### Check if all flights included #####
    rot_check_included_flights = [rt for rt in schedule if rt.n_legs != len(rt.flights)]

    ##### Check that flights in rotation depart and arrive in AMS #####
    rot_check_departure_arrival_ams = []
    # Consider only rotations not found in previous check
    for rotation in [rt for rt in schedule if rt not in rot_check_included_flights]:
        rotation.flights = sorted(rotation.flights, key=lambda x: x.leg_number)
        if rotation.flights[0].airport_dep != G.AIRPORT_BASE \
            or rotation.flights[-1].airport_arr != G.AIRPORT_BASE:
            rot_check_departure_arrival_ams.append(rotation)

    ##### Check airports chain #####
    rot_check_airport_chain = []
    # Consider only rotations not found in previous checks
    for rotation in [rt for rt in schedule if rt not in rot_check_departure_arrival_ams
                     and rt not in rot_check_included_flights]:
        for fl_index in range(len(rotation.flights)-1):
            flight1 = rotation.flights[fl_index]
            flight2 = rotation.flights[fl_index+1]
            if flight1.airport_arr!=flight2.airport_dep:
                rot_check_airport_chain.append(rotation)
    rot_check_airport_chain = list(set(rot_check_airport_chain))


    ##### Check that scheduled departure of each leg follows scheduled arrival of previous one #####
    rot_check_departure_time = []
    for rotation in schedule:
        for fl_index in range(len(rotation.flights)-1):
            flight1 = rotation.flights[fl_index]
            flight2 = rotation.flights[fl_index+1]
            days_between_flights = flight2.weekday_dep - flight1.weekday_arr
            # Account for week change
            if days_between_flights < 0:
                days_between_flights = days_between_flights + 7
            if flight1.time_arr > flight2.time_dep and days_between_flights == 0:
                rot_check_departure_time.append(rotation)
    # Account for the fact that more than one problem can be found in a rotation
    rot_check_departure_time = list(set(rot_check_departure_time))


    ##### Rotations to exclude #####
    # Exclude rotations incompatible with simulation
    rotations_to_exclude = rot_check_departure_arrival_ams + rot_check_included_flights + rot_check_airport_chain

    # If requested, also exclude rotations with overlapping scheduled flights
    if G.EXCLUDE_ROTATIONS_OVERLAPPING_FLIGHTS == True:
        rotations_to_exclude = rotations_to_exclude + rot_check_departure_time
    # If rotations with overlapping flights are not excluded, make sure that there is not overlap between them and
    # rotations to exclude
    else:
        rot_check_departure_time = [rt for rt in rot_check_departure_time if rt not in rotations_to_exclude]

    # Make the rotations to exclude a set
    rotations_to_exclude = list(set(rotations_to_exclude))

    ##### Generate report #####
    check_rotations_report = {'n_flights': rot_check_included_flights,
                              'dep_arr_ams': rot_check_departure_arrival_ams,
                              'airport_chain': rot_check_airport_chain,
                              'flights_dep_time': rot_check_departure_time,
                              'exclude': rotations_to_exclude}
    return check_rotations_report


def cross_reference_instances(obj, iteration_n):
    '''Creates cross-references between instances'''
    # Assing subtype to aircraft
    for aircraft in obj.aircraft:
        subtype = next(st for st in obj.subtypes if st.id == aircraft.subtype)
        aircraft.subtype = subtype
        aircraft.type = G.SUBTYPES_TYPES[subtype.IATA]
        subtype.aircraft.append(aircraft)

    # Rotations and flights
    for flight in obj.flights:
        rotation = next((rt for rt in obj.schedule if str(rt.id) == str(flight.rotation)), None)
        if rotation is not None:
            flight.rotation = rotation
            rotation.flights.append(flight)
        else:
            log_warning(f"Flight {flight.id} could not find rotation {flight.rotation}")

    # Order flights by leg order in rotation
    for rotation in obj.schedule:
        rotation.flights = sorted(rotation.flights, key=lambda x: x.leg_number)

    # Check that rotations include all flights
    rotations_check = check_rotations(obj.schedule)
    if rotations_check['exclude'] != [] or rotations_check['flights_dep_time'] != []:
        warning_string = 'SCHEDULE REPORT\n\n' +\
                         str(len(rotations_check['exclude']))+' rotations ' +\
                         str([rt.id for rt in rotations_check['exclude']]) +\
                         ' were excluded from the schedule.\n'+ \
                         str([rt.id for rt in rotations_check['n_flights']]) + \
                         ': excluded due to missing flights in the rotation\n' + \
                         str([rt.id for rt in rotations_check['dep_arr_ams']]) + \
                         ': excluded due to the rotation not departing or arriving in AMS\n' + \
                         str([rt.id for rt in rotations_check['airport_chain']]) + \
                         ': excluded due to a flight not departing from the arrival airport of previous flight\n'

        if G.EXCLUDE_ROTATIONS_OVERLAPPING_FLIGHTS:
            warning_string = warning_string + \
                             str([rt.id for rt in rotations_check['flights_dep_time']]) + \
                             ': excluded due to flight departing before previous scheduled arrival time.\n'
        else:
            warning_string = warning_string + '\n'+ \
                             str(len(rotations_check['flights_dep_time']))+' rotations ' +\
                             str([rt.id for rt in rotations_check['flights_dep_time']]) +\
                             ' are included, but they contain flights departing before previous scheduled arrival time.\n'

        if iteration_n == 0:
            log_error(warning_string, print_error=False)
        else:
            log_info(warning_string)

        # Remove flights from scheduled flights
        flights_to_remove = [fl for rt in rotations_check['exclude'] for fl in rt.flights]
        obj.flights = [fl for fl in obj.flights if fl not in flights_to_remove]
        # Remove rotations from schedule
        obj.schedule = [rt for rt in obj.schedule if rt not in rotations_check['exclude']]

    # Flights and airports
    for flight in obj.flights:
        airport_dep = next(ap for ap in obj.airports if ap.id == flight.airport_dep)
        flight.airport_dep = airport_dep
        airport_arr = next(ap for ap in obj.airports if ap.id == flight.airport_arr)
        flight.airport_arr = airport_arr

    # Rotation and subtype
    for rotation in obj.schedule:
        subtype = [st for st in obj.subtypes if st.IATA == rotation.subtypes]
        if subtype == [] and G.ONLY_ROTATIONS_SELECTED_SUBTYPES == 1:
            raise Exception('Subtype not found')
        rotation.subtypes = subtype

    # Flights and subtype
    for flight in obj.flights:
        subtype = [st for st in obj.subtypes if st.IATA == flight.subtypes]
        if subtype == [] and G.ONLY_ROTATIONS_SELECTED_SUBTYPES == 1:
            raise Exception('Subtype not found')
        flight.subtypes = subtype

    # TO slots and norm slots
    for slot in obj.slots_TO + obj.slots_AChecks:
        slot_norm = next(sl for sl in obj.slotsNorm if sl.id == slot.slotNorm)
        slot.slotNorm = slot_norm

    # Slots norm and aircraft
    for slot_norm in obj.slotsNorm:
        aircraft = [ac for ac in obj.aircraft if ac.id in slot_norm.ac_allowed]
        slot_norm.ac_allowed = aircraft

    # TO Slots and assigned aircraft
    slots_with_assigned_ac = [sl for sl in obj.slots_TO + obj.slots_AChecks if sl.aircraft_pre_assigned!=None]
    for slot in slots_with_assigned_ac:
        # If slot assigned to aircraft not used in simulation
        aircraft = next((ac for ac in obj.aircraft if ac.id == slot.aircraft_pre_assigned), None)
        slot.aircraft_pre_assigned = aircraft

    # LM slots and aircraft
    for slot in obj.slots_LM:
        aircraft = next(ac for ac in obj.aircraft if ac.id == slot.aircraft_pre_assigned)
        slot.aircraft_pre_assigned = aircraft
        slot.aircraft = aircraft
        slot.subtype = aircraft.subtype.IATA

    # Requirements and subtypes
    for rq in obj.requirements:
        subtypes = [st for st in obj.subtypes if st.IATA in rq.subtypes]
        if subtypes == []:
            raise Exception('No subtype was found for a requirement')
        rq.subtypes = subtypes
