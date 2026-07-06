import pandas as pd
import simpy
from numpy import random, array
from config import G, P, M, MODULES, RUN_CONFIG, RESULTS
from datetime import datetime as dt
from output.output_functions import csv_generate_or_append, write_csv_from_dataframe
from matplotlib import pyplot as plt
from math import floor, ceil
import numpy as np
import os
import sys
import logging
import timeit
from tqdm import tqdm
from numpy.random import default_rng
from simulation import modules
from datetime import timedelta, datetime, time
from classes.classes_maintenance import Slot, Task
from classes.classes_operations import Rotation, Flight, ReserveSlot
from classes.classes_aircraft import Aircraft
from statistics import mean
from simulation._module_recovery import LongAOGDutyIncompatibility
from output.output_functions import log_info, log_error, log_warning
from validation.validation_network import print_hist

# Empty class to include simulation processes
class EmptyClass:
    pass


@modules.initialize_simulation_modules()
class Anemos:
    def __init__(self, obj, data, sim_run_id, sim_iteration, iteration_start=0):
        # Scenario
        self.scenario = data.scenario
        # Network
        self.aircraft = obj.aircraft
        self.airports = self.__initialize_airports(obj.airports, data.TAT)
        self.hub = next(ap for ap in self.airports if ap.id == G.AIRPORT_BASE)
        self.rotations_schedule = obj.schedule
        self.rotations = obj.rotations
        self.reserve_slots = obj.reserve_slots
        # Maintenance
        self.slots_TO = obj.slots_TO
        self.slots_LM = obj.slots_LM
        self.slots_AChecks = obj.slots_AChecks
        self.slots = self.slots_TO + self.slots_LM + self.slots_AChecks
        self.slots_AG = []
        self.requirements = obj.requirements

        # Distributions
        self.disruptions_hub = self.__initialize_disruptions_hub(data.disruptions_hub)
        self.disruptions_outstations = self.__initialize_disruptions_outstations(data.disruptions_outstations)
        self.distributions_AOG = self.__initialize_AOG(data.distributions_AOG)
        self.distributions_NR = self.__initialize_NR(data.distributions_NR)
        self.distributions_DDs = self.__initialize_DDs_distributions(data.distributions_DDs)

        # Model environment
        self.env = simpy.Environment()
        self.start = self.scenario['Simulation_start']
        self.now = self.start
        self.id_simulation_run = sim_run_id
        self.iteration = sim_iteration
        self.iteration_start = iteration_start
        self.id = self.__find_simulation_id(sim_run_id, sim_iteration)
        # Keep conditional NR magnitude sampling off the global operations RNG. Static NR uses
        # precomputed FIFO magnitudes, so drawing predicted magnitudes from the global stream would
        # shift all later disruption/AOG draws and break paired-seed comparisons.
        base_seed = os.environ.get('ANEMOS_BASE_SEED')
        nr_seed = (int(base_seed) + sim_iteration + 10_000_019) if base_seed is not None else None
        self.nr_realization_rng = default_rng(nr_seed)
        self.AMS_disruption_state = 0
        self.proc = EmptyClass()
        self.results = self.__initialize_results()


        # Initialize network lists
        self.rotations_executed = []
        self.rotations_cancelled = []
        self.rotations_cancelled_recovery = []
        self.rotations_open = []

        # Initialize maintenance lists
        self.scheduler_maintenance = None
        self.scheduler_tail_assignment = None
        self.scheduler_tail_assignment_next_call = self.now
        self.slots_cycle_duration = self.__find_cycle_duration()
        self.slots_cycle = 0
        self.maintenance_scheduling_window_end = None
        self.slots_open = []        # slots_scheduling and still not completed slots
        self.slots_scheduling = []  # slots open for scheduling (no AG)
        self.slots_executed = []
        self.slots_unused = []
        self.slots_cancelled = []
        self.tasks_open = self.__initialize_requirements()
        if self.nr_predictor is not None:
            task_codes = [t.requirement.code for t in self.tasks_open
                          if getattr(t, 'requirement', None) is not None]
            for fleet in {str(ac.type) for ac in self.aircraft}:
                self.nr_predictor.calibrate_realized_code_mix(fleet, task_codes)
        self.DDs = self.__initialize_DDs(data.DDs)
        self.tasks = [ts for ts in self.tasks_open] + [ts for ts in self.DDs]
        self.tasks_in_progress = [] # List of tasks to be executed in a workpackage in progress
        self.tasks_executed = []
        self.tasks_missed = []
        # Completed A-check NR totals keyed by (fleet, check type). Static reservation uses the
        # plain mean of the last G.NR_STATIC_HISTORY_WINDOW completed work packages.
        self.nr_completed_workpackage_history = {}

        self.clean_values = []
        self.workpackages_anticipation = []

        # Timing attributes
        self.timer_start = timeit.default_timer()

        log_info('ANEMOS initialization done '+ self.id)


    def run(self):
        # Initialize logs of simulation
        self.__verification_validation_initialize()

        # Initialize the simulation generators
        self.__airline_operations()

        # Define progress bar for simulation
        if RUN_CONFIG.MODE == 0:
            progress_pos = 0
        else:
            progress_pos = self.iteration-self.iteration_start
        progress = tqdm(range(1,G.SIM_DURATION), file=sys.stdout, position=progress_pos, leave=True)
        # Run the simulation
        for i in progress:
            # Update progress bar every day
            progress_description = 'ANEMOS iter '+ str(self.iteration) + ': ' + self.__now_string()
            progress.set_description(progress_description)
            self.env.run(until=i* 24 * 60)

        self.__generate_final_results()
        self.__check_results()
        # Run validation code if required from config
        self.__validate_model()

        self.timer_run_time = timeit.default_timer() - self.timer_start
        log_info('Iteration', self.iteration, 'ended. Run time:', round(self.timer_run_time/60), 'min\n\n')

        recovery_call_disr = self.results['dynamic']['recovery_module_disr_call_count']
        recovery_call_aog = self.results['dynamic']['recovery_module_AOG_call_count']
        recovery_calls_total = recovery_call_aog + recovery_call_disr
        log_info('Recovery module calls total:', recovery_calls_total,
              '|| Disrutions:', recovery_call_disr, '|| AOG:', recovery_call_aog )



    # =================================================================================#
    # INITIALIZATION
    # =================================================================================#

    def __initialize_airports(self, airports, TAT):
        ''' This function samples TAT for each airport, and it return airports objects.
        The TAT duration is sampled based on the airport type (Hub, OutStationLongTAT, OutStationShortTAT) '''
        for airport in airports:
            # Find reference airport type.
            try:
                TAT_type = next(tt for tt in TAT if airport.id in tt['airports'])

            # If airport not found, use OutStationLongTAT model
            except:
                TAT_type = next(tt for tt in TAT if tt['type'] == 'OutStationLongTAT')

            # Analytical model
            TAT_model = TAT_type['TAT_fitted_dist'].model
            # Sample size: a fixed number of samples is used every 365 days of simulation
            sample_size = round(P.TAT_SAMPLING_SIZE_PER_DAY * G.SIM_DURATION)
            # Find sample
            sample = self.__sample_from_distribution(TAT_model, sample_size)
            # Remove sampled data lower than observed in reality
            sample = [round(sl) for sl in sample if sl >= TAT_type['TAT_min']]
            airport.TAT_sampled = sample
            # Find expected TAT at airport for arrival at AMS estimation
            airport.TAT_mean = self.__find_expected_value_of_distribution(TAT_model)

        return airports

    def __initialize_disruptions_hub(self, disruptions):
        '''
        This function precomputes durations of disruption events and aircraft delays for use during
        simulation. The durations are found by sampling the analytical distributions fitted to historical data.
        '''

        # DISRUPTION EVENTS DURATION
        for disr_level in disruptions.disruption_levels:
            # Analytical model
            disruption_events_model = disr_level['events_fitted_dist'].model
            # Sample size: a fixed number of samples is used every 365 days of simulation
            sample_size = round(P.EVENTS_SAMPLING_SIZE_PER_YEAR * G.SIM_DURATION/365)
            # Find sample
            sample = self.__sample_from_distribution(disruption_events_model, sample_size)
            # Add one unit to the sample since the distribution was shifted by one unit during fitting
            # (events duration > 0) and floor sample to the unit
            sample = [floor(sl+1) for sl in sample]
            # Add sample to disruptions object
            disr_level['events_sampled'] = sample

        # AIRCRAFT DELAYS DURATIONS
        for disr_level in disruptions.disruption_levels:
            # Analytical model
            delays_model = disr_level['delays_fitted_dist'].model
            # Sample size: a fixed number of samples is used every 365 days of simulation
            sample_size = round(P.DELAYS_SAMPLING_SIZE_PER_YEAR * G.SIM_DURATION / 365)
            # Find sample
            sample = self.__sample_from_distribution(delays_model, sample_size)
            # Round sample to the unit
            sample = [round(sl) for sl in sample]
            # Remove delays less or equal to zero. Before sampling a delay, a different probability
            # is used to determine if a flight will be on time or not.
            sample = [sl for sl in sample if sl>0]
            # Add sample to disruptions object
            disr_level['delays_sampled'] = sample

        # VALIDATION LISTS: empty lists filled with used values for validation
        for disr_level in disruptions.disruption_levels:
            disr_level['delays_validation'] = []
            disr_level['events_validation'] = []

        return disruptions

    def __initialize_disruptions_outstations(self, delays_outstations):
        '''
        This function precomputes delays at outstations for use during simulation.
        The durations are found by sampling the analytical distributions fitted to historical data.
        '''

        # Analytical model
        delays_model = delays_outstations['delays_fitted_dist'].model
        # Sample size: a fixed number of samples is used every 365 days of simulation
        sample_size = round(P.DELAYS_SAMPLING_SIZE_PER_YEAR * G.SIM_DURATION / 365)
        # Find sample
        sample = self.__sample_from_distribution(delays_model, sample_size)
        # Round sample to the unit
        sample = [round(sl) for sl in sample]
        # Remove delays less or equal to zero. Before sampling a delay, a different probability
        # is used to determine if a flight will be on time or not.
        sample = [sl for sl in sample if sl > 0]
        # Add sample to disruptions object
        delays_outstations['delays_sampled'] = sample

        # Add empty list for validation
        delays_outstations['delays_validation'] = []

        return delays_outstations


    def __initialize_AOG(self, distributions_AOG):
        '''
        This function precomputes durations and inter arrival time of AOG slots for use during simulation. The
        durations are found by sampling the analytical distributions fitted to historical data.
        '''
        for distr_fleet in distributions_AOG:
            # Analytical model
            model_inter_arrival = distr_fleet['time_between_AOG_fitted'].model
            model_duration = distr_fleet['AOG_duration_fitted'].model

            # Sample size: a fixed number of samples is used every 365 days of simulation
            n_fleet = len(self.aircraft)
            sample_size_inter_arrival = round(M.SAMPLING_SIZE_PER_YEAR_AOG_INTER_ARRIVAL * G.SIM_DURATION
                                              * n_fleet / 365)
            sample_size_duration = round(M.SAMPLING_SIZE_PER_YEAR_AOG_DURATION * G.SIM_DURATION
                                         * n_fleet / 365)

            # Find sample
            sample_inter_arrival = self.__sample_from_distribution(model_inter_arrival, sample_size_inter_arrival)
            sample_duration = self.__sample_from_distribution(model_duration, sample_size_duration)
            # Make sample into minutes and round it to the unit
            sample_inter_arrival = [round(sl*24*60) for sl in sample_inter_arrival] # Inter arrival is given in days
            sample_duration = [round(sl*60) for sl in sample_duration]              # Duration is given in hours
            # Remove samples less or equal to zero
            sample_inter_arrival = [sl for sl in sample_inter_arrival if sl > 0]
            sample_duration = [sl for sl in sample_duration if sl > 0]
            # Remove sample of duration larger than allowed AOG duration
            sample_duration = [sl for sl in sample_duration if sl / 60 / 24 < M.MAX_DURATION_AOG]

            # Add sample to disruptions object
            distr_fleet['time_between_AOG_sampled'] = sample_inter_arrival
            distr_fleet['AOG_duration_sampled'] = sample_duration

            # Add empty list for validation
            distr_fleet['time_between_AOG_validation'] = []
            distr_fleet['AOG_duration_validation'] = []

        return distributions_AOG

    def __initialize_NR(self, distributions_NR):
        '''
        This function precomputes Non-Routine labor hours to be executed in each maintenance slot for use during
        simulation. The durations are found by sampling the analytical distributions fitted to historical data.
        '''
        for distr_fleet in distributions_NR:
            # Analytical model
            model = distr_fleet['labor_fitted'].model

            # Sample size: a fixed number of samples is used every 365 days of simulation
            sample_size = round(M.SAMPLING_SIZE_PER_YEAR_PER_AC_NR / 365 * G.SIM_DURATION * len(self.aircraft))

            # Find sample
            sample = self.__sample_from_distribution(model, sample_size)
            # Round to two decimals
            sample = [round(sl, 2) for sl in sample]
            # Remove samples less or equal to zero and too long
            sample = [sl for sl in sample if sl > 0 and sl <= M.MAX_DURATION_NR]


            # Add sample to NR distributions dictionary
            distr_fleet['labor_fitted_sampled'] = sample

            # Add empty list for validation
            distr_fleet['labor_fitted_validation'] = []

        # PAPER_DESIGN: conditional NR prediction (inference-only). When in 'predicted' mode,
        # build the predictor once per iteration. The static FIFO sample above is left intact so
        # the per-fleet probability_NR gate stays identical between modes (preserves NR mass,
        # changes placement only -- sec. 4.4). See simulation/nr_predictor.py.
        self.nr_predictor = None
        if getattr(G, 'NR_MODE', 'static') == 'predicted':
            from simulation.nr_predictor import load_predictor
            self.nr_predictor = load_predictor()
            log_info('NR mode = predicted (conditional injection, buffer q=' +
                     str(G.NR_BUFFER_QUANTILE) + ', variance_scale=' + str(G.NR_VARIANCE_SCALE) + ')')

        return distributions_NR


    @staticmethod
    def __find_expected_value_of_distribution(distribution_model):
        ''' Given a distribution model, returns the expected value '''
        expected_value = distribution_model['distr'].expect(args = distribution_model['arg'],
                                                      loc=distribution_model['loc'],
                                                      scale=distribution_model['scale'])
        return round(expected_value)

    @staticmethod
    def __sample_from_distribution(distribution_model, sample_size):
        ''' Given a distribution model and a sampling size, returns a sample rounded to the unit as a list'''
        # Seed the generator from the (optionally seeded) legacy np.random global so that, when the
        # experiment runner sets ANEMOS_BASE_SEED, ALL distribution sampling -- NR, AOG, disruptions --
        # is reproducible and paired cells truly share the same stochastic draws (PAPER_DESIGN sec. 4.4).
        # With no base seed, np.random is entropy-seeded, so behaviour stays random as before.
        rng = default_rng(int(np.random.randint(0, 2 ** 31 - 1)))
        sample = distribution_model['distr'].rvs(*distribution_model['arg'],
                                                      loc=distribution_model['loc'],
                                                      scale=distribution_model['scale'],
                                                      size=sample_size,
                                                      random_state=rng)
        return sample

    def __get_disruption_sample(self, disruption_level_id, sample_type):
        ''' Retun one sample of disruption event duration or aircraft delay, from precomputed sample'''
        # Get correct disruption level
        disruption_level = next(dl for dl in self.disruptions_hub.disruption_levels if dl['levelId'] == disruption_level_id)
        # Get sample
        sample = disruption_level[sample_type].pop(0)
        return sample


    @staticmethod
    def __find_simulation_id(sim_run_id, sim_iteration):
        return sim_run_id + '_' +str(sim_iteration)

    def __initialize_results(self):
        ''' Initialize dictionary containing final results'''
        # Dynamic overall results
        dynamic_results = {'ac_swaps': 0,
                           'ac_swaps_2hr': 0,
                           'ac_swaps_4hr': 0,
                           'ac_swaps_6hr': 0,
                           'slot_delayed_ffs': 0,
                           'slot_swaps': 0,
                           'slot_swaps_opportunities': 0,
                           'tasks_executed': 0,
                           'tasks_missed': 0,
                           'rotations_executed': 0,
                           'slots_executed':0,
                           'slots_included_in_aog':0,
                           'recovery_module_disr_call_count': 0,
                           'recovery_module_AOG_call_count': 0
                           }

        # Health dataframe
        health = pd.DataFrame({'date': pd.date_range(start=(self.now + timedelta(days=1)).date(),
                                                     end=(self.now + timedelta(days=G.SIM_DURATION)).date(),
                                                     freq='D')})
        health['simulation_run_id'] = self.id_simulation_run
        health['simulation_id'] = self.id
        health['health'] = None
        health['checked'] = 0
        health['aircraft'] = [[ac.id for ac in self.aircraft] for x in range(len(health))]
        health = health.explode('aircraft')

        # Dictionary of all results
        results = {'dynamic':dynamic_results,
                   'health':health}

        # Recovery results
        filename = RESULTS.FILE_NAMES['recovery_kpis']+'_'+self.id
        csv_generate_or_append(filename,
                               ['simulation_run_id', 'simulation_id', 'call_id', 'call_time','call_reason',
                                'ac_swaps', 'ac_swaps_2hr', 'ac_swaps_4hr', 'ac_swaps_6hr',
                                'rotations_cancelled', 'rotations_delayed',
                                'slot_swaps', 'slot_swaps_opportunities', 'slot_swaps_opportunities_per_slot',
                                'slot_postponement_opportunities_per_slot', 'slot_swap_opportunities_per_slot_today',
                                'slot_postponement_opportunities_per_slot_today',
                                'slots_cancelled', 'slots_delayed', 'slots_delayed_ffs'],
                               simulation_id=self.id_simulation_run)

        return results


    def __get_duty_id(self):
        now = datetime.now().strftime("%d/%m/%Y_%H:%M:%S:%f")
        now_env = str(self.env.now)
        id = now + '_' + now_env
        return id

    def __find_cycle_duration(self):
        ''' This function finds the duration of the slots cycle used in the simulation'''
        if self.slots != []:
            slot = next(sl for sl in self.slots if sl.slotNorm!=None)
            cycle_duration = slot.slotNorm.cycle_duration
        else:
            cycle_duration = G.SIM_DURATION + 1
        return cycle_duration

    def __initialize_requirements(self):
        ''' This function initialize the requirements for each registration by creating the first instance '''
        tasks_list = []
        horizon_end = self.scenario['Simulation_start'] + timedelta(days=G.SIM_DURATION)

        for requirement in self.requirements:
            aircraft_of_requirement = [ac for ac in self.aircraft if ac.subtype in requirement.subtypes]
            for aircraft in aircraft_of_requirement:
                if G.MAINTENANCE_SCHEDULE_ONE_SHOT:
                    tasks = requirement.generate_instances_for_horizon(
                        aircraft=aircraft,
                        sim_start=self.scenario['Simulation_start'],
                        sim_end=horizon_end
                    )
                else:
                    tasks = [requirement.generate_instance(
                        'first',
                        aircraft=aircraft,
                        sim_start=self.scenario['Simulation_start']
                    )]
                aircraft.tasks_open.extend(tasks)
                tasks_list.extend(tasks)

        return tasks_list

    def __initialize_DDs_distributions(self,distributions_DDs):
        '''
        Initialize the data on DD inter arrival time and DD count by adding lists for validation and by computing
        the intervals for each choice value. The computation of the intervals is necessary because the weighted
        numpy.random.choice() function trows an exception with the current numpy version
        '''

        # Add lists to distributions_DDs dictionary for validation
        for distr_fleet in distributions_DDs:
            distr_fleet['inter_arrival_time_validation'] = []
            distr_fleet['dd_count_validation'] = []

        # Inter arrival time: Find intervals for choice of weighted values
        for distr_fleet in distributions_DDs:
            distr_fleet['inter_arrival_time_p_intervals'] = {}
            interval_min = 0
            iat_min = min(distr_fleet['inter_arrival_time_probabilities'].keys())
            iat_max = max(distr_fleet['inter_arrival_time_probabilities'].keys())
            for iat in range(iat_min, iat_max+1):
                interval_max = interval_min + distr_fleet['inter_arrival_time_probabilities'][iat]
                distr_fleet['inter_arrival_time_p_intervals'][iat] = (interval_min, interval_max)
                interval_min = interval_max

        # DDs count: Find intervals for choice of weighted values
        for distr_fleet in distributions_DDs:
            distr_fleet['dd_count_p_intervals'] = {}
            interval_min = 0
            iat_min = min(distr_fleet['dd_count_probabilities'].keys())
            iat_max = max(distr_fleet['dd_count_probabilities'].keys())
            for iat in range(iat_min, iat_max + 1):
                interval_max = interval_min + distr_fleet['dd_count_probabilities'][iat]
                distr_fleet['dd_count_p_intervals'][iat] = (interval_min, interval_max)
                interval_min = interval_max
        return distributions_DDs

    def __initialize_DDs(self, df_DDs):
        ''' Initialize Deferred Defects (MEL and NSRE tasks), and ADHOC tasks by generating for each aircraft the
        deferred defects that will arrive during the simulated period. Return a list of all DDs and adds DDs to each
        aircraft's list of DDs '''

        # List of all the generated DDs
        DDs = []
        date_start = (self.now - timedelta(days=30)).date()
        date_end =  (self.now + timedelta(days=G.SIM_DURATION+30)).date()

        # Generate DDs for each simulated aircraft
        for aircraft in self.aircraft:
            # Initialize count for DD id
            aircraft_DD_id = 0
            # Initialize list of future DDs for the aircraft
            aircraft.tasks_DDs_future = []
            # Find correct data set
            distributions = next(distr for distr in self.distributions_DDs if distr['fleet'] == aircraft.type)
            # Initialize last arrival day to one month ago
            arrival_day = date_start

            # Find DDs until one month after simulation
            while arrival_day <= date_end:
                # Sample inter-arrival time. Note that the following lines do not work due to a bug in numpy. A
                # custom funciton is therefore built
                # inter_arrival_time = random.choice(a=list(distributions['inter_arrival_time_probabilities'].keys()),
                #                                    p=list(distributions['inter_arrival_time_probabilities'].values()))
                inter_arrival_time = self.__get_weighted_choice_sample(distributions['inter_arrival_time_p_intervals'])
                # Add value to validation list
                distributions['inter_arrival_time_validation'].append(inter_arrival_time)
                # Update arrival day
                arrival_day = arrival_day + timedelta(days=inter_arrival_time)

                # If day is after start of simulation, add DDs
                if arrival_day >= self.now.date():
                    # Sample how many tasks should be generated
                    arrived_DDs = self.__get_weighted_choice_sample(distributions['dd_count_p_intervals'])
                    # Add value to validation list
                    distributions['dd_count_validation'].append(arrived_DDs)

                    # Select some rows from DDs tasks dataframe
                    DDs_selected = df_DDs[df_DDs['ac_fleet']==aircraft.type].sample(n=arrived_DDs)

                    # Generate selected DDs
                    for index, DD_row in DDs_selected.iterrows():
                        # Id of the new DD
                        DD_id = 'DD_'+aircraft.id + '_' + str(aircraft_DD_id)
                        aircraft_DD_id += 1

                        # Task type
                        if DD_row['task_type'] == 'CORR' and 'MEL' in DD_row['deferral_class']:
                            task_type = 'MEL'
                        elif DD_row['task_type'] == 'CORR' and 'NSRE' in DD_row['deferral_class']:
                            task_type = 'NSRE'
                        elif DD_row['task_type'] == 'ADHOC':
                            task_type = 'ADHOC'
                        else:
                            raise Exception('Task type not supported')

                        # Arrival and due date
                        date_arrival = datetime.combine(arrival_day, time(3,0))
                        date_due = arrival_day + timedelta(days=DD_row['deferral_days'])
                        date_due = datetime.combine(date_due, time(23, 59))
                        date_arrival = G.TIMEZONE_UTC.localize(date_arrival)
                        date_due = G.TIMEZONE_UTC.localize(date_due)

                        # Generate DD
                        DD_new = Task(id = DD_id,
                                      durationEst = timedelta(hours=DD_row['duration']),
                                      laborEst = timedelta(hours=DD_row['labor_sched']),
                                      laborAct = timedelta(hours=DD_row['labor_act']),
                                      aircraft = aircraft,
                                      dateArrival = date_arrival,
                                      dateReady = date_arrival,
                                      dateDue = date_due,
                                      workType = DD_row['task_work_type'],
                                      type = task_type,
                                      info=DD_row['deferral_class'])

                        # Append task to relevant lists
                        DDs.append(DD_new)
                        aircraft.tasks_DDs_future.append(DD_new)

        return DDs

    @staticmethod
    def __get_weighted_choice_sample(weights_intervals):
        '''
        Return a sample from a weighted choice described in dictionary weights_interval.

        Given a dictionary in the form {choice1: (0, max_range_1), choice2: (max_range_1, max_range_2),
        choice_n: (max_range_n-1, 1)}, where (max_range_n - max_range_n-1) is the probability of observing choice n,
        returns a randomly selected value among the available choices.
        '''
        sample_uniform = random.uniform()
        choice = next(key for key in weights_intervals.keys()
                      if sample_uniform >= weights_intervals[key][0] and sample_uniform <= weights_intervals[key][1])
        return choice

    # =================================================================================#
    # TIME MANAGEMENT
    # =================================================================================#
    def _update_now(self): # Cannot be __ otherwise cannot be called from added modules
        self.now = self.start + timedelta(minutes=self.env.now)

    def __now_string(self):
        ''' Returns the simulated now as string 'YYYY-MM-DD HH:MM' '''
        self._update_now()
        return dt.strftime(self.now, '%Y-%m-%d %H:%M')

    def __get_now(self):
        ''' Update now and return it'''
        self._update_now()
        return self.now

    def __time_from_now(self, target_time):
        ''' Returns the number of minutes between the simulated current time and an input datetime '''
        # Update current simulated time
        self._update_now()
        # Define time from now
        time_from_now = self.__minutes_between_datetimes(self.now, target_time)
        return time_from_now

    @staticmethod
    def __minutes_between_datetimes(date1, date2):
        ''' Given to datetimes, return the number of minutes between them (date1 is earlier than date2)'''
        return (date2 - date1).total_seconds()/60

    @staticmethod
    def __minutes_from_timedelta(duration):
        ''' Given a duration as timedelta returns the corresponding minutes'''
        return duration.total_seconds()/60

    @staticmethod
    def __minutes_from_days(days):
        ''' Given a number of days, find the corresponding number of minutes'''
        return days * 24 * 60

    @staticmethod
    def __days_between_dates(datetime1, datetime2):
        ''' Given two datetimes, find the number of calendar days between the two'''
        date1 = datetime1.date()
        date2 = datetime2.date()
        days_between_dates = (date2-date1).days
        return days_between_dates

    # =================================================================================#
    # SIMULATION LOGS
    # =================================================================================#
    def __log_sim(self, aircraft, duty, start_or_end, id=None,first_flight=False, delayed_flight=None,
                  cancellation_reason=None):

        # If log should not be produced, return
        if G.SAVE_SIMULATION_GANTT == 0:
            return

        filename = 'log_sim_'+self.id
        col_simulation_time = None
        # Cancelled rotation
        if aircraft == 'disruption_state':
            col_registration = ''
            col_ac_subtype = 'AMS state'
        elif aircraft == 'cancelled' and isinstance(duty, Rotation):
            col_registration = duty.rotation_norm.id
            col_ac_subtype = 'Cancelled'
        elif aircraft == 'cancelled' and isinstance(duty, Flight):
            col_registration = duty.rotation.rotation_norm.id
            col_ac_subtype = 'Cancelled'
        else:
            col_registration = aircraft.id
            col_ac_subtype = aircraft.subtype.IATA

        # Executed duties
        col_notes = None
        if isinstance(duty, Slot):
            col_duty_type = 'Maintenance_slot'
            if duty.remarks == 'AG':
                col_color = 'AOG'
            elif duty.location == 'H':
                col_color = 'maintenance - hangar'
            elif duty.location == 'P':
                col_color = 'maintenance - platform'
            col_duty_id = duty.id
            col_label = duty.remarks
            col_view_mode = 'flights'
        elif isinstance(duty, Flight):
            col_duty_type = 'Flight'
            if aircraft == 'cancelled':
                col_color = 'cancelled_'+cancellation_reason
            else:
                col_color = 'flight or rotation'
            col_duty_id = duty.id
            col_label = duty.id
            col_view_mode = 'flights'
        elif isinstance(duty, Rotation):
            col_duty_type = 'Rotation'
            if aircraft == 'cancelled':
                col_color = 'cancelled_'+cancellation_reason
            else:
                col_color = 'flight or rotation'
            col_duty_id = duty.id
            col_label = duty.id
            col_view_mode = 'rotations'
        elif duty == 'delay':
            col_duty_type = 'Delay'
            col_color = 'delay'
            col_duty_id = 'delay'+str(id)
            col_label = ''
            col_view_mode = 'flights'
        elif duty == 'towing':
            col_duty_type = 'Towing'
            col_color = 'towing'
            col_duty_id = 'towing'+str(id)
            col_label = ''
            col_view_mode = 'flights'
        elif duty == 'disruption_state':
            col_duty_type = 'Disruption state'
            col_color = 'AMS disruption '+str(self.AMS_disruption_state)
            col_duty_id = 'disruption_state'+str(id)
            col_label = ''
            col_view_mode = 'flights'

        else:
            raise Exception('Duty type not accepted')

        type_output = 'log'
        start_end = start_or_end

        # Start and end columns
        if aircraft == 'cancelled':
            col_time_start = datetime.strftime(duty.dep_sched,' %d/%m/%Y %H:%M')
            col_time_end = datetime.strftime(duty.arr_sched,' %d/%m/%Y %H:%M')
        elif duty=='delay' and start_end=='start':
            col_time_start = datetime.strftime(delayed_flight.dep_sched,' %d/%m/%Y %H:%M')
            col_time_end = None
        else:
            now_str = datetime.strftime(self.__get_now(),' %d/%m/%Y %H:%M')
            if start_end == 'start':
                col_time_start = now_str
                col_time_end = None
            elif start_end == 'end':
                col_time_start = None
                col_time_end = now_str
            else:
                raise Exception('start_or_end not supported')


        self.__csv_gantt(filename=filename, col_simulation_time=col_simulation_time, col_registration=col_registration,
                         col_notes=col_notes, col_duty_type=col_duty_type, col_duty_id=col_duty_id,
                         col_time_start=col_time_start, col_time_end=col_time_end,col_label=col_label, col_color=col_color,
                         col_ac_subtype=col_ac_subtype, type_output=type_output, start_end=start_end, view_mode=col_view_mode)

        # If slot, towing or delay, also draw on the line of rotations
        if isinstance(duty, Slot) or (duty=='delay' and first_flight==True) or (duty=='towing') \
                or (duty=='disruption_state'):
            self.__csv_gantt(filename=filename, col_simulation_time=col_simulation_time,
                             col_registration=col_registration, col_notes=col_notes, col_duty_type=col_duty_type,
                             col_duty_id=col_duty_id, col_time_start=col_time_start, col_time_end=col_time_end,
                             col_label=col_label,  col_color=col_color, col_ac_subtype=col_ac_subtype,
                             type_output=type_output, start_end=start_end, view_mode='rotations')

    # =================================================================================#
    # LOG FOR MAP
    # =================================================================================#

    def __log_discrete_time_initialize(self):
        ''' Generate a csv log file for the aircraft position at fixed time steps'''
        log_filename = 'log_map_'+ self.id
        csv_generate_or_append(log_filename,
                               ['TimeSimulation',
                                'Time',
                                'AircraftId',
                                'Subtype',
                                'Latitude',
                                'Longitude',
                                'State',
                                'Flight',
                                'FlightOrig',
                                'FlightDest'
                                ],
                               simulation_id= self.id_simulation_run,
                               output_type='log_map')

    def __log_discrete_time_update(self):
        ''' Update a csv file with the events of the simulation'''
        # Find log name
        log_name = 'log_' + self.id
        # Find column values
        col_time_simulation = self.env.now
        col_time = self.__now_string()
        for aircraft in self.aircraft:
            col_aircraft = aircraft.id
            col_subtype = aircraft.subtype.IATA
            col_latitude = aircraft.coordinates['latitude']
            col_longitude = aircraft.coordinates['longitude']
            col_state = aircraft.state
            if col_state == 'flying':
                # Update aircraft coordinates
                aircraft.update_coordinates_flying()
                flight = aircraft.flights[-1]
                col_flight = flight.id
                col_ap_orig = flight.airport_dep.id
                col_ap_dest = flight.airport_arr.id
            else:
                col_flight = ''
                col_ap_orig =''
                col_ap_dest =''
            # Append row to log
            csv_generate_or_append(log_name,
                                   [col_time_simulation,
                                    col_time,
                                    col_aircraft,
                                    col_subtype,
                                    col_latitude,
                                    col_longitude,
                                    col_state,
                                    col_flight,
                                    col_ap_orig,
                                    col_ap_dest],
                                   simulation_id=self.id_simulation_run,
                                   output_type='log'
                                   )

    def __log_discrete_time_generator(self):
        ''' Generator for logging at regular time interval'''
        while True:
            self.__log_discrete_time_update()
            yield self.env.timeout(G.LOG_DISCRETE_TIME_STEP)

    # =================================================================================#
    # VERIFICATION & VALIDATION
    # =================================================================================#
    def __verification_validation_initialize(self):
        ''' Initialize file for model verification and validation'''
        # Gantt of simulation
        if G.SAVE_SIMULATION_GANTT == 1 or self.iteration == 0:
            self.__gantt_initialize(gantt_type='log')
        # Verification & validation output, if requested
        if G.LOG_DISCRETE_TIME == 1:
            self.__log_discrete_time_initialize()
        # Initialize logs if requested for validation, or if first iteration for always having at least one file
        if 3 in G.VERIFICATION_VALIDATION or self.iteration == 0:
            self.__gantt_initialize(gantt_type='tail_assignment')
        if 4 in G.VERIFICATION_VALIDATION or self.iteration == 0:
            self.__gantt_initialize(gantt_type='recovery')

    def __csv_gantt(self,
                    filename,
                    col_simulation_time,
                    col_registration,
                    col_notes,
                    col_duty_type,
                    col_duty_id,
                    col_time_start,
                    col_time_end,
                    col_label,
                    col_color,
                    col_ac_subtype,
                    type_output,
                    start_end = 'start_end',
                    view_mode = 'flights',
                    line = 'orig',
                    model_id = ''):
        '''
        Generate two new lines for a gantt log
        type: 'tail_assignment', 'log'
        start_end: 'start_end', 'start', 'end'
        '''

        # Find destination folder
        if type_output == 'tail_assignment':
            output_type = 'verification_validation'
        elif type_output == 'log':
            output_type = 'log_sim'
        elif type_output == 'recovery':
            output_type = 'recovery'
        else:
            raise Exception('Type not supported')

        if start_end == 'start' or start_end == 'start_end':
            csv_generate_or_append(filename,
                                    [self.id,
                                    self.id_simulation_run,
                                    self.iteration,
                                    col_simulation_time,
                                    col_registration,
                                    col_notes,
                                    col_duty_type,
                                    col_duty_id,
                                    'time_start',
                                    col_time_start,
                                    col_label,
                                    col_color,
                                    col_ac_subtype,
                                    view_mode,
                                    line,
                                    model_id],
                                    simulation_id=self.id_simulation_run,
                                    output_type=output_type)

        if start_end == 'end' or start_end == 'start_end':
            csv_generate_or_append(filename,
                                [self.id,
                                self.id_simulation_run,
                                self.iteration,
                                col_simulation_time,
                                col_registration,
                                col_notes,
                                col_duty_type,
                                col_duty_id,
                                'time_end',
                                col_time_end,
                                '',
                                col_color,
                                col_ac_subtype,
                                view_mode,
                                line,
                                model_id],
                                simulation_id=self.id_simulation_run,
                                output_type=output_type)

    def __gantt_initialize(self, gantt_type):
        '''
        Generate a csv file for Gantt representation
        type: 'tail_assignment', 'log'
        '''
        if gantt_type == 'tail_assignment':
            type_name = 'tail_assignment_'
            output_type = 'verification_validation'
        elif gantt_type == 'log':
            type_name = 'log_sim_'
            output_type = 'log_sim'
        elif gantt_type == 'recovery':
            type_name = 'recovery_'
            output_type='verification_validation'
        else:
            raise Exception('Type not supported')
        filename =  type_name + self.id

        csv_generate_or_append(filename,
                               ['simulation_id',
                                'simulation_run_id',
                                'iteration',
                                'Simulation_time',
                                'Registration',
                                'Notes',
                                'Duty_type',
                                'Duty_id',
                                'Time_type',
                                'Time_value',
                                'Label',
                                'Color',
                                'Aircraft_subtype',
                                'View_mode',
                                'Line',
                                'Model_id'
                                ],
                               simulation_id=self.id_simulation_run,
                               output_type=output_type)



    def __gantt_print_empty_row(self, filename, col_simulation_time, aircraft):
        csv_generate_or_append(filename,
                               [col_simulation_time,
                                aircraft.id,
                                '',
                                '',
                                '',
                                '',
                                '',
                                '',
                                '',
                                aircraft.subtype.IATA],
                               simulation_id=self.id_simulation_run,
                               output_type='verification_validation')
    @staticmethod
    def datetime_to_string(dt_to_convert):
        return dt_to_convert.strftime('%Y-%m-%d %H:%M:%S')


    def __gantt_print_segments(self, aircraft, segments, filename, col_simulation_time, output_type,
                               model, line = None):
        ''' Print the csv line'''
        if segments == []:
            self.__gantt_print_empty_row(filename, col_simulation_time, aircraft)
        for segment in segments:
            if type(segment) == Rotation:
                time_dep = self.datetime_to_string(segment.dep_act)
                time_arr = self.datetime_to_string(segment.arr_act)
            else:
                time_dep = self.datetime_to_string(segment.dep_sched)
                time_arr = self.datetime_to_string(segment.arr_sched)
            col_registration = aircraft.id
            col_ac_subtype = aircraft.subtype.IATA
            col_duty_id = segment.id
            col_time_start = time_dep
            col_time_end = time_arr
            col_label = segment.id

            # NOTES AND DUTY TYPE
            if type(segment) == Rotation:
                col_notes = 'Rotation'
                col_duty_type = 'Rotation'
            elif type(segment) == ReserveSlot:
                col_notes = 'RS'
                col_duty_type = 'RS'
            else:
                raise Exception('Segment type not supported')


            # COLOR COLUMN AND LABEL
            if type(segment) == ReserveSlot:
                col_color = 'RS'

            elif type(segment) == Rotation and output_type == 'tail_assignment':
                subtypes_rotation = list(set([st.IATA for st in segment.rotation_norm.subtypes]))
                subtypes_second_choice = []
                for st in subtypes_rotation:
                    second_choices = next(sc for sc in G.PREFERRED_SUBTYPES_GROUPS if st in sc)
                    subtypes_second_choice = subtypes_second_choice + second_choices
                subtypes_second_choice = list(set(subtypes_second_choice))
                if aircraft.subtype.IATA in subtypes_rotation:
                    col_color = 'rotation_assigned_preference_best'
                elif aircraft.subtype.IATA in subtypes_second_choice:
                    col_color = 'rotation_assigned_preference_mid'
                else:
                    col_color = 'rotation_assigned_preference_low'
                col_label = col_label + '_' + str(subtypes_rotation)

            elif type(segment) == Rotation and output_type == 'recovery':
                if segment not in model.model.set_rotations:
                    col_color = 'rotation_fixed'
                elif line == 'orig':
                    col_color = 'rotation'
                elif line == 'final' and segment._val_recovery_prec_assignment == None:
                    col_color = 'rotation_assign_unchanged'
                elif line == 'final' and segment._val_recovery_prec_assignment != None:
                    col_color = 'rotation_assign_changed'
                    col_label = col_label + '_' + segment._val_recovery_prec_assignment.id
                    # Initialize precedent assignment to None again
                    segment._val_recovery_prec_assignment = None
                else:
                    raise Exception('Condition not supported')
            else:
                raise Exception('Output type not supported')

            # Append row to csv
            self.__csv_gantt(filename=filename,
                             col_simulation_time=col_simulation_time,
                             col_registration=col_registration,
                             col_notes=col_notes,
                             col_duty_type=col_duty_type,
                             col_duty_id=col_duty_id,
                             col_time_start=col_time_start,
                             col_time_end=col_time_end,
                             col_label=col_label,
                             col_color=col_color,
                             col_ac_subtype=col_ac_subtype,
                             type_output=output_type,
                             view_mode = 'flights',
                             line = line,
                             model_id=model.name)


            # PRINT DELAYS
            col_label_delay = 'delay_' + col_label
            col_notes_delay = 'delay'
            col_color_delay = 'delay'
            col_duty_type_delay = 'delay'
            col_duty_id_delay = 'delay_' + col_duty_id

            # If segment is current duty (or last duty) print delay
            if output_type == 'recovery' and (segment == aircraft.duty_current or segment == aircraft.duty_last) \
                    and segment.arr_sched<segment.arr_act:
                arr_sched = self.datetime_to_string(segment.arr_sched)
                arr_act = self.datetime_to_string(segment.arr_act)
                self.__csv_gantt(filename=filename,
                                 col_simulation_time=col_simulation_time,
                                 col_registration=col_registration,
                                 col_notes=col_notes_delay,
                                 col_duty_type=col_duty_type_delay,
                                 col_duty_id=col_duty_id_delay,
                                 col_time_start=arr_sched,
                                 col_time_end=arr_act,
                                 col_label=col_label_delay,
                                 col_color=col_color_delay,
                                 col_ac_subtype=col_ac_subtype,
                                 type_output=output_type,
                                 view_mode='flights',
                                 line=line,
                                 model_id=model.name)

            # If segment in recovery window is delayed, print delay
            elif output_type == 'recovery' and segment in model.model.set_rotations and segment.dep_sched < segment.dep_act:
                dep_sched = self.datetime_to_string(segment.dep_sched)
                dep_act = self.datetime_to_string(segment.dep_act)
                self.__csv_gantt(filename=filename,
                                 col_simulation_time=col_simulation_time,
                                 col_registration=col_registration,
                                 col_notes=col_notes_delay,
                                 col_duty_type=col_duty_type_delay,
                                 col_duty_id=col_duty_id_delay,
                                 col_time_start=dep_sched,
                                 col_time_end=dep_act,
                                 col_label=col_label_delay,
                                 col_color=col_color_delay,
                                 col_ac_subtype=col_ac_subtype,
                                 type_output=output_type,
                                 view_mode='flights',
                                 line=line,
                                 model_id=model.name)



    def __gantt_print_unassigned_segments(self, segments_unassigned, filename, col_simulation_time, output_type,
                                          model, reason='', line='orig'):
        for segment in segments_unassigned:
            time_dep = self.datetime_to_string(segment.dep_sched)
            time_arr = self.datetime_to_string(segment.arr_sched)

            col_time_start = time_dep
            col_time_end = time_arr
            col_duty_id = segment.id
            col_label = segment.id

            if type(segment) == Rotation:
                col_notes = 'Rotation'
                col_registration = 'unassigned '+segment.rotation_norm.id
                subtypes_rotation = list(set([st.IATA for st in segment.rotation_norm.subtypes]))
                col_color = 'rotation_unassigned_'+reason
                col_duty_type = 'Rotation'
                if output_type == 'recovery_final':
                    col_label = col_label + '_' + segment._val_recovery_prec_assignment.id
                    segment._val_recovery_prec_assignment = None
                elif output_type == 'tail_assignment':
                    col_label =  col_label + '_' + str(subtypes_rotation)
            elif type(segment) == ReserveSlot:
                col_notes = 'RS'
                col_registration = 'unassigned reserve slot'
                col_label = segment.id
                col_color = 'RS_unassigned'
                col_duty_type = 'RS'

            else:
                raise Exception('Segment type not supported')


            col_ac_subtype = 'unassigned'
            self.__csv_gantt(filename=filename,
                             col_simulation_time=col_simulation_time,
                             col_registration=col_registration,
                             col_notes=col_notes,
                             col_duty_type=col_duty_type,
                             col_duty_id=col_duty_id,
                             col_time_start=col_time_start,
                             col_time_end=col_time_end,
                             col_label=col_label,
                             col_color=col_color,
                             col_ac_subtype=col_ac_subtype,
                             type_output=output_type,
                             view_mode='flights',
                             line=line,
                             model_id=model.name)


    def __gantt_print_slots(self, aircraft, slots, filename, col_simulation_time, output_type, model, line='orig'):
        for slot in slots:
            if aircraft!=None:
                time_dep = self.datetime_to_string(slot.dateStart_final)
                time_arr = self.datetime_to_string(slot.dateEnd_final)
            else:
                time_dep = self.datetime_to_string(slot.dateStart_init)
                time_arr = self.datetime_to_string(slot.dateEnd_init)
            col_notes = slot.remarks
            col_duty_type = 'MaintenanceSlot'
            col_duty_id = slot.id
            col_time_start = time_dep
            col_time_end = time_arr
            col_label = slot.remarks + '_' + slot.id

            if aircraft!=None:
                col_registration = aircraft.id
                col_ac_subtype = aircraft.subtype.IATA
            else:
                col_registration = 'unassigned '+ str(slot.slotNorm.id)
                col_ac_subtype = 'unassigned'

            col_color = slot.location
            if slot.remarks == 'AG':
                col_color = 'AOG'
            elif output_type == 'recovery' and slot not in model.model.set_slots:
                col_color = col_color + '_fixed'
            elif aircraft == None:
                col_color = col_color + '_cancelled'
            elif slot._val_recovery_ffs == True:
                col_label = col_label +  '_' +self.datetime_to_string(slot.dateStart_original)\
                            + '_' + self.datetime_to_string(slot.dateEnd_original)
                col_color = col_color + '_ffs'
                slot._val_recovery_ffs = False
            elif slot._val_recovery_prec_assignment != None:
                col_label = col_label + '_' + slot._val_recovery_prec_assignment.id
                col_color = col_color + '_swap'
                slot._val_recovery_prec_assignment = None

            self.__csv_gantt(filename=filename,
                             col_simulation_time=col_simulation_time,
                             col_registration=col_registration,
                             col_notes=col_notes,
                             col_duty_type=col_duty_type,
                             col_duty_id=col_duty_id,
                             col_time_start=col_time_start,
                             col_time_end=col_time_end,
                             col_label=col_label,
                             col_color=col_color,
                             col_ac_subtype=col_ac_subtype,
                             type_output=output_type,
                             view_mode='flights',
                             line=line,
                             model_id=model.name)

            # PRINT DELAY
            col_label_delay = 'delay_' + col_label
            col_notes_delay = 'delay'
            col_color_delay = 'delay'
            col_duty_type_delay = 'delay'
            col_duty_id_delay = 'delay_' + col_duty_id
            # If slot is current duty (or last duty) print delay
            if aircraft!=None and output_type == 'recovery'\
                    and (slot == aircraft.duty_current or slot == aircraft.duty_last) \
                    and slot.dateEnd_init < slot.dateEnd_final and slot.remarks != 'AG':
                arr_sched = self.datetime_to_string(slot.dateEnd_init)
                arr_act = self.datetime_to_string(slot.dateEnd_final)
                self.__csv_gantt(filename=filename,
                                 col_simulation_time=col_simulation_time,
                                 col_registration=col_registration,
                                 col_notes=col_notes_delay,
                                 col_duty_type=col_duty_type_delay,
                                 col_duty_id=col_duty_id_delay,
                                 col_time_start=arr_sched,
                                 col_time_end=arr_act,
                                 col_label=col_label_delay,
                                 col_color=col_color_delay,
                                 col_ac_subtype=col_ac_subtype,
                                 type_output=output_type,
                                 view_mode='flights',
                                 line=line,
                                 model_id=model.name)

            # If slot in recovery window is delayed, print delay
            elif aircraft != None and output_type == 'recovery' \
                    and slot in model.model.set_slots and slot.dateStart_init < slot.dateStart_final:
                dep_sched = self.datetime_to_string(slot.dateStart_init)
                dep_act = self.datetime_to_string(slot.dateStart_final)
                self.__csv_gantt(filename=filename,
                                 col_simulation_time=col_simulation_time,
                                 col_registration=col_registration,
                                 col_notes=col_notes_delay,
                                 col_duty_type=col_duty_type_delay,
                                 col_duty_id=col_duty_id_delay,
                                 col_time_start=dep_sched,
                                 col_time_end=dep_act,
                                 col_label=col_label_delay,
                                 col_color=col_color_delay,
                                 col_ac_subtype=col_ac_subtype,
                                 type_output=output_type,
                                 view_mode='flights',
                                 line=line,
                                 model_id=model.name)

    def __tail_assignment_verification_update(self, reason, model):
        ''' Update csv file with maintenance slots and flights '''
        output_type = 'tail_assignment'
        scheduler = self.scheduler_tail_assignment.model
        filename = output_type + '_' + self.id
        # Find columns values
        col_simulation_time = self.__now_string()
        for aircraft in scheduler.setAircraft:
            ##### ASSIGNED SEGMENTS #####
            rotations_assigned = [rt for (rt, ac) in scheduler.dvRotAc
                                  if ac==aircraft and round(scheduler.dvRotAc[rt, ac]())==1]
            reserve_slots_assigned = [rs for (rs, ac) in scheduler.dvReserveAc
                                      if ac==aircraft and round(scheduler.dvReserveAc[rs, ac]())==1]
            segments = rotations_assigned + reserve_slots_assigned
            # Order segments
            segments = sorted(segments, key=lambda x: x.dep_sched)
            # Print gantt for segments
            self.__gantt_print_segments(aircraft=aircraft, segments=segments, filename=filename,
                                        col_simulation_time=col_simulation_time, output_type=output_type,
                                        model=model)

        # UNASSIGNED ROTATIONS
        rotations_unassigned = [rt for rt in scheduler.setRotations if round(scheduler.dvRotUnassign[rt]())==1]
        reserve_slots_unassigned = [rs for rs in scheduler.setReserveSlots
                                    if round(scheduler.dvReserveUnassign[rs]())==1]
        segments_unassigned = rotations_unassigned + reserve_slots_unassigned
        # Print gantt for unassigned segments
        self.__gantt_print_unassigned_segments(segments_unassigned=segments_unassigned, filename=filename,
                                               col_simulation_time=col_simulation_time, output_type=output_type,
                                               reason=reason, line='orig', model=model)

        # MAINTENANCE SLOTS
        for aircraft in self.aircraft:
            slots = aircraft.slots
            self.__gantt_print_slots(aircraft=aircraft, slots=slots, filename=filename,
                                     col_simulation_time=col_simulation_time, output_type=output_type, line='orig',
                                     model=model)


    def __verification_recovery_update(self, line, recovery):
        ''' Update csv file with original and final solution of the recovery module '''
        output_type = 'recovery'
        filename = output_type+'_' + self.id
        print_time_limit = self.__get_now() + timedelta(days=G.RECOVERY_WITHIN_DAYS + 2)
        # Find columns values
        col_simulation_time = self.__now_string()
        # ROTATIONS and RESERVE SLOTS
        for aircraft in self.aircraft:
            rotations = [rt for rt in aircraft.rotations if rt.dep_sched<print_time_limit]
            reserve_slots = [rs for rs in aircraft.reserve_slots if rs.dep_sched<print_time_limit]
            segments = rotations + reserve_slots
            # Add last duty of the aircraft if not included in list
            if type(aircraft.duty_last) == Rotation and aircraft.duty_last not in segments:
                segments.append(aircraft.duty_last)
            # segments = sorted(segments, key=lambda x:x.dep_sched)
            # Print rows for segments
            self.__gantt_print_segments(aircraft=aircraft, segments=segments, filename=filename,
                                        col_simulation_time=col_simulation_time, output_type=output_type,
                                        line=line, model=recovery)


        # CANCELLED ROTATIONS: only in resulting solution
        if line == 'final':
            rotations_cancelled = [rt for rt in recovery.model.set_rotations
                                   if round(recovery.model.dv_rotation_cancelled[rt]())==1]
            self.__gantt_print_unassigned_segments(segments_unassigned=rotations_cancelled, filename=filename,
                                                   col_simulation_time=col_simulation_time, output_type=output_type,
                                                   line=line, model=recovery)


        # MAINTENANCE SLOTS
        for aircraft in self.aircraft:
            slots = [sl for sl in aircraft.slots if sl.dateStart_init<print_time_limit]
            # Add last duty of the aircraft if not included in list
            if type(aircraft.duty_last) == Slot and aircraft.duty_last not in slots:
                slots.append(aircraft.duty_last)
            slots = sorted(slots, key=lambda x: x.dateStart_final)
            self.__gantt_print_slots(aircraft=aircraft, slots=slots, filename=filename,
                                     col_simulation_time=col_simulation_time, output_type=output_type, line=line,
                                     model=recovery)

        # CANCELLED MAINTENANCE SLOTS
        slots_cancelled = [sl for sl in recovery.model.set_slots if round(recovery.model.dv_slot_cancelled[sl]())==1]
        self.__gantt_print_slots(aircraft=None, slots=slots_cancelled, filename=filename,
                                 col_simulation_time=col_simulation_time, output_type=output_type, line=line,
                                 model=recovery)
    # =================================================================================#
    # =================================================================================#
    # DISCRETE EVENT SIMULATION
    # =================================================================================#
    # =================================================================================#

    def __airline_operations(self):
        ''' Function that initialize all simulation generators'''
        # Initialize discrete time log generator, if requested
        if G.LOG_DISCRETE_TIME==1:
            self.proc.log_discrete_time = self.env.process(self.__log_discrete_time_generator())
        # Initialize the update of the maintenance slots cycle
        self.proc.update_maintenance_slots_cycle = self.env.process(self.__generator_update_maintenance_slots_cycle())
        # Initialize disruption level generator at AMS
        self.proc.disruption_level_AMS = self.env.process(self.__generator_disruption_level_AMS())
        # Initialize maintenance schedule process
        self.proc.maintenance_scheduler = self.env.process(self.__generator_maintenance_scheduler())
        # Initialize tail assignment process
        self.proc.tail_assignment = self.env.process(self.__generator_tail_assignment())
        # Initialize process of tasks going due
        self.proc.tasks_missed = self.env.process(self.__generator_update_empty_slots_missed_tasks_cancelled_rotations())

        # Generate aircraft at the beginning of the simulation
        for aircraft in self.aircraft:
            # Initialize operations process for each aircraft
            aircraft.process = self.env.process(self.__generator_aircraft_process(aircraft))
            # Initialize processes for AOG, and events for AOG start and end
            aircraft.process_AOG = self.env.process(self.__generator_AOG_process(aircraft))
            #aircraft.event_AOG_arrived = self.env.event()
            #aircraft.event_AOG_started = self.env.event()
            aircraft.event_AOG_ended = self.env.event()
            aircraft.current_duty_ended = self.env.event()
            # Initialize event that triggers when a new rotation or slot is assigned to an aircraft
            aircraft.next_duty_changed = self.env.event()
            aircraft.next_duty_start_reached = self.env.event()
            # Initialize unscheduled maintenance process
            #self.env.process(self.generator_unscheduled_maintenance(aircraft))

    def __generator_update_maintenance_slots_cycle(self):
        ''' Generator that updates the current maintenance slots cycle'''
        # If first cycle, find how many days are left till the end of the cycle. Needed because cycles are initiated
        # on Monday
        if self.slots_TO!=[]:
            init_day = self.slots_TO[0].slotNorm.simulation_start_weekday
        else:
            init_day = 0
        days_left = self.slots_cycle_duration - init_day
        minutes_left = days_left * 24 * 60
        yield self.env.timeout(minutes_left)
        self.slots_cycle += 1
        while True:
            yield self.env.timeout(self.slots_cycle_duration)
            self.slots_cycle += 1


    # =================================================================================#
    # DES - DISRUPTION LEVEL
    # =================================================================================#
    def __generator_disruption_level_AMS(self):
        '''
        This generator updates the disruption level at the hub. On the basis of historical data, time is
        discretized in 20 minutes long brackets. Four arbitrarily defined (config) disruption levels are considered:
        0:norm, 1:low, 2:mid, 3:high. When the airport is experiencing a certain disruption level, a different
        distribution of aircraft delays is followed. A certain disruption level is modelled as a discrete event that lasts a certain
        number of brackets, and transition between disruption levels is modelled through a transition probability matrix
        built on the basis of historical data. Following are some details of the model:
        - At the beginning of each day, the disruption level is always zero.
        - Between midnight and six am, the disruption level is not updated, and the last observed level is
            prolonged until the next morning
        '''
        # Generator initialization, wait until 6am reached
        self._update_now()
        bracket_start_time = P.BRACKETS_TIME_START
        bracket_end_time = P.BRACKETS_TIME_START + timedelta(minutes= P.BRACKETS_N * P.BRACKETS_DURATION-1)

        datetime_start_process = self.now.replace(hour= bracket_start_time.hour,
                                                  minute= bracket_start_time.minute,
                                                  second= bracket_start_time.second)
        time_to_wait = self.__minutes_between_datetimes(self.now, datetime_start_process)
        time_to_wait = max(0, time_to_wait)
        next_disruption_state = 0

        disruption_state_id = self.__get_duty_id()
        self.__log_sim(aircraft='disruption_state', duty='disruption_state', start_or_end='start',
                       id=disruption_state_id)
        yield self.env.timeout(time_to_wait)
        self.__log_sim(aircraft='disruption_state', duty='disruption_state', start_or_end='end',
                       id=disruption_state_id)

        # Standard generator
        while True:
            self._update_now()

            # Set new disruption state to previously found next state
            self.AMS_disruption_state = next_disruption_state


            # DISRUPTION STATE DURATION
            # Sample duration of the disruption state
            disruption_state_brackets_duration = self.__get_disruption_sample(next_disruption_state, 'events_sampled')

            # Keep a copy of sampled value for validation
            validation_disruption_event_duration = disruption_state_brackets_duration
            disruption_state_duration = round(disruption_state_brackets_duration*P.BRACKETS_DURATION)
            time_state_end = self.now + timedelta(minutes=disruption_state_duration)

            # If bracket would end after end of day brackets, prolong timeout until next day at 6
            end_of_day = self.now.replace(hour=bracket_end_time.hour,
                                          minute=bracket_end_time.minute,
                                          second=bracket_end_time.second)


            if time_state_end >= end_of_day:
                # Modify disruption event duration validation
                validation_disruption_event_duration = self.__minutes_between_datetimes(self.now, end_of_day) / P.BRACKETS_DURATION
                next_disruption_state = 0
                time_state_end = (self.now + timedelta(days=1)).replace(hour=bracket_start_time.hour,
                                                                        minute=bracket_start_time.minute,
                                                                        second=bracket_start_time.second)
                # Find disruption state timeout duration
                disruption_state_duration = self.__minutes_between_datetimes(self.now, time_state_end)
            # Otherwise, find next state
            else:
                # Sample number [0,1]
                sample = random.uniform(0,1)
                transition_array = self.disruptions_hub.transition_probability_matrix[self.AMS_disruption_state, :]
                probability_sum = 0
                next_state_tested = 0
                next_state_found = 0

                while next_state_found == 0 and next_state_tested < len(transition_array):
                    probability_sum += transition_array[next_state_tested]
                    if (sample < probability_sum and next_state_tested != self.AMS_disruption_state) or (next_state_tested == len(transition_array) - 1):
                        next_disruption_state = next_state_tested
                        next_state_found = 1
                    else:
                        next_state_tested += 1

            # Save disruption event duration value for validation
            disruption_state_duration = max(0, disruption_state_duration)
            self.disruptions_hub.disruption_levels[self.AMS_disruption_state]['events_validation'].append(validation_disruption_event_duration)

            # Add to log for visualization
            disruption_state_id = self.__get_duty_id()
            self.__log_sim(aircraft='disruption_state', duty='disruption_state', start_or_end='start',
                           id=disruption_state_id)
            yield self.env.timeout(disruption_state_duration)
            self.__log_sim(aircraft='disruption_state', duty='disruption_state', start_or_end='end',
                           id=disruption_state_id)
    # =================================================================================#
    # DES - MAINTENANCE SCHEDULER
    # =================================================================================#
    def __generator_maintenance_scheduler(self):
        ''' Generator for scheduling maintenance.

        One-shot mode (G.MAINTENANCE_SCHEDULE_ONE_SHOT): the optimiser is called once at the start
        of the run and plans maintenance over the whole horizon, producing a reference plan that is
        then held fixed. Deferred defects arriving later are opened periodically and folded into
        slots at execution (see __fold_open_DDs_into_slot); disruptions are absorbed by the recovery
        module. Rolling mode: the optimiser re-runs every G.MAINTENANCE_SCHEDULE_INTERVAL days. '''
        time_to_timeout = self.__minutes_from_days(G.MAINTENANCE_SCHEDULE_INTERVAL)
        if G.MAINTENANCE_SCHEDULE_ONE_SHOT:
            # Plan the whole horizon once
            self.__update_LM_slots()
            self.__update_slots_scheduling()
            self.__update_open_DDs()
            self.scheduler_maintenance = self.schedule_maintenance()
            self.__assign_scheduled_slots_to_aircraft()
            # Thereafter keep bookkeeping current without re-planning the schedule
            while True:
                yield self.env.timeout(time_to_timeout)
                self.__update_LM_slots()
                self.__update_open_DDs()
        else:
            while True:
                self.__update_LM_slots()
                self.__update_slots_scheduling()
                self.__update_open_DDs()
                self.scheduler_maintenance = self.schedule_maintenance()
                self.__assign_scheduled_slots_to_aircraft()

                yield self.env.timeout(time_to_timeout)

    def __update_LM_slots(self):
        # Update current time
        self._update_now()
        for aircraft in self.aircraft:
            # Find executed slot for the aircraft
            LM_slot_executed = [sl for sl in aircraft.slots_LM if sl.dateStart_init < self.now]

            # Check that only one slot executed found for each aircraft
            if len(LM_slot_executed)==0 and self.scheduler_maintenance!=None:
                raise Exception('Line Maintenance slot not found')
            elif len(LM_slot_executed)>1:
                raise Exception('More than one slot executed found')

            if len(LM_slot_executed)!=0:
                # Update tasks and slot execution
                self.__update_tasks_execution(aircraft=aircraft, slot=LM_slot_executed[0], execution_type='executed_LM')


    def __update_slots_scheduling(self):
        ''' Update the maintenance scheduling window and find the corresponding open slots '''
        # Update now
        self._update_now()
        # Find end of scheduling window. In one-shot mode the window spans the whole remaining
        # horizon, so the single optimiser call plans all maintenance to the end of the run.
        window_days = G.SIM_DURATION if G.MAINTENANCE_SCHEDULE_ONE_SHOT else G.MAINTENANCE_SCHEDULE_WINDOW
        end_scheduling_window = self.now + timedelta(days=window_days) - timedelta(minutes=1)
        # Change datetime to the end of the day
        self.maintenance_scheduling_window_end = end_scheduling_window

        # Find open slots
        slots_scheduling = [sl for sl in self.slots
                            if sl.remarks!='AG'
                            and sl not in self.slots_cancelled
                            and sl.dateStart_init >= self.now
                            and sl.dateStart_init <= self.maintenance_scheduling_window_end]
        self.slots_scheduling = slots_scheduling
        self.slots_open = list(set(self.slots_open + self.slots_scheduling))

    def __update_open_DDs(self):
        ''' Open the DDs that will arrive before the maintenance scheduled is called again '''
        # All DDs arriving before the next time the maintenance scheduler is called should be open
        max_date_arrival = self.__get_now() + timedelta(days=G.MAINTENANCE_SCHEDULE_INTERVAL)

        for aircraft in self.aircraft:
            # DDs to be opened for the aircraft
            DDs_to_open = [ts for ts in aircraft.tasks_DDs_future if ts.dateArrival < max_date_arrival]
            # Add to lists of open tasks
            aircraft.tasks_open.extend(DDs_to_open)
            self.tasks_open.extend(DDs_to_open)
            # Remove open DDs from list of future DDs
            aircraft.tasks_DDs_future = [ts for ts in aircraft.tasks_DDs_future if ts not in DDs_to_open]


    def __assign_scheduled_slots_to_aircraft(self):
        ''' Assign tasks to maintenance slots and tasks to aircraft according to results from scheduler '''
        # Empty previous lists
        for aircraft in self.aircraft:
            aircraft.slots = [sl for sl in aircraft.slots if sl in self.slots_open
                              and sl not in self.slots_scheduling]
            aircraft.slots_LM = [sl for sl in aircraft.slots_LM if sl in self.slots_open
                                  and sl not in self.slots_scheduling]

        # Divide TO and LM slots
        slots_to = [sl for sl in self.slots_scheduling if sl.remarks=='TO']
        slots_lm = [sl for sl in self.slots_scheduling if sl.remarks=='LM']

        ##### TO SLOTS #####
        for slot in slots_to:
            slot.aircraft = None
            slot.tasks = []
            slot.workpackage_anticipation = None
            slot.aircraft_clean_days = None

        # Impose new assignment
        ac_slots_assignment = [(ac, sl) for (ac, sl) in self.scheduler_maintenance.model.set_aircraft_slot
                               if round(self.scheduler_maintenance.model.dv_aircraft_slot[(ac, sl)]()) == 1]

        for (aircraft, slot) in ac_slots_assignment:
            aircraft.slots.append(slot)
            slot.aircraft = aircraft
            # Find tasks assigned to the slot
            tasks = [ts for (ts, sl) in self.scheduler_maintenance.model.set_task_slot
                     if sl == slot and round(self.scheduler_maintenance.model.dv_task_slot[(ts, slot)]()) == 1]
            slot.tasks = tasks
            # Compute scheduled duration and due date of the workpackage for the slot
            slot.compute_duration('scheduled')
            slot.find_workpackage_due_date()

        # Check if duty current and next have changed
        for aircraft in self.aircraft:
            self.__check_aircraft_duty_current_and_next(aircraft)


        ##### LM SLOTS #####
        for slot in slots_lm:
            # Assign tasks to slot
            tasks = [ts for (ts, sl) in self.scheduler_maintenance.model.set_task_slot_lm
                     if sl == slot and round(self.scheduler_maintenance.model.dv_task_slot[(ts, slot)]()) == 1]
            slot.tasks = tasks

        # Mark tasks of next lm slots as in progress
        for aircraft in self.aircraft:
            slots_lm_ac = [sl for sl in slots_lm if sl.aircraft == aircraft]
            # Add slot to list of aircraft's slots (Cancelled before)
            aircraft.slots_LM.extend(slots_lm_ac)
            # Mark next slot tasks as in progress
            next_slot_lm = min(aircraft.slots_LM, key=lambda x:x.dateStart_init)
            self.tasks_in_progress = self.tasks_in_progress + next_slot_lm.tasks


        # Compute clean value for each aircraft, slot assignment
        for aircraft in self.aircraft:
            tasks_assigned = []
            ac_slots = [sl for sl in (aircraft.slots+aircraft.slots_LM) if sl.remarks!='AG']
            ac_slots = sorted(ac_slots, key=lambda x:x.dateStart_final)
            # If all assigned slots are LM slots, continue
            if set(ac_slots).issubset(set(aircraft.slots_LM)):
                continue

            for slot in ac_slots:
                tasks_assigned = tasks_assigned + slot.tasks
                if slot.remarks!='LM':
                    # Slot work package anticipation
                    if [ts for ts in slot.tasks if ts.dateDue != None] == []:
                        wp_anticipation = 1000
                    else:
                        wp_due_date = sorted([ts for ts in slot.tasks if ts.dateDue != None], key=lambda x:x.dateDue)[0].dateDue
                        wp_anticipation = self.__days_between_dates(slot.dateStart_final, wp_due_date)
                    self.workpackages_anticipation.append(wp_anticipation)
                    slot.workpackage_anticipation = wp_anticipation

                    # Clean values
                    tasks_health = [ts for ts in aircraft.tasks_open
                                    if ts.dateDue.date() >= slot.dateStart_final.date()
                                    and ts.dateArrival.date() <= slot.dateStart_final.date()
                                    and ts not in tasks_assigned]
                    if tasks_health == []:
                        clean_value = 1000
                    else:
                        tasks_health = sorted(tasks_health, key=lambda x:x.dateDue)
                        clean_value = self.__days_between_dates(slot.dateStart_final, tasks_health[0].dateDue)
                    self.clean_values.append(clean_value)
                    slot.aircraft_clean_days = clean_value


    # =================================================================================#
    # DES - TAIL ASSIGNMENT SCHEDULER
    # =================================================================================#

    def __generator_tail_assignment(self):
        ''' Generator for executing the tail assignment at fixed intervals '''
        # Find interval for tail assignment operations in minutes
        time_to_timeout = self.__minutes_from_days(G.TAIL_ASSIGNMENT_INTERVAL)
        while True:
            # Update the next time the tail assignment model will be called for scheduled calls
            self.scheduler_tail_assignment_next_call = self.__get_now() + timedelta(days=G.TAIL_ASSIGNMENT_INTERVAL)
            self.__update_rotations_open()
            self.__call_tail_assignment(reason='planning')
            yield self.env.timeout(time_to_timeout)

    def __call_tail_assignment(self, reason='planning'):
        ''' Call the tail assignment module and update rotations assignments'''
        self.scheduler_tail_assignment = self.tail_assignment()
        self.__assign_rotations_to_aircraft(reason=reason)
        # Print verification file if requested
        if 3 in G.VERIFICATION_VALIDATION:
            self.__tail_assignment_verification_update(reason=reason, model = self.scheduler_tail_assignment)

    def __update_rotations_open(self):
        self._update_now()
        rotations_open = [rt for rt in self.rotations
                          if rt not in self.rotations_cancelled
                          and rt not in self.rotations_cancelled_recovery
                          and rt not in self.rotations_executed
                          and rt.dep_sched <= self.__get_now() + timedelta(days=G.TAIL_ASSIGNMENT_WINDOW)]
        self.rotations_open = rotations_open

    def __assign_rotations_to_aircraft(self, reason):
        ''' Updates the assignment of rotations based on the results of the tail assignment'''
        # Remove previous assignment
        for aircraft in self.aircraft:
            aircraft.rotations = [rt for rt in aircraft.rotations if rt in self.rotations_open
                                  and rt not in self.scheduler_tail_assignment.model.setRotations]
        for aircraft in self.aircraft:
            aircraft.reserve_slots = [rs for rs in aircraft.reserve_slots
                                      if rs not in self.scheduler_tail_assignment.model.setReserveSlots]
        for rotation in self.scheduler_tail_assignment.model.setRotations:
            rotation.aircraft = None

        ##### ROTATIONS #####
        # Impose new assignment
        rot_ac_assignment = [(rot, ac) for (rot, ac) in self.scheduler_tail_assignment.model.setRotAc
                             if round(self.scheduler_tail_assignment.model.dvRotAc[rot, ac]())==1]
        for (rotation, aircraft) in rot_ac_assignment:
            aircraft.rotations.append(rotation)
            rotation.aircraft = aircraft

        # Cancel rotations that are scheduled to happen before the next scheduled call of the tail assignment model +
        # the fixed window
        end_cancellation_window = self.scheduler_tail_assignment_next_call + timedelta(days=G.TAIL_ASSIGNMENT_FIX)
        rotations_cancelled = [rt for rt in self.scheduler_tail_assignment.model.setRotations
                               if round(self.scheduler_tail_assignment.model.dvRotUnassign[rt]())==1
                               and rt.dep_sched <= end_cancellation_window]


        # Find reason for next function
        if reason == 'planning':
            reason_canc = 'tail_assignment'
        elif reason == 'recovery':
            reason_canc = reason
        else:
            log_error('self.__assign_rotations_to_aircraft: REASON OF ROTATION CANCELLATION NOT SUPPORTED')
            reason_canc = reason
        for rot_cancelled in rotations_cancelled :
            self.__cancel_rotation(rot_cancelled, reason=reason_canc)#'tail_assignment')

        ##### RESERVE SLOTS #####
        # Impose new assignment
        rs_ac_assignment = [(rs, ac) for (rs, ac) in self.scheduler_tail_assignment.model.setReserveAc
                            if round(self.scheduler_tail_assignment.model.dvReserveAc[rs,ac]())==1]
        for (reserve_slot, aircraft) in rs_ac_assignment:
            aircraft.reserve_slots.append(reserve_slot)

        ##### CHECK IF DUTY CURRENT AND NEXT HAVE CHANGED #####
        for aircraft in self.aircraft:
            self.__check_aircraft_duty_current_and_next(aircraft)


    def __check_aircraft_duty_current_and_next(self, aircraft):
        ''' Check if the duty current and next of an aircraft have changed'''
        # Check that current duty not changed
        if (aircraft.duty_current != None) \
                and (aircraft.duty_current not in aircraft.rotations) \
                and aircraft.duty_current not in aircraft.slots:
            print('duty_current ', aircraft.id, aircraft.duty_current.id,
                  ' does not appear in aircraft rotations or slots list')
            breakpoint()
            raise Exception('Current duty cannot be re-assigned')

        # Check if next duty has changed
        duty_next_new = self.__find_next_duty(aircraft)
        if duty_next_new != aircraft.duty_next:
            self.__next_aircraft_duty_has_changed(aircraft)


    # =================================================================================#
    # DES - AOG
    # =================================================================================#
    def __generator_AOG_process(self, aircraft):
        ''' Generator for AOG situations for an aircraft '''
        while True:
            ##### WAIT NEXT AOG ARRIVAL #####
            time_wait_AOG = self.__get_AOG_sample(aircraft, 'time_between_AOG_sampled')
            yield self.env.timeout(time_wait_AOG)

            # If aircraft is undergoing duty, wait for it to end before the AOG is disclosed
            if aircraft.duty_current!=None:
                yield aircraft.current_duty_ended

            # Find AOG duration
            AOG_duration = self.__get_AOG_sample(aircraft, 'AOG_duration_sampled')
            # Generate maintenance slot
            AOG_slot = self.__generate_AOG_slot(aircraft, AOG_duration)
            # Add slot to list of aircraft slots and to list of open slots
            aircraft.slots.append(AOG_slot)
            self.slots.append(AOG_slot)
            self.slots_open.append(AOG_slot)
            self.slots_AG.append(AOG_slot)

            # Call next duty has changed
            self.__next_aircraft_duty_has_changed(aircraft)

            # Call recovery controller
            self.__recovery_controller_AOG(slot_aog=AOG_slot)
            # Wait the end of the grounding
            yield aircraft.event_AOG_ended

    def __get_AOG_sample(self, aircraft, sample_type):
        # Get correct fleet data
        fleet_data = next(fl for fl in self.distributions_AOG if fl['fleet']==aircraft.type)
        # Get sample
        if not fleet_data[sample_type]:
            # If we run out of AOG samples, return a large time-between-failure
            # so the simulation doesn't crash and simply continues without new AOGs.
            if sample_type == 'time_between_AOG_sampled':
                return 10000  # 10,000 minutes until next AOG
            else:
                return 120  # Default 2 hours for the AOG duration itself
        sample = fleet_data[sample_type].pop(0)
        return sample

    def __generate_AOG_slot(self, aircraft, duration):
        ''' Generate an AOG slot for the input aircraft, starting when the function is called and lasting the specified
        duration [min]'''
        now = self.__get_now()
        slot_id = 'AOG_'+aircraft.id+'_'+self.__now_string()
        # Initially schedule at finding time so that executed ASAP
        date_start_init = now
        date_end_init = now+timedelta(minutes=duration)

        # Generate slot
        slot = Slot(
            id=slot_id,
            subtype=None,
            dateStart_init=date_start_init,
            dateEnd_init=date_end_init,
            remarks='AG',
            duration=timedelta(minutes=duration),
            cycle=None,
            slotNorm=None,
            location='P',
            aircraft_pre_assigned=aircraft
        )

        date_start_final, date_end_final = self.__AOG_find_expected_start_end(aircraft, slot)
        # Update expected start and end time
        slot.dateStart_final = date_start_final
        slot.dateEnd_final = date_end_final

        # Impose scheduled duration as initial duration
        slot.compute_duration('scheduled')
        slot.aircraft = aircraft
        return slot

    def __AOG_find_expected_start_end(self, aircraft, aog_slot):
        '''
        Given an aircraft and an aog slot find the expected start and end time of the aog slot.
        If the aircraft has no current nor previous duties, or if the previous duty has finished to allow its buffer
        time to pass (TAT/2 or towing time), then the slot will start right away. If this is not the case,
        the current duty of the aircraft or, if there is no current duty, the last one is taken as reference. Then,
        the start time of the AOG is estimated as the end of the reference duty + its buffer time + the towing time
        of the AOG slot. The end time is computed summing the slot duration to the estimated start time.
        '''

        now = self.__get_now()

        # Estimate actual start and end time and update time
        if isinstance(aircraft.duty_current, Rotation):
            date_start_final = aircraft.duty_current.arr_act + timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)
        elif isinstance(aircraft.duty_current, Slot):
            date_start_final = aircraft.duty_current.dateEnd_final + aircraft.duty_current.towing_time
        elif isinstance(aircraft.duty_last, Rotation) \
                and aircraft.duty_last.arr_act + timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT) > now:
            date_start_final = aircraft.duty_last.arr_act + timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)
        elif isinstance(aircraft.duty_last, Slot) \
                and aircraft.duty_last.dateEnd_final + aircraft.duty_last.towing_time > now:
            date_start_final = aircraft.duty_last.dateEnd_final + aircraft.duty_last.towing_time
        else:
            date_start_final = aog_slot.dateStart_init

        # Add towing time if present
        date_start_final = date_start_final + aog_slot.towing_time
        date_end_final = date_start_final + aog_slot.duration_init

        return date_start_final, date_end_final

    def __AOG_slot_has_ended(self, aircraft):
        ''' Triggers the event for the end of the AOG slot so that the generator of AOG can start again'''
        aircraft.event_AOG_ended.succeed()
        aircraft.event_AOG_ended = self.env.event()


    # =================================================================================#
    # DES - AIRCRAFT PROCESSES
    # =================================================================================#
    def __generator_aircraft_process(self, aircraft):
        ''' Generator for the aircraft process '''
        while True:
            ##### WAIT NEXT DUTY REACHED #####
            aircraft.wait_for_next_duty = self.__generator_wait_for_next_duty(aircraft)
            self.env.process(aircraft.wait_for_next_duty)
            yield aircraft.next_duty_start_reached
            aircraft.next_duty_start_reached = self.env.event()
            aircraft.duty_current = aircraft.duty_next
            aircraft.duty_next = self.__find_next_duty(aircraft)



            ##### ROTATION #####
            if isinstance(aircraft.duty_current, Rotation):
                rotation = aircraft.duty_current
                flights = sorted(rotation.flights, key=lambda x: x.dep_sched)
                for flight in flights:
                    # IF NOT FIRST FLIGHT, WAIT TURN AROUND TIME OR SCHEDULED DEPARTURE TIME
                    if flight!=flights[0]:
                        first_flight = False
                        time_to_flight_departure = self.__time_to_duty_start(flight)
                        TAT = self.__find_TAT_flight(flight)
                        wait_flight_departure = self.env.timeout(time_to_flight_departure)
                        wait_TAT = self.env.timeout(TAT)

                        yield(wait_flight_departure & wait_TAT)
                    else:
                        first_flight=True
                    # FLIGHT DELAY
                    flight_delay = self.__find_flight_delay(flight)

                    # Save delays per category
                    flight.delay_primary = flight_delay
                    # reactionary and technical delay
                    if self.__get_now() > flight.dep_sched:
                        delay_react = int(self.__minutes_between_datetimes(flight.dep_sched, self.__get_now()))
                        if first_flight and type(aircraft.duty_last)==Slot:
                            flight.delay_technical = delay_react
                        else:
                            flight.delay_reactionary = delay_react

                    if G.TEST_RECOVERY: # TODO For testing disruption recovery model
                        choice_add_delay = random.uniform(0,1)
                        if choice_add_delay <= 0.03:
                            choice_delay_duration = random.uniform(2,6)
                            flight_delay = flight_delay+60*choice_delay_duration

                    condition_log_delay = (self.__get_now() > flight.dep_sched) or (flight_delay>0)
                    id = self.__get_duty_id()
                    if condition_log_delay:
                        self.__log_sim(aircraft, 'delay', 'start', id=id, first_flight=first_flight,
                                       delayed_flight=flight)
                    yield self.env.timeout(flight_delay)
                    if condition_log_delay:
                        self.__log_sim(aircraft, 'delay', 'end', id=id, first_flight=first_flight)

                    # BLOCK TIME
                    # Set aircraft state
                    aircraft.state = 'flying'
                    # Set flight departure time
                    flight.dep_act = self.__get_now()

                    if flight.dep_act - flight.dep_sched > timedelta(hours=4):
                        log_warning('Long delay') # TODO delayed departure test


                    # If flight is first flight in rotation, set departure time of rotation
                    if flight == flights[0]:
                        rotation.dep_act = flight.dep_act
                        rotation.aircraft = aircraft
                        self.__log_sim(aircraft, rotation, 'start')

                    self.__recovery_controller(aircraft, flight=flight) #duty_current=rotation, duty_next=aircraft.duty_next, flight=flight)

                    # Find flight duration
                    flight_duration = flight.flight_norm.block_time
                    self.__log_sim(aircraft, flight, 'start')
                    yield self.env.timeout(flight_duration)
                    self.__log_sim(aircraft, flight, 'end')

                    # AIRCRAFT HAS ARRIVED TO DESTINATION
                    # Set aircraft state
                    aircraft.state = 'on ground'
                    # Update aircraft coordinates
                    aircraft.coordinates = flight.airport_arr.coordinates
                    # Set arrival time for the flight
                    flight.arr_act = self.__get_now()


                    # Update rotation arrival
                rotation.arr_act = self.__get_now()

                # UPDATE ROTATIONS LISTS
                self.__update_rotations_execution(aircraft, rotation)
                self.__log_sim(aircraft, rotation, 'end')
                self.__update_duty_current(aircraft)

            ##### SLOT #####
            elif isinstance(aircraft.duty_current, Slot):
                slot = aircraft.duty_current
                # Set the planned tasks as in progress
                self.tasks_in_progress = self.tasks_in_progress + slot.tasks
                # Update results dataframe of health
                self.results['health']['checked'] = self.results['health']['checked'].mask(
                    (self.results['health']['date'].dt.date==self.__get_now().date())
                    &(self.results['health']['aircraft']==aircraft.id), 1)

                # TOWING TO HANGAR
                if slot.location == 'H':
                    aircraft.state = 'towing'
                    id = self.__get_duty_id()
                    self.__log_sim(aircraft, 'towing', 'start', id=id)
                    yield self.env.timeout(self.__minutes_from_timedelta(slot.towing_time))
                    self.__log_sim(aircraft, 'towing', 'end', id=id)

                # MAINTENANCE EXECUTION
                aircraft.state = 'maintenance'
                slot.dateStart_final = self.__get_now()

                # Fold deferred defects that arrived after the reference plan was made into this
                # slot's work package, up to remaining capacity (one-shot planning only).
                if G.MAINTENANCE_SCHEDULE_ONE_SHOT:
                    self.__fold_open_DDs_into_slot(aircraft, slot)

                # Add non-routines to slot and call recovery controller. The crew is sized to the
                # planned reserve (routine + reserved NR); only realized NR above the reserve
                # extends the A-check (delay) -- this is where prediction/buffer create value.
                reserved_NR = self.__get_reserved_NR_for_slot(slot)
                realized_NR = self.__get_NR_for_slot(slot)
                slot.add_non_routines(realized_nr_hours=realized_NR, reserved_nr_hours=reserved_NR)
                self.__recovery_controller(aircraft)

                slot_departure_time = self.__get_now() #TODO remove variable, here for debugging

                # Wait for slot completion
                slot_duration_minutes = self.__minutes_from_timedelta(slot.duration_final)
                self.__log_sim(aircraft, slot, 'start')
                yield self.env.timeout(slot_duration_minutes)
                self.__log_sim(aircraft, slot, 'end')

                # Update arrival time
                if slot.dateEnd_final != self.__get_now():
                    breakpoint()
                    warning_text = 'Aircraft', aircraft.id, '|| Slot', slot.id,\
                                   '|| Start final', dt.strftime(slot.dateStart_final, '%Y-%m-%d %H:%M'), \
                                   '|| End final', dt.strftime(slot.dateStart_final, '%Y-%m-%d %H:%M'),\
                                   '|| Now', self.__now_string()
                    log_error(warning_text)
                slot.dateEnd_final = self.__get_now()
                if slot.remarks == 'A':
                    history_key = (str(slot.aircraft.type), slot.remarks)
                    history = self.nr_completed_workpackage_history.setdefault(history_key, [])
                    history.append(float(slot.realized_nr_hours))

                # Update tasks executed
                self.__update_tasks_execution(aircraft, slot)
                self.__update_duty_current(aircraft)

                # TOWING FROM HANGAR
                if slot.location == 'H':
                    aircraft.state = 'towing'
                    id = self.__get_duty_id()
                    self.__log_sim(aircraft, 'towing', 'start', id=id)
                    yield self.env.timeout(G.TOWING_HANGAR)
                    self.__log_sim(aircraft, 'towing', 'end', id=id)
                aircraft.state = 'on ground'

                # IF AOG SLOT, TRIGGER START OF AOG GENERATOR
                if slot.remarks == 'AG':
                    self.__AOG_slot_has_ended(aircraft)

            else:
                raise Exception('Next aircraft duty not supported')

            # Trigger even of current duty ending
            self.__aircraft_current_duty_has_ended(aircraft)


    def __find_expected_arrival_time_rotation(self, flight_current):
        ''' Returns the expected rotation arrival time given the flight that is currently departing '''
        flights = [fl for fl in flight_current.rotation.flights if fl.dep_sched >= flight_current.dep_sched]
        rotation_arrival_time = self.__get_now()

        # If not last flight
        if len(flights) != 1:
            for flight in flights[:-1]:
               rotation_arrival_time = rotation_arrival_time \
                                       + timedelta(minutes=flight.flight_norm.block_time)\
                                       + timedelta(minutes=flight.airport_arr.TAT_mean)
        # Add last flight
        rotation_arrival_time = rotation_arrival_time \
                                + timedelta(minutes=flights[-1].flight_norm.block_time)

        # If expected arrival earlier than scheduled arrival, return scheduled arrival.
        # Needed because this estimation does not wait for scheduled departure time of included flights
        if rotation_arrival_time < flight_current.rotation.arr_sched:
            rotation_arrival_time = flight_current.rotation.arr_sched

        return rotation_arrival_time


    def __get_NR_for_slot(self, slot):
        ''' Return total non-routine labour hours [hours] to add to a work package.

        NR is injected PER routine task (JIC card), not once per slot: each task carrying a
        requirement draws an NR occurrence at the fleet per-task probability_NR, with a magnitude
        from the fleet-average FIFO (static) or the task code's conditional quantile (predicted).
        Summing over the slot's tasks makes total NR scale with the work in the slot
        (mass ~= n_tasks * p_nr * mean_per_occurrence). The previous one-draw-per-slot gate
        under-injected by ~(tasks/slot)x and tied total NR to slot *count* rather than work, which
        distorted the rung comparison. The per-task gate is identical in both modes, so static and
        predicted stay mass-matched per occurrence and differ only in placement.

        NOTE: this assumes probability_NR is a PER-TASK rate, which holds for the real artifact
        (build_real_inputs.py: mean of jics.nr_task_1). The mock pickle's 0.3 is a per-slot value
        and would over-inject here -- rebuild the mock before reusing it. '''
        # NR are added to slots executed in the hangar, and when they are not AG slots
        if slot.remarks == 'AG' or slot.location != 'H':
            return 0
        # Get correct fleet data
        if slot.aircraft is None:
            raise Exception('Slot has no aircraft assigned in __get_NR_for_slot')
        try:
            fleet_data = next(fl for fl in self.distributions_NR if fl['fleet']==slot.aircraft.type)
        except StopIteration:
            raise Exception('NR distribution not found for fleet ' + str(slot.aircraft.type))

        predictor = getattr(self, 'nr_predictor', None)
        fleet = slot.aircraft.type
        has_fleet = predictor.has_fleet(fleet) if predictor is not None else False
        signal = getattr(G, 'NR_PREDICTED_SIGNAL', 'probability')
        scale = float(getattr(G, 'NR_VARIANCE_SCALE', 1.0))
        if predictor is not None:
            self.__update_aircraft_usage(slot.aircraft)

        # Only routine tasks (JIC cards with a requirement code) can carry NR; deferred defects
        # (requirement is None) cannot -- NR is found-in-JIC in the historical data.
        nr_tasks = [t for t in slot.tasks if getattr(t, 'requirement', None) is not None]

        total = 0.0
        for task in nr_tasks:
            code = task.requirement.code
            if predictor is not None:
                # PREDICTED. The prediction can drive the per-task NR *probability* (the gate),
                # the *magnitude*, or both -- see G.NR_PREDICTED_SIGNAL. If the artifact lacks this
                # fleet, fall back to the static distribution (never inject 0).
                if signal in ('probability', 'both') and has_fleet:
                    p_gate = predictor.slot_probability(fleet, [code])
                    if p_gate is None:
                        p_gate = fleet_data['probability_NR']
                else:
                    p_gate = fleet_data['probability_NR']
                if random.uniform(0, 1) > p_gate:
                    continue
                if signal in ('magnitude', 'both') and has_fleet:
                    # REALIZED NR: inverse-CDF draw of the code's conditional magnitude (the buffer
                    # quantile drives the PLANNED reserve instead -- see __get_reserved_NR_for_slot).
                    sample = predictor.realized_magnitude(
                        fleet, code, self.nr_realization_rng.uniform(0, 1))
                else:  # 'probability', or fleet missing -> magnitude from the (unbiased) static dist
                    sample = self.__pop_static_nr(fleet_data)
            else:
                # STATIC: fleet per-task probability gate + fleet-average sampled magnitude (FIFO).
                if random.uniform(0, 1) > fleet_data['probability_NR']:
                    continue
                sample = self.__pop_static_nr(fleet_data)

            # Workload dispersion (EXPERIMENT_PLAN sec. 2.3): scale the realized NR spread around
            # the fleet mean, mean-preserving, so higher G.NR_VARIANCE_SCALE makes per-occurrence NR
            # more volatile (harder to anticipate) without changing the average. Applied
            # symmetrically to the static and predicted realizations. NOTE: because injected NR is
            # treated as ground truth, the predicted mode incurs no forecast error, so scenario D
            # reports an upper bound on the value of prediction. Clipping to [0, MAX_DURATION_NR]
            # adds a small upward mean drift at high scale, visible in the predictor calibration report.
            if sample > 0 and scale != 1.0:
                m = self.__fleet_nr_mean(fleet_data)
                sample = round(min(max(m + (sample - m) * scale, 0.0), M.MAX_DURATION_NR), 2)

            fleet_data['labor_fitted_validation'].append(sample)
            total += sample

        return round(total, 2)

    def __get_reserved_NR_for_slot(self, slot):
        '''Planning NR reserve [hours] used to size the A-check workforce.

        Static mode represents incumbent industry practice: reserve the plain average TOTAL NR
        hours observed in the previous five completed work packages for the same fleet/check type.
        It deliberately does not normalize by package size. During cold start, before any package
        has completed, use the fleet-average expected total for the current package.

        Predicted mode instead sums p_nr times each task's conditional buffer-quantile magnitude.
        Only realized NR above the selected reserve delays the slot at execution.'''
        if slot.remarks == 'AG' or slot.location != 'H' or slot.aircraft is None:
            return 0.0
        try:
            fleet_data = next(fl for fl in self.distributions_NR if fl['fleet']==slot.aircraft.type)
        except StopIteration:
            return 0.0
        predictor = getattr(self, 'nr_predictor', None)
        fleet = slot.aircraft.type
        has_fleet = predictor.has_fleet(fleet) if predictor is not None else False
        p = float(fleet_data['probability_NR'])

        if getattr(G, 'NR_MODE', 'static') == 'static':
            history_key = (str(fleet), slot.remarks)
            history = self.nr_completed_workpackage_history.get(history_key, [])
            window = max(1, int(getattr(G, 'NR_STATIC_HISTORY_WINDOW', 5)))
            previous_totals = list(history[-window:])
            slot.nr_reserve_history = previous_totals
            if previous_totals:
                slot.nr_reserve_basis = 'rolling_total_mean'
                return round(float(np.mean(previous_totals)) *
                             getattr(G, 'NR_STATIC_RESERVE_SCALE', 1.0), 2)

            # No pre-simulation package history is bundled. Bootstrap only the first concurrent
            # wave from the incumbent fleet-average expectation; subsequent starts use completed
            # package totals and no task-count normalization.
            routine_task_count = sum(
                1 for task in slot.tasks if getattr(task, 'requirement', None) is not None)
            slot.nr_reserve_basis = 'cold_start_expected'
            slot.nr_reserve_history = []
            return round(routine_task_count * p * self.__fleet_nr_mean(fleet_data), 2)

        mean_mag = None if has_fleet else self.__fleet_nr_mean(fleet_data)
        total = 0.0
        slot.nr_reserve_basis = 'predicted_quantile'
        slot.nr_reserve_history = []
        for task in slot.tasks:
            if getattr(task, 'requirement', None) is None:
                continue
            if predictor is not None and has_fleet:
                mag = predictor.reserved_magnitude(fleet, task.requirement.code)
            else:
                mag = mean_mag
            total += p * mag
        return round(total, 2)

    def __fold_open_DDs_into_slot(self, aircraft, slot):
        ''' Fold the aircraft's open deferred defects into the work package of a slot about to be
        executed, up to the slot's remaining labour capacity. Used in one-shot maintenance planning:
        the optimiser runs only at the start of the run and cannot assign deferred defects that
        arrive later, so they are slotted in reactively here. Routine requirements stay as planned;
        only deferred defects (tasks with no recurring requirement) are folded in. '''
        if slot.remarks in ('LM', 'AG'):
            return
        now = self.__get_now()
        # Labour already committed to the slot by the reference plan
        labour_assigned = sum(ts.laborEst.total_seconds() / 3600 for ts in slot.tasks)
        labour_remaining = slot.laborMax.total_seconds() / 3600 - labour_assigned
        # Candidate deferred defects: open, this aircraft, ready, not past due, slot-compatible
        candidate_DDs = [ts for ts in aircraft.tasks_open
                         if ts.requirement is None
                         and ts not in slot.tasks
                         and ts not in self.tasks_in_progress
                         and ts.dateReady <= now
                         and ts.dateDue >= now
                         and self.__task_fits_slot(ts, slot)]
        # Most urgent (earliest due) first
        for ts in sorted(candidate_DDs, key=lambda t: t.dateDue):
            ts_labour = ts.laborEst.total_seconds() / 3600
            if ts_labour <= labour_remaining:
                slot.tasks.append(ts)
                self.tasks_in_progress.append(ts)
                labour_remaining -= ts_labour

    @staticmethod
    def __task_fits_slot(task, slot):
        ''' Slot-compatibility check for folding a deferred defect into a slot at execution:
        work location, duration and per-task labour (mirrors the scheduler's task-slot rules). '''
        if task.workType == 'H' and slot.location == 'P':
            return False
        if task.durationEst > slot.duration_init:
            return False
        if task.laborEst > slot.laborMax_per_task:
            return False
        return True

    def __fleet_nr_mean(self, fleet_data):
        ''' Mean NR labour per occurrence for the fleet (expected value of the fitted distribution),
        cached on the fleet record. Used as the fixed centre for the mean-preserving workload-
        dispersion scaling in __get_NR_for_slot (EXPERIMENT_PLAN sec. 2.3).'''
        m = fleet_data.get('nr_mean_per_occurrence')
        if m is None:
            m = self.__find_expected_value_of_distribution(fleet_data['labor_fitted'].model)
            fleet_data['nr_mean_per_occurrence'] = m
        return m

    def __update_aircraft_usage(self, aircraft):
        ''' Update a cheap per-aircraft cumulative usage proxy (FH/FC) from elapsed sim time.
        anemos has no per-tail utilization, so this is a fleet-average elapsed estimate kept for
        traceability and the future real-data phase; it does not alter the NR magnitude.'''
        elapsed_days = max((self.__get_now() - self.start).days, 0)
        ac_type2 = str(getattr(aircraft, 'type', '') or '')[:2]
        util = next((u for u in G.AIRCRAFT_UTILIZATION
                     if u['ac_type'] == ac_type2 and u['season'] == 'summer'), None)
        if util is None:
            util = G.AIRCRAFT_UTILIZATION[0]
        aircraft.cum_fh = round(elapsed_days * util['FH'], 1)
        aircraft.cum_fc = round(elapsed_days * util['FC'], 1)

    def __pop_static_nr(self, fleet_data):
        ''' Pop one static NR labour sample from the fleet FIFO, refilling from the fitted
        distribution if exhausted (guards long runs and the predicted-probability mode, where
        per-slot occurrence rates can shift FIFO consumption).'''
        if not fleet_data['labor_fitted_sampled']:
            model = fleet_data['labor_fitted'].model
            refill = self.__sample_from_distribution(model, M.SAMPLING_SIZE_PER_YEAR_PER_AC_NR)
            refill = [round(sl, 2) for sl in refill if sl > 0 and sl <= M.MAX_DURATION_NR]
            fleet_data['labor_fitted_sampled'].extend(refill if refill else [0])
        return fleet_data['labor_fitted_sampled'].pop(0)



    # =================================================================================#
    # DES - RECOVERY
    # =================================================================================#
    @staticmethod
    def __find_buffer_of_duty(duty):
        '''Return the buffer to be added before and after the duty for towing or TAT'''
        if isinstance(duty, Rotation):
            buffer = timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)
        elif isinstance(duty, Slot):
            buffer = duty.towing_time
        else:
            raise Exception('Duty type not supported')
        return buffer

    @staticmethod
    def __find_duty_arr_act(duty):
        if type(duty) == Rotation:
            return duty.arr_act
        elif type(duty) == Slot:
            return duty.dateEnd_final
        else:
            raise Exception('Duty type not supported')

    def __recovery_controller(self, aircraft, flight=None):
        '''
        Give an aircraft checks if the recovery module should be called based on the previously expected delay of the
        duty and the currently expected delay.
        The recovery module is called when all the following conditions apply:
            - there is a change in expected delay of at least G.CALL_RECOVERY_ROTATION_DELAY_ADDED
            - The expected delay of the next rotation exceeds the threshold G.CALL_RECOVERY_ROTATION_DELAY_TOTAL.
                When the aircraft needs to execute an AG slot, then the first duty after the AG slot is considered as
                next duty.

        Note that the recovery module is also called whenever an AOG is found. In this case, the calling of the
        relevant modules is done by the method __recovery_controller_AOG
        '''

        # Find current and next duty
        duty_current = aircraft.duty_current
        # If aircraft has an AOG as next duty, take later duty as next duty
        arriving_aog_slot = next((sl for sl in aircraft.slots if sl.remarks == 'AG'
                                  and sl!=aircraft.duty_current), None)
        if arriving_aog_slot!=None:
            duty_next = self.__find_next_duty(aircraft=aircraft, include_aog=False)
        else:
            duty_next = aircraft.duty_next

        ##### UPDATE EXPECTED ARRIVAL TIMES OF CURRENT DUTY AND FOLLOWING AG SLOTS #####
        if isinstance(duty_current, Rotation):
            # Previously expected arrival time
            duty_current_expected_arrival_prev = duty_current.arr_act
            # Update expected arrival time
            duty_current_expected_arrival = self.__find_expected_arrival_time_rotation(flight)
            duty_current.arr_act = duty_current_expected_arrival

        elif isinstance(duty_current, Slot):
            # Previously expected end time
            duty_current_expected_arrival_prev = duty_current.dateEnd_final
            # Update expected start and end time (considering towing)

            duty_current_expected_arrival = duty_current.dateStart_final + duty_current.duration_final
            duty_current.dateEnd_final = duty_current_expected_arrival
        else:
            raise Exception('Current duty type not supported')

        # If an aircraft must undergo an AG slot which is not the next duty, update its start and end time
        if arriving_aog_slot != None:
            aog_slot_expected_start, aog_slot_expected_end = self.__AOG_find_expected_start_end(aircraft, arriving_aog_slot)
            arriving_aog_slot.dateStart_final = aog_slot_expected_start
            arriving_aog_slot.dateEnd_final = aog_slot_expected_end

        ##### IF NO NEXT DUTY IS ASSIGNED, THEN NO RECOVERY ACTION NEEDED #####
        if duty_next == None:
            return

        ##### DELAY OF CURRENT DUTY #####
        # Find current duty scheduled and currently expected arrival
        if isinstance(duty_current, Rotation):
            duty_current_arrival_sched = duty_current.arr_sched
        elif isinstance(duty_current, Slot):
            duty_current_arrival_sched = duty_current.dateEnd_init
        else:
            raise Exception('Duty current type not supported')

        # If the expected arrival corresponds to the scheduled arrival
        # or if the expected delay has not increased significantly, no need of recovery
        condition_delay_is_zero = duty_current_expected_arrival <= duty_current_arrival_sched
        condition_delay_not_increased = duty_current_expected_arrival - duty_current_expected_arrival_prev \
                                         < timedelta(minutes=G.CALL_RECOVERY_ROTATION_EXPECTED_DELAY_CHANGE)
        if condition_delay_is_zero or condition_delay_not_increased:
            return

        ##### DELAY OF NEXY DUTY #####
        # Find next duty expected scheduled departure. If next duty is AG slot, take
        if isinstance(duty_next, Rotation):
            duty_next_dep_act = duty_next.dep_act
        elif isinstance(duty_next, Slot):
            duty_next_dep_act = duty_next.dateStart_final
        else:
            raise Exception('Duty next type not supported')

        # Find TAT between current and next duty
        TAT_duty_current = self.__find_buffer_of_duty(duty_current)
        TAT_duty_next = self.__find_buffer_of_duty(duty_next)
        TAT = TAT_duty_current + TAT_duty_next

        # If aircraft has AOG, Add time of AOG to TAT
        if arriving_aog_slot != None:
            TAT_aog = arriving_aog_slot.towing_time
            TAT = TAT + arriving_aog_slot.duration_final + 2*TAT_aog


        # Find expected delay of next duty and difference in expected delay for current rotation
        duty_next_expected_delay = (duty_current_expected_arrival + TAT) - duty_next_dep_act

        # Call the recovery module if the expected delay of next rotation is greater than a certain threshold,
        # and the expected delay has increased with respect to previous expectation
        if duty_next_expected_delay > timedelta(minutes=G.CALL_RECOVERY_ROTATION_NEXT_EXPECTED_DELAY_CHANGE):
            self.results['dynamic']['recovery_module_disr_call_count'] += 1


            if arriving_aog_slot == None:
                print_arriving_aog = None
            else:
                print_arriving_aog = arriving_aog_slot.id
            log_info('\nRECOVERY MODULE CALL RECOVERY ||', duty_current.aircraft.id,
                  '|| Is AOG:', print_arriving_aog ,'|| Duty Current:', duty_current.id,
                  'Expected arrival current: ', dt.strftime(duty_current_expected_arrival,'%Y-%m-%d %H:%M'),
                  'Expected arrival prev', dt.strftime(duty_current_expected_arrival_prev,'%Y-%m-%d %H:%M'),
                  '|| Duty Next: ', duty_next.id, 'delay',  self.__minutes_from_timedelta(duty_next_expected_delay))


            # When long disruptions occur, it might be necessary to call the tail assignment before the recovery
            # module. This situation usually happens with long AOG, hence, the name of the exeption raised,
            # but it can also happen with 'regular' operations delays
            try:
                self.__call_recovery_module()
            except LongAOGDutyIncompatibility as e:
                log_info(str(e))
                self.__call_tail_assignment(reason='recovery')
                self.__call_recovery_module()

    def __recovery_controller_AOG(self, slot_aog):
        '''
        This method is called whenever an AOG slot arrives, and it activates the necessary recovery measures. If the
        AOG slot does not disrupt the existing schedule, then no measure is taken. If this is not the case, the
        following tasks are executed:

        - Manage the OVERLAPPING MAINTENANCE SLOTS. When an AOG happens, a partial or total overlap with a
        scheduled maintenance slot can happen. In these cases, it is reasonable to suppose that if the AOG slot
        is long enough, the tasks scheduled in the scheduled slot are executed in the AOG slot. This happens when the
        scheduled duration of the slot is shorter than the AOG slot duration by the multiplicative factor
        G.SLOT_IN_AOG_DURATION_FACTOR. However, if the maintenance slot is not long enough, then the maintenance slot
        must be postponed to after the AOG. Although a postponement could be done by the recovery module itself,
        in some cases a longer delay then the one allowed by the recovery model must be imposed. For this reason,
        the maintenance slots are automatically incorporated in the AOG or postponed after it.

        - Manage the TAIL ASSIGNMENT. In the case of long AOG slots, the end of the maintenance slot can fall outside
        the recovery window. In these cases, the tail assignment must be called before the recovery module in order
        to provide long-term tail assignment feasibility. In some cases, the AOG slot does not exceed the recovery
        window, but it overlaps with a rotation that falls outside of the recovery window (the recovery window is
        determined by the arrival of a rotation rather than its departure). In these cases, the exception
        LongAOGDutyIncompatibility is raised by the recovery module, and the tail assignment is called before the
        recovery module is called again.
        '''

        aircraft = slot_aog.aircraft
        now = self.__get_now()

        ##### CHECK IF RECOVERY ACTION NEEDED #####
        # Find next duty of aircraft excluding AOG
        next_duty = self.__find_next_duty(aircraft, include_aog=False)
        if isinstance(next_duty, Rotation):
            next_duty_start = next_duty.dep_act - timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)
        elif isinstance(next_duty, Slot):
            next_duty_start = next_duty.dateStart_final - next_duty.towing_time
        elif next_duty == None:
            next_duty_start = now + timedelta(days=G.RECOVERY_WITHIN_DAYS)
        else:
            raise Exception('Next duty type not supported')
        # If the delay of next duty is estimated to be lower than maximum allowed, no recovery is needed
        next_duty_delay = slot_aog.dateEnd_final + slot_aog.towing_time - next_duty_start
        if next_duty_delay < timedelta(minutes=G.CALL_RECOVERY_ROTATION_NEXT_EXPECTED_DELAY_CHANGE):
            return

        ##### MAINTENANCE SLOTS PREPROCESSING #####
        # Update slots based on overlap with AOG slot.
        # List of slots overlapping with AOG
        slots_overlap = [sl for sl in aircraft.slots
                         if sl!=aircraft.duty_current and sl.remarks!='AG'
                         and (sl.dateStart_final-sl.towing_time) < (slot_aog.dateEnd_final+slot_aog.towing_time)]
        # Slots to include in AOG
        slots_include_aog = [sl for sl in slots_overlap
                             if (sl.duration_scheduled * G.SLOT_IN_AOG_DURATION_FACTOR) < slot_aog.duration_init]

        # Slots to postpone after AOG
        slots_postpone_aog = [sl for sl in slots_overlap if sl not in slots_include_aog]

        # Include slots in AOG
        for slot in slots_include_aog:
            # Transfer tasks to AOG
            slot_aog.tasks = slot_aog.tasks + slot.tasks
            slot.tasks = []
            slot.aircraft = None
            slot.included_in_aog = True
            self.slots_cancelled.append(slot)
            self.results['dynamic']['slots_included_in_aog'] += 1
            # Update lists
            try:
                aircraft.slots.remove(slot)
                self.slots_open.remove(slot)
            except:
                breakpoint()
                raise Exception('Slot is not present in one of the lists')

            try:
                self.slots_scheduling.remove(slot)
            except:
                log_error('Slot',slot.id, 'is not in slots_scheduling. Now: ', self.__now_string(),
                          '\nslot start (init, final): ', dt.strftime(slot.dateStart_init, '%Y-%m-%d %H:%M'),
                          dt.strftime(slot.dateStart_final, '%Y-%m-%d %H:%M'),
                          '\nslot end (init, final): ', dt.strftime(slot.dateEnd_init, '%Y-%m-%d %H:%M'),
                          dt.strftime(slot.dateEnd_final, '%Y-%m-%d %H:%M'))

            # Set cancellation cause
            slot.cancellation_reason = 'included_in_AOG'

        # Postpone slots
        for slot in slots_postpone_aog:
            postponement = (slot_aog.dateEnd_final + slot_aog.towing_time + slot.towing_time) - slot.dateStart_final
            slot.dateStart_final = slot.dateStart_final + postponement
            slot.dateEnd_final = slot.dateEnd_final + postponement
            slot.postponed_after_aog = True

        ##### RECOVERY MODULE AND TAIL ASSIGNMENT #####
        # Check if any of the postponed slot ends after the end of the recovery window
        sl_postponed_out_of_recovery_window = [sl for sl in slots_postpone_aog
                                               if sl.dateEnd_final > now + timedelta(days=G.RECOVERY_WITHIN_DAYS)]
        # If the end of the AOG happens after the end of the recovery window, call the tail assignment module first
        if slot_aog.dateEnd_final + slot_aog.towing_time > now + timedelta(days=G.RECOVERY_WITHIN_DAYS) \
                or sl_postponed_out_of_recovery_window!=[]:
            self.__call_tail_assignment(reason='recovery')

        log_info('\nRECOVERY MODUL CALL AOG ||', aircraft.id,
              '##########################################################################')

        # In the case where the AOG finishes before the end of the recovery window, but it overlaps with some
        # rotation that falls out of the recovery window, the tail assignment modul must also be called. In this
        # case, the exception LongAOGDutyIncompatibility is raised within the recovery module, and the tail
        # assignment is called.
        try:
            self.__call_recovery_module(call_reason='AOG')
        except LongAOGDutyIncompatibility as e:
            print(str(e))
            self.__call_tail_assignment(reason='recovery')
            self.__call_recovery_module(call_reason='AOG')

        # Update call count
        self.results['dynamic']['recovery_module_AOG_call_count'] += 1



    def __call_recovery_module(self, call_reason='disruption'):
        '''
        Call the recovery module to optimize the aircraft recovery and apply the defined changes.
        Changes include:
            - Changing of the assignment of a rotation or slot. In this case the process for changing the next duty
            is called.
            - Cancelling of a rotation or slot.
            - Moving a slot to free fleet space
            - Swapping to maintenance slots
            - Delaying a slot or rotation. Note that the delay is not assigned as defined by the recovery model,
            but as the minimum feasible delay. This is done because the delays considered by the recovery model are
            discrete, but a duty should start, in practise, as soon as possible.
            For example: aircraft PHXXX is executing rotation A with an expected arrival delay, which will cause the
            next rotation B to be delayed by 65 minutes. To guarantee feasibiliy the recovery model assigns the
            closest allowed delay, which is 120 minutes. In this function, the expected departure delay of the next
            rotation will still be fixed at 65 minutes.
        '''
        ##### OPTIMIZE RECOVERY #####
        recovery_model = self.recover_disruption()

        ##### APPLY RECOVERY CHANGES #####
        self.__apply_recovery_changes(recovery_model, call_reason)

        ##### CHANGE EXPECTED DEPARTURE DELAY OF DELAYED DUTIES #####
        self.__propagate_delay_from_recovery(recovery_model)



    def __apply_recovery_changes(self, recovery_model, call_reason):
        '''
        Apply the changes in the plan based on the solution of the recovery module. The changes include:
        - Changing the assignment of a rotation.
        - Cancelling a rotation or slot.
        - Moving a slot to free fleet space
        - Swapping maintenance slots
        Note that delaying a slot or rotation is a recovery option, but implemented in other function.
        '''

        if 4 in G.VERIFICATION_VALIDATION:
            try:
                self.__verification_recovery_update(line='orig', recovery=recovery_model)
            except:
                log_error('Validation recovery original went wrong')

        count_rot_assignment_change = 0
        count_rot_assignment_change_2hr = 0
        count_rot_assignment_change_4hr = 0
        count_rot_assignment_change_6hr = 0
        count_rot_close_start_assignment_change = 0
        count_rot_delayed = 0
        count_rot_cancelled = 0
        count_sl_swaps = 0
        count_sl_delayed = 0
        count_sl_ffs = 0
        count_sl_cancelled = 0

        now_sim = self.__get_now()

        ##### APPLY ROTATIONS RECOVERY CHANGES #####
        for rotation in recovery_model.model.set_rotations:
            rot_ac = [ac for (rt,ac) in recovery_model.model.dv_rotation_aircraft
                       if rt == rotation
                       and round(recovery_model.model.dv_rotation_aircraft[rt,ac]())==1]
            rot_delayed_ac = [ac for (rd,ac) in recovery_model.model.dv_rotation_delayed_aircraft
                              if rd.rotation == rotation
                              and round(recovery_model.model.dv_rotation_delayed_aircraft[rd,ac]())==1]
            rot_cancelled = [rt for rt in recovery_model.model.dv_rotation_cancelled
                             if rt == rotation
                             and round(recovery_model.model.dv_rotation_cancelled[rt]())==1]


            # ROTATION ASSIGNMENT HAS CHANGED
            if (rot_ac != [] and rot_ac[0] != rotation.aircraft):
                aircraft_new = rot_ac[0]
            elif (rot_delayed_ac != [] and rot_delayed_ac[0] != rotation.aircraft):
                aircraft_new = rot_delayed_ac[0]
            else:
                aircraft_new = None
            # If rotation assignment has changed
            if aircraft_new != None:
                # Remove from original aircraft assignment
                aircraft_orig = rotation.aircraft
                aircraft_orig.rotations.remove(rotation)
                # Add to new aircraft
                aircraft_new.rotations.append(rotation)
                # Change rotation assignment
                rotation.aircraft = aircraft_new
                # Keep track of previous assignment for validation
                rotation._val_recovery_prec_assignment = aircraft_orig
                # Update results
                self.results['dynamic']['ac_swaps'] += 1
                if rotation.dep_sched <= now_sim + timedelta(hours=2):
                    self.results['dynamic']['ac_swaps_2hr'] += 1
                    count_rot_assignment_change_2hr += 1
                if rotation.dep_sched <= now_sim + timedelta(hours=4):
                    self.results['dynamic']['ac_swaps_4hr'] += 1
                    count_rot_assignment_change_4hr += 1
                if rotation.dep_sched <= now_sim + timedelta(hours=6):
                    self.results['dynamic']['ac_swaps_6hr'] += 1
                    count_rot_assignment_change_6hr += 1

                # Update counters
                count_rot_assignment_change += 1
                if rotation.dep_sched <= now_sim + timedelta(hours=3):
                    count_rot_close_start_assignment_change += 1


            # ROTATION IS CANCELLED
            if rot_cancelled!=[]:

                # Remove from original aircraft assignment
                aircraft_orig = rotation.aircraft
                aircraft_orig.rotations.remove(rotation)
                # Keep track of old assignment
                rotation._val_recovery_prec_assignment = aircraft_orig

                # Mark rotation as cancelled
                self.__cancel_rotation(rotation, reason='recovery')
                count_rot_cancelled += 1

            # ROTATION DELAYED
            if rot_delayed_ac!=[]:
                count_rot_delayed += 1

        log_warning(count_rot_assignment_change, 'rotations were assigned to a different aircraft')
        log_warning(count_rot_close_start_assignment_change, 'rotations departing within three hours were assigned to a different aircraft ')
        log_warning(count_rot_delayed, 'rotations were delayed')
        log_warning(count_rot_cancelled, 'rotations were cancelled')


        ##### APPLY SLOTS RECOVERY CHANGES #####
        slots_moved = []
        for slot in recovery_model.model.set_slots_full:
            slot_cancelled = True if round(recovery_model.model.dv_slot_cancelled[slot]()) == 1 else False
            slot_delayed = [sd for sd in recovery_model.model.dv_slot_delayed if sd.slot == slot
                            and round(recovery_model.model.dv_slot_delayed[sd]())==1]
            slot_swap = next((ss for ss in recovery_model.model.dv_slot_swap
                              if ss[0] == slot
                              and round(recovery_model.model.dv_slot_swap[ss]()) == 1), False)
            slot_ffs = next((ffs for ffs in recovery_model.model.dv_slot_free_fleet_space
                             if ffs.slot == slot
                             and round(recovery_model.model.dv_slot_free_fleet_space[ffs]()) == 1), False)

            # SLOT IS CANCELLED
            if slot_cancelled:
                self.__cancel_slots(slot)
                warning_text  = 'Slot '+slot.id+' scheduled to start '+slot.dateStart_init.strftime('%Y%m%d_%H%M%S') + ' (final scheduled start '+\
                                slot.dateStart_final.strftime('%Y%m%d_%H%M%S') +') was cancelled.'
                log_warning(warning_text)
                count_sl_cancelled += 1

            # SLOTS ARE SWAPPED (swap must be made only once, for both ac at the same time not to lose original data)
            elif slot_swap != False and slot not in slots_moved:
                slot1 = slot_swap[0]
                slot2 = slot_swap[1]

                # Update list of moved slots
                slots_moved.append(slot1)
                slots_moved.append(slot2)

                # Assign each slot as swap to the other
                slot1.swap = slot2
                slot2.swap = slot1

                # Repeat for validation
                slot1._val_recovery_prec_assignment = slot2
                slot2._val_recovery_prec_assignemnt = slot1

                # Swap slot time window and initialize expected actual time window
                slot1.dateStart_init, slot2.dateStart_init = slot2.dateStart_init, slot1.dateStart_init
                slot1.dateEnd_init, slot2.dateEnd_init = slot2.dateEnd_init, slot1.dateEnd_init
                slot1.dateStart_final = slot1.dateStart_init
                slot2.dateStart_final = slot2.dateStart_init
                slot1.dateEnd_final = slot1.dateEnd_init
                slot2.dateEnd_final = slot2.dateEnd_init

                # Update results
                self.results['dynamic']['slot_swaps'] += 1
                count_sl_swaps += 1

            # SLOT IS MOVED TO FREE FLEET SPACE
            elif slot_ffs != False:
                # Update list of moved slots
                slots_moved.append(slot)

                # Update the property of slot to be moved to free fleet space
                slot.free_fleet_space = True
                slot._val_recovery_ffs = True

                # Update start and end time of slot and initialize expected actual time window
                slot.dateStart_init = slot_ffs.node_start.time # TODO nodes consider towing time, so it should be subtracted here
                slot.dateEnd_init = slot.dateStart_init + slot.duration_init
                slot.dateStart_final = slot.dateStart_init
                slot.dateEnd_final = slot.dateEnd_init

                # Update results
                self.results['dynamic']['slot_delayed_ffs'] += 1
                count_sl_ffs += 1

            # SLOT DELAYED - UPDATE COUNTER
            elif slot_delayed != []:
                count_sl_delayed += 1

        # Number of swaps opportunities in solution
        slot_swap_opportunities = int(len(recovery_model.model.set_slots_flex_swaps)/2)
        self.results['dynamic']['slot_swaps_opportunities'] += slot_swap_opportunities

        # Number of swap opportunities per maintenance slot
        n_slots = int(len(recovery_model.model.set_slots_TO))
        slot_assigned_swaps = [sw for sw in recovery_model.model.set_slots_flex_swaps
                               if sw[0] not in recovery_model.model.set_slots_free]
        slot_postpone_swaps = [sw for sw in slot_assigned_swaps
                               if sw[0].dateStart_init < sw[1].dateStart_init]
        slots_today = [sl for sl in recovery_model.model.set_slots_TO
                       if sl.dateStart_init.date() == now_sim.date()]
        slot_assigned_swaps_today = [sw for sw in slot_assigned_swaps if sw[0] in slots_today]
        slot_postpone_today = [sw for sw in slot_assigned_swaps_today if sw[0].dateStart_init < sw[1].dateStart_init]

        if n_slots == 0:
            slot_swap_opportunities_per_slot = -1
            slot_postponement_opportunities_per_slot = -1
        else:
            slot_swap_opportunities_per_slot = len(slot_assigned_swaps) / n_slots
            slot_postponement_opportunities_per_slot = len(slot_postpone_swaps) / n_slots

        if len(slots_today) == 0:
            slot_swap_opportunities_per_slot_today = -1
            slot_postponement_opportunities_per_slot_today = -1
        else:
            slot_swap_opportunities_per_slot_today = len(slot_assigned_swaps_today) / len(slots_today)
            slot_postponement_opportunities_per_slot_today = len(slot_postpone_today) / len(slots_today)

        ##### LOG FOR VERIFICATION #####
        if 4 in G.VERIFICATION_VALIDATION:
            try:
                self.__verification_recovery_update(line='final', recovery=recovery_model)
            except:
                log_error('Validation recovery final went wrong')

        ##### CHECK IF NEXT DUTY OF AIRCRAFT HAS CHANGED #####
        for aircraft in recovery_model.model.set_aircraft:
            # Check that current duty not changed
            if (aircraft.duty_current!=None)\
                    and(aircraft.duty_current not in aircraft.rotations)\
                    and aircraft.duty_current not in aircraft.slots:
                print('Recovery: duty_current ',aircraft.id, aircraft.duty_current.id,
                      ' does not appear in aircraft rotations or slots list')
                breakpoint()
                raise Exception('Current duty cannot be re-assigned')

            # Check if next duty has changed or if its scheduled starting date has changed (slot has been moved)
            duty_next_new = self.__find_next_duty(aircraft)
            if duty_next_new != aircraft.duty_next or aircraft.duty_next in slots_moved:
                aircraft.duty_next = duty_next_new
                self.__next_aircraft_duty_has_changed(aircraft)

        #### SAVE RESULTS RECOVERY IF REQUESTED #####
        if G.SAVE_RECOVERY_KPIS == 1:
            csv_generate_or_append(RESULTS.FILE_NAMES['recovery_kpis']+'_'+self.id,
                                   [self.id_simulation_run, self.id,
                                    recovery_model.name, self.__now_string(), call_reason,
                                    count_rot_assignment_change, count_rot_assignment_change_2hr,
                                    count_rot_assignment_change_4hr, count_rot_assignment_change_6hr,
                                    count_rot_cancelled, count_rot_delayed,
                                    count_sl_swaps, slot_swap_opportunities, slot_swap_opportunities_per_slot,
                                    slot_postponement_opportunities_per_slot, slot_swap_opportunities_per_slot_today,
                                    slot_postponement_opportunities_per_slot_today,
                                    count_sl_cancelled, count_sl_delayed, count_sl_ffs],
                                   simulation_id=self.id_simulation_run)


    def __propagate_delay_from_recovery(self, recovery_model):
        '''
        When a rotation is delayed by the recovery module, propagate the delay within the aircraft duty to
        establish minimum delay.

        The recovery module delays rotations allowing a discrete set of delays. This means that the delay imposed to
        a rotation can, in some cases, be higher that the minimum delay that would allow a feasible solution.

        For example, assume rotation KL0001 could be delayed 90 minutes to account for the propagated delay from
        previous rotations. If discrete set of delays considered by the recovery module comprises delays of 1,2,
        or 3 hours, then the model will assign a delay of 120 minutes to generate a feasible solution. In this
        function a delay of only 90 minutes is assigned to the rotation instead.

        Note that in some cases, the assignment of effective propagated delay to rotations can lead to later
        rotations in the flight line that are not delayed at all, despite being delayed in the recovery solution. In
        this case, the estimated departure time is aligned with the scheduled departure time.

         Following the previous example, assume flight KL0002 follows KL0001 in the flight line, and that it is
         delayed by 10 minutes in the recovery solution. This delay is not present when propagating the estimated
         delays within the flight line, so the estimated departure delay of flight KL0002 is assumed to coincide with
         its scheduled departure time.
        '''

        def pop_first_duty(list_rotations, list_slots):
            ''' Function that finds the first arriving duty given a list of rotation and slots. '''

            if list_slots == []:
                duty = list_rotations.pop(0)
            elif list_rotations == []:
                duty = list_slots.pop(0)
            elif list_rotations[0].dep_act < list_slots[0].dateStart_final:
                duty = list_rotations.pop(0)
            else:
                duty = list_slots.pop(0)

            return duty, list_rotations, list_slots

        # Adjust start time of duties
        for aircraft in recovery_model.model.set_aircraft:
            # List of delayed rotations and slots
            delayed_rot_ac = [rd.rotation for (rd,ac) in recovery_model.model.dv_rotation_delayed_aircraft
                              if ac == aircraft
                              and round(recovery_model.model.dv_rotation_delayed_aircraft[rd,ac]())==1]
            delayed_slots_ac = [sd.slot for sd in recovery_model.model.dv_slot_delayed
                                if sd.slot.aircraft == aircraft
                                and round(recovery_model.model.dv_slot_delayed[sd]())==1]
            # If no delayed duty found
            if delayed_rot_ac == [] and delayed_slots_ac == []:
                continue

            # Full list of rotations included in recovery
            full_rot_ac = [rt for rt in aircraft.rotations if rt in recovery_model.model.set_rotations]
            full_slots_ac = [sl for sl in aircraft.slots if sl in recovery_model.model.set_slots_full]
            full_rot_ac = sorted(full_rot_ac, key=lambda x:x.dep_act)
            full_slots_ac = sorted(full_slots_ac, key=lambda x:x.dateStart_final)

            # Check that the delayed duties are a subset of the full
            assert set(delayed_slots_ac).issubset(set(full_slots_ac)) \
                   and set(delayed_rot_ac).issubset(set(full_rot_ac))


            # First duty in chain
            aog_slot = next((sl for sl in aircraft.slots if sl.remarks == 'AG'), None)
            if aog_slot != None:
                duty_prev = aog_slot
            elif aircraft.duty_current!=None:
                duty_prev = aircraft.duty_current
            elif aircraft.duty_last!=None:
                duty_prev = aircraft.duty_last
            else:
                duty_prev, full_rot_ac, full_slots_ac = pop_first_duty(full_rot_ac, full_slots_ac)
                # If the first duty in the chain is delayed duty, remove it from list (it can happen at the beginning
                # of the simulation)
                if duty_prev in delayed_rot_ac:
                    delayed_rot_ac.remove(duty_prev)
                elif duty_prev in delayed_slots_ac:
                    delayed_slots_ac.remove(duty_prev)

            # While all delayed duties have been considered
            while delayed_slots_ac!=[] or delayed_rot_ac!=[]:
                # Find duty to evaluate as first one to be scheduled
                duty_considered, full_rot_ac, full_slots_ac = pop_first_duty(full_rot_ac, full_slots_ac)

                # If duty is delayed, find start time given previous duty
                if duty_considered in delayed_rot_ac or duty_considered in delayed_slots_ac:
                    TAT_duty_prev = self.__find_buffer_of_duty(duty_prev)
                    TAT_duty_considered = self.__find_buffer_of_duty(duty_considered)
                    duty_prev_end = self.__find_duty_arr_act(duty_prev)
                    duty_considered_dep_act = duty_prev_end + TAT_duty_prev + TAT_duty_considered

                # Apply new expected departure time only if it is later than scheduled departure time
                # Apply new departure time: Rotation
                if duty_considered in delayed_rot_ac:
                    if duty_considered_dep_act > duty_considered.dep_sched:
                        duty_considered.dep_act = duty_considered_dep_act
                        duty_considered_delay = duty_considered.dep_act - duty_considered.dep_sched
                        duty_considered.arr_act = duty_considered.arr_sched + duty_considered_delay
                    delayed_rot_ac.remove(duty_considered)
                # Apply new departure time: Slots
                elif duty_considered in delayed_slots_ac:
                    if duty_considered_dep_act > duty_considered.dateStart_init:
                        duty_considered.dateStart_final = duty_considered_dep_act
                        duty_considered_delay = duty_considered.dateStart_final - duty_considered.dateStart_init
                        duty_considered.dateEnd_final = duty_considered.dateEnd_init + duty_considered_delay
                    delayed_slots_ac.remove(duty_considered)

                # New previous duty is considered duty
                duty_prev = duty_considered





    # =================================================================================#
    # DES - WAIT FOR NEXT DUTY
    # =================================================================================#
    def __generator_wait_for_next_duty(self, aircraft):
        ''' Wait until the next duty (slot or rotation) is reached'''
        duty_start_reached = False
        # Keep track of elapsed TAT for when duty is changed
        TAT_elapsed = 0
        TAT_start = self.env.now
        while duty_start_reached == False:
            aircraft.duty_next = self.__find_next_duty(aircraft)
            # If next duty is still none, wait for next duty to arrive
            if aircraft.duty_next == None:
                yield aircraft.next_duty_changed
                aircraft.duty_next = self.__find_next_duty(aircraft)

            # Wait for the start of the next duty or until next duty is changed
            time_to_next_duty = self.__time_to_duty_start(aircraft.duty_next)
            turn_around_time = self.__find_TAT_aircraft(aircraft, TAT_elapsed)
            wait_next_duty_start = self.env.timeout(time_to_next_duty)
            wait_turn_around_time = self.env.timeout(turn_around_time)
            next_duty_change = aircraft.next_duty_changed
            yield (wait_next_duty_start & wait_turn_around_time) | next_duty_change
            # Repeat if next duty has changed
            if not next_duty_change.triggered:


                duty_start_reached = True
                aircraft.next_duty_start_reached.succeed()
            else:
                TAT_elapsed = self.env.now-TAT_start

    def __aircraft_current_duty_has_ended(self, aircraft):
        ''' Triggers the event of the aircraft finishing its current duty and create a new event to be triggered in
        the future '''
        aircraft.current_duty_ended.succeed()
        aircraft.current_duty_ended = self.env.event()

    def __next_aircraft_duty_has_changed(self, aircraft):
        ''' Triggers the event of the next duty of the aircraft being changed and creates a new event for future
        triggering'''
        aircraft.next_duty_changed.succeed()
        aircraft.next_duty_changed = self.env.event()

    @staticmethod
    def __find_next_duty(aircraft, include_aog=True):
        '''
        Return next duty of the input aircraft.
        - If an AOG slot is present in the list of slots, it is always next duty
        - Otherwise, first scheduled slot is next slot
        '''
        ##### CHECK IF AOG PRESENT #####
        # If an AOG slot is present in the list of slots, it is always next duty
        AOG_slot = next((sl for sl in aircraft.slots if sl.remarks == 'AG' and sl!=aircraft.duty_current), None)
        if AOG_slot != None and include_aog==True:
            return AOG_slot

        # Find next rotation and next slot, if any assigned
        try:
            next_rotations = [rt for rt in aircraft.rotations if rt!=aircraft.duty_current]
            next_rotation = sorted(next_rotations, key=lambda x: x.dep_act)[0]
        except:
            next_rotation = None
        try:
            next_slot = [sl for sl in aircraft.slots if sl != aircraft.duty_current]
            if include_aog==False:
                next_slot = [sl for sl in next_slot if sl.remarks!='AG']
            next_slot = sorted(next_slot, key=lambda x: x.dateStart_final)[0]
        except:
            next_slot = None
        # If no slot nor rotation, return none
        if next_rotation == None and next_slot == None:
            next_duty = None
        # If next rotation or slot not available, return the other
        elif next_rotation == None:
            next_duty = next_slot
        elif next_slot == None:
            next_duty = next_rotation
        # If both available, return earliest
        elif next_slot.dateStart_final < next_rotation.dep_sched:
            next_duty = next_slot
        else:
            next_duty = next_rotation

        return next_duty

    def __time_to_duty_start(self, duty):
        ''' Given a duty (Slot/Rotation/Flight)returns the time from its start time'''
        if isinstance(duty, Slot):
            start_time = duty.dateStart_init - duty.towing_time
        elif isinstance(duty, Rotation) or isinstance(duty, Flight):
            start_time = duty.dep_sched
        else:
            raise Exception('Duty type not found')
        time_to_duty = self.__time_from_now(start_time)
        if time_to_duty < 0:
            time_to_duty = 0
        return time_to_duty

    def __find_TAT_aircraft(self, aircraft, TAT_elapsed):
        '''
        Given an aircraft, find the TAT between the last and next duty, according to the following rules:
        - Last duty is Rotation, next is Rotation: sampled TAT at hub
        - Last duty is Rotation, next is Slot or vice-versa: sampled TAT at hub divided by two
        - Last duty is Slot, next is Slot: zero
        If the next duty has changed and some TAT time has already elapsed, subtract it from the portion of TAT
        imputable to the last duty.
        '''
        duty1 = aircraft.duty_last
        duty2 = aircraft.duty_next
        TAT_reduction = 0     # Initialise elapsed TAT that can be reduced from TAT

        def get_safe_hub_tat():
            if len(self.hub.TAT_sampled) > 0:
                return self.hub.TAT_sampled.pop(0)
            else:
                # log_warning('Hub TAT samples exhausted, using fallback')
                return 90
        if duty1 == None and duty2!=None:
            TAT = 0
        elif isinstance(duty1, Rotation) and isinstance(duty2, Rotation):
            TAT = get_safe_hub_tat()
            TAT_reduction = min(TAT_elapsed, TAT/2)
        elif isinstance(duty1, Rotation) and isinstance(duty2, Slot):
            TAT = get_safe_hub_tat()/2
            TAT_reduction = min(TAT_elapsed, TAT)
        elif isinstance(duty1, Slot) and isinstance(duty2, Rotation):
            TAT = get_safe_hub_tat()/2
        elif isinstance(duty1, Slot) and isinstance(duty2, Slot):
            TAT = 0
            log_warning('Two maintenance slots were consequently allocated to an aircraft')
        else:
            raise Exception('Case not supported')

        # Subtract the already elapsed TAT
        TAT = TAT - TAT_reduction
        TAT = round(TAT)

        # if TAT<=0, make it ==1 to allow the AG generator to possibly activate an arrived AG slot
        if TAT<=0:
            TAT = 1

        return TAT

    # =================================================================================#
    # DES - ROTATION
    # =================================================================================#
    def __find_TAT_flight(self, flight):
        ''' Given a flight, return the turn around time before the flight'''
        if not flight.airport_dep.TAT_sampled:
            # LOG A WARNING: "Airport {flight.airport_dep.id} ran out of TAT samples!"
            # Return a fallback value (e.g., 60 minutes) so the simulation can continue
            return 60

        return flight.airport_dep.TAT_sampled.pop(0)

    def __find_flight_delay(self, flight):
        '''
        Return the sampled departure delay of a flight.
        - Flights departing from hub: sample probability of having a delay and delay durations from different
                                        distributions based on current disruption level of AMS
        - Flights departing from outstations: one probability of delay and delay duration distribution
        '''

        ##### HUB #####
        if flight.airport_dep == self.hub:
            # Find disruption level
            disr_level = next(dl for dl in self.disruptions_hub.disruption_levels
                              if dl['levelId'] == self.AMS_disruption_state)
            # DETERMINE IF FLIGHT WILL EXPERIENCE A DELAY
            # Sample random number between 0 and 1
            sample = random.uniform(0, 1)
            # Use the probability of not having delay to determine if the flight will experience
            # delay for a reason other that technical and from delay propagation
            if sample <= disr_level['probability_no_delay']:
                delay = 0
            else:
                # If flight experienced delay, sample the delay
                delay = self.__get_disruption_sample(self.AMS_disruption_state, 'delays_sampled')
            # Save delay for validation
            disr_level['delays_validation'].append(delay)


        ##### OUTSTATIONS #####
        else:
            # DETERMINE IF FLIGHT WILL EXPERIENCE A DELAY
            # Sample random number between 0 and 1
            sample = random.uniform(0, 1)
            # Use the probability of not having delay to determine if the flight will experience
            # delay for a reason other that technical and from delay propagation
            if sample <= self.disruptions_outstations['probability_no_delay']:
                delay = 0
            else:
                # If flight experienced delay, sample the delay
                delay = self.disruptions_outstations['delays_sampled'].pop(0)
            # Save delay for validation
            self.disruptions_outstations['delays_validation'].append(delay)

        return delay


    # =================================================================================#
    # DES: LISTS UPDATE
    # =================================================================================#
    def __update_rotations_execution(self, aircraft, rotation):
        ''' Update lists of executed rotations'''
        # Add rotation to list of executed rotations
        self.rotations_executed.append(rotation)
        aircraft.rotations_executed.append(rotation)

        # Remove executed rotation from lists of open rotations
        try: # Rotation might not be open if fixed assignment within window
            self.rotations_open.remove(rotation)
        except:
            pass
        # TODO following try-except is done for checking raised exception
        try:
            aircraft.rotations.remove(rotation)
        except:
            breakpoint()
            raise Exception('Rotation ' +rotation.id+ ' is not present in aircraft rotations list')
        # Update results
        self.results['dynamic']['rotations_executed'] += 1

    def __update_tasks_execution(self, aircraft, slot, execution_type='executed'):
        ''' Function called when a slot ends. Updates the list of open tasks, and generate new instance for
        requirements'''
        # Add tasks to list of executed tasks
        self.tasks_executed = self.tasks_executed + slot.tasks
        aircraft.tasks_executed = aircraft.tasks_executed + slot.tasks

        # Remove executed tasks from open tasks and list of tasks in progress
        self.tasks_open = [ts for ts in self.tasks_open if ts not in slot.tasks]
        self.tasks_in_progress = [ts for ts in self.tasks_in_progress if ts not in slot.tasks]
        aircraft.tasks_open = [ts for ts in aircraft.tasks_open if ts not in slot.tasks]

        # Add new requirements instances to open tasks
        today = self.__get_now()
        # If slot is line maitnenance slot, find start time
        if slot.remarks == 'LM':
            LM_start = slot.dateStart_init
        else:
            LM_start = None
        requirements_executed = [ts for ts in slot.tasks if ts.requirement!=None]
        for req in requirements_executed:
            req.dateExecution = today
            if not G.MAINTENANCE_SCHEDULE_ONE_SHOT:
                req_generated = req.requirement.generate_instance(
                    execution_type=execution_type,
                    aircraft=aircraft,
                    today=today,
                    task=req,
                    LM_start=LM_start
                )
                self.tasks.append(req_generated)
                self.tasks_open.append(req_generated)
                aircraft.tasks_open.append(req_generated)

            # Update results
            self.results['dynamic']['tasks_executed'] += 1

        # Update execution date of DDs
        dds_executed = [ts for ts in slot.tasks if ts.requirement==None]
        for dd in dds_executed:
            dd.dateExecution = today

        # Move slot to executed slot
        self.slots_open.remove(slot)
        self.slots_executed.append(slot)
        if slot.remarks == 'LM':
            aircraft.slots_LM.remove(slot)
        else:
            aircraft.slots.remove(slot)
        aircraft.slots_executed.append(slot)
        # Update results
        if slot.remarks != 'AG':
            self.results['dynamic']['slots_executed'] += 1

    def __update_duty_current(self, aircraft):
        ''' Sets the aircraft's last duty as the current duty, and the current duty to None '''
        aircraft.duty_last = aircraft.duty_current
        aircraft.duty_current = None

    def __cancel_rotation(self, rotation, reason):
        # Update lists
        self.rotations_cancelled.append(rotation)
        self.rotations_open.remove(rotation)

        if reason == 'recovery':
            self.rotations_cancelled_recovery.append(rotation)

        # Add cancelled rotation to log
        self.__log_sim('cancelled', rotation, 'start', cancellation_reason=reason)
        self.__log_sim('cancelled', rotation, 'end', cancellation_reason=reason)
        for flight in rotation.flights:
            self.__log_sim('cancelled', flight, 'start', cancellation_reason=reason)
            self.__log_sim('cancelled', flight, 'end', cancellation_reason=reason)

    def __cancel_slots(self, slot):
        self.slots_cancelled.append(slot)
        self.slots_open = [sl for sl in self.slots_open if sl!=slot]
        self.slots_scheduling = [sl for sl in self.slots_scheduling if sl!=slot]
        slot.aircraft.slots.remove(slot)


    def __generator_update_empty_slots_missed_tasks_cancelled_rotations(self):
        ''' Every day update missed tasks and empty passed slots'''

        def rot_is_current(rotation):
            ''' Return True if rotation is current duty, False otherwise'''
            if pd.isnull(rotation.aircraft):
                return False
            elif rotation == rotation.aircraft.duty_current:
                return True
            else:
                return False


        while True: # Update list once per day
            yield self.env.timeout(self.__minutes_from_days(1))
            now_sim = self.__get_now()

            # CANCELLED ROTATIONS
            # NOTE: Rotations should be cancelled before this phase. This code here to check
            # Find cancelled rotations, leave slack for delays
            rotations_cancelled = [rot for rot in self.rotations if rot.dep_sched < now_sim - timedelta(days=1)
                                   and rot.aircraft == None and rot not in self.rotations_cancelled
                                   and rot_is_current(rot) == False]
            for rotation in rotations_cancelled:
                self.__cancel_rotation(rotation, reason='missed')
                log_error('A rotation was missed but not cancelled')

            # PASSED RESERVE SLOTS
            for aircraft in self.aircraft:
                aircraft.reserve_slots = [rs for rs in aircraft.reserve_slots
                                          if rs.dep_sched.date() >= now_sim.date()]

            # MISSED TASKS
            # Find missed tasks that are not assigned to slot currently in progress
            tasks_missed = [ts for ts in self.tasks_open if ts.dateDue < now_sim and ts not in self.tasks_in_progress]
            # If any missed task is found, update lists
            if tasks_missed != []:
                # Move to missed tasks list
                self.tasks_missed = self.tasks_missed + tasks_missed
                self.tasks_open = [ts for ts in self.tasks_open if ts not in tasks_missed]
                # Update results
                self.results['dynamic']['tasks_missed'] += len(tasks_missed)
                for task in tasks_missed:
                    task.aircraft.tasks_open.remove(task)
                    task.aircraft.tasks_missed.append(task)

                # Generate new instance for missed requirements
                if not G.MAINTENANCE_SCHEDULE_ONE_SHOT:
                    requirements_missed = [ts for ts in tasks_missed if ts.requirement != None]
                    for req in requirements_missed:
                        req_generated = req.requirement.generate_instance(
                            execution_type='missed',
                            aircraft=req.aircraft,
                            today=now_sim,
                            task=req
                        )
                        req.aircraft.tasks_open.append(req_generated)
                        self.tasks_open.append(req_generated)
                        self.tasks.append(req_generated)


            # UNUSED MAINTENANCE SLOTS
            unused_slots = [sl for sl in self.slots_open if sl.dateStart_final<now_sim and sl.aircraft==None and
                            sl.remarks!='AG']
            self.slots_open = [sl for sl in self.slots_open if sl not in unused_slots]
            self.slots_unused = self.slots_unused + unused_slots

            # HEALTH RESULTS
            for aircraft in self.aircraft:
                tasks_health = [ts for ts in aircraft.tasks_open
                                if ts not in self.tasks_in_progress
                                and ts.type!='NSRE']
                if tasks_health!=[]:
                    ac_tasks = sorted(tasks_health, key=lambda x:x.dateDue)
                    earliest_due_date = ac_tasks[0].dateDue
                    ac_health = self.__days_between_dates(now_sim, earliest_due_date)
                    if ac_health > RESULTS.HEALTH_MAX:
                        ac_health = RESULTS.HEALTH_MAX
                    elif ac_health < 0:
                        ac_health = 0
                else:
                    ac_health = RESULTS.HEALTH_MAX
                # Save health in df
                df_health = self.results['health']
                df_health['health'] = df_health['health'].mask((df_health['date'].dt.date==now_sim.date())
                                                               &(df_health['aircraft']==aircraft.id), ac_health)


    # =================================================================================#
    # FINAL RESULTS UPDATE
    # =================================================================================#
    def __generate_final_results(self):
        ''' Generate and save dataframes of results'''
        # Generate dataframes
        results_rotations = self.__generate_results_rotations_and_flights(rotations_or_flights='rotations')
        results_flights = self.__generate_results_rotations_and_flights(rotations_or_flights='flights')
        results_slots = self.__generate_results_slots()
        results_tasks = self.__generate_results_tasks()

        # Add to dictionary of simulation results
        self.results['rotations'] = results_rotations
        self.results['flights'] = results_flights
        self.results['slots'] = results_slots
        self.results['tasks'] = results_tasks


        # Generate dictionary and csv of results overview
        results_overview, results_overview_df = self.__generate_results_overview()
        self.results['overview'] = results_overview

        # Save all dataframes as csv files
        write_csv_from_dataframe(RESULTS.FILE_NAMES['rotations']+'_'+self.id, results_rotations,self.id_simulation_run)
        write_csv_from_dataframe(RESULTS.FILE_NAMES['flights']+'_'+self.id, results_flights, self.id_simulation_run)
        write_csv_from_dataframe(RESULTS.FILE_NAMES['slots']+'_'+self.id, results_slots, self.id_simulation_run)
        write_csv_from_dataframe(RESULTS.FILE_NAMES['tasks']+'_'+self.id, results_tasks, self.id_simulation_run)
        write_csv_from_dataframe(RESULTS.FILE_NAMES['overview']+'_'+self.id, results_overview_df, self.id_simulation_run)
        write_csv_from_dataframe(RESULTS.FILE_NAMES['health']+'_'+self.id, self.results['health'],self.id_simulation_run)

        # PAPER_DESIGN sec. 4.4: emit the NR mass-preservation / coverage report when predicting
        if getattr(self, 'nr_predictor', None) is not None:
            nr_report = self.nr_predictor.calibration_report()
            if not nr_report.empty:
                nr_report['sim_id'] = self.id
                write_csv_from_dataframe('results_nr_calibration_'+self.id, nr_report, self.id_simulation_run)


    def __generate_results_rotations_and_flights(self, rotations_or_flights):
        '''
        Generate dataframe of final results of rotations or flights
        :param rotations_or_flights: 'rotations' for generating rotations df, 'flights' for flights df
        '''
        # Initialize list used to generate dataframe
        data_items = []
        for rotation in self.rotations:
            # Rotation state
            if rotation in self.rotations_executed:
                rotation_state = 'executed'
            elif rotation in self.rotations_cancelled and rotation not in self.rotations_cancelled_recovery:
                rotation_state = 'cancelled_tail_assignment'
            elif rotation in self.rotations_cancelled_recovery:
                rotation_state = 'cancelled_recovery'
            else:
                rotation_state = 'future_rotation'

            # Aircraft
            if rotation_state == 'executed':
                aircraft_id = rotation.aircraft.id
            else:
                aircraft_id = None

            # List with rotation or flights included in rotation
            if rotations_or_flights=='rotations':
                items = [rotation]
                item_type = 'rotation'
            elif rotations_or_flights == 'flights':
                items = [fl for fl in rotation.flights]
                item_type = 'flight'
            else:
                raise Exception("Only 'rotations' or 'flights' accepted")

            for item in items:
                # Departure and arrival delay
                delay_departure = self.__find_delay_of_item(item, 'dep')
                delay_arrival = self.__find_delay_of_item(item, 'arr')


                item_dict = {'simulation_id': self.id, 'simulation_run_id': self.id_simulation_run,
                             'iteration': self.iteration, 'item_type': item_type , 'aircraft': aircraft_id,
                             'id': item.id, 'id_rotation': rotation.id,
                             'id_rotation_norm': rotation.rotation_norm.id_general,
                             'execution_state':rotation_state, 'departure_sched':item.dep_sched,
                             'departure_act': item.dep_act, 'arrival_sched': item.arr_sched,
                             'arrival_act':item.arr_act, 'delay_departure': delay_departure,
                             'delay_arrival': delay_arrival}

                # If item is flight add origin and destination
                if rotations_or_flights == 'flights':
                    # Flight info
                    item_dict['id_flight_norm'] = item.flight_norm.id_general
                    item_dict['leg_number'] = item.flight_norm.leg_number
                    # Origin and destination airports
                    item_dict['airport_dep'] = item.flight_norm.airport_dep.id
                    item_dict['airport_arr'] = item.flight_norm.airport_arr.id
                    # Block time
                    item_dict['block_time_act'] = item.flight_norm.block_time
                    block_time_scheduled = self.__minutes_between_datetimes(item.dep_sched, item.arr_sched)
                    item_dict['block_time_sched'] = block_time_scheduled
                    # Delays per category
                    item_dict['delay_primary'] = item.delay_primary
                    item_dict['delay_reactionary'] = item.delay_reactionary
                    item_dict['delay_technical'] = item.delay_technical

                data_items.append(item_dict)

        df_items = pd.DataFrame(data_items)

        # Add OTP column in flights dataframe
        if rotations_or_flights == 'flights':
            df_items['OTP15'] = 0
            df_items['OTP15'] = df_items['OTP15'].mask(df_items['delay_arrival'] <= 15, 1)

        return df_items


    def __generate_results_slots(self):
        data_slots = []
        for slot in self.slots:
            # Slot state
            if slot in self.slots_executed:
                slot_state = 'executed'
            elif slot in self.slots_unused:
                slot_state = 'unused'
            elif slot in self.slots_cancelled:
                slot_state = 'cancelled'
            else:
                slot_state = 'future_slot'

            # Aircraft
            aircraft = slot.aircraft
            if aircraft == None:
                aircraft_id = None
                aircraft_type = None
            else:
                aircraft_id = aircraft.id
                aircraft_type = aircraft.subtype.IATA

            # Departure and arrival delay
            delay_departure = self.__find_delay_of_item(slot, 'dep')
            delay_arrival = self.__find_delay_of_item(slot, 'arr')

            # Non-routine time
            nr_task = next((ts for ts in slot.tasks if ts.type == 'NON-ROUTINE'), None)
            if nr_task == None:
                nr_labor = 0
            else:
                nr_labor = nr_task.laborEst.total_seconds()/3600

            # Labor hours and duration max
            work_labor_max = slot.laborMax.total_seconds()/3600
            if slot.scheduled_work_labor != None:
                work_sched_labor = slot.scheduled_work_labor.total_seconds()/3600
                work_sched_duration = slot.scheduled_work_duration.total_seconds()/3600

                slot_filling = work_sched_labor/work_labor_max
            else:
                work_sched_labor = None
                work_sched_duration = None

                slot_filling = None

            if pd.isnull(slot.swap):
                slot_swap = None
            else:
                slot_swap = slot.swap.id

            # Duration scheduled and actual
            duration_scheduled = self.__minutes_between_datetimes(slot.dateStart_init, slot.dateEnd_init)
            duration_act = self.__minutes_between_datetimes(slot.dateStart_final, slot.dateEnd_final)

            slot_dict = {'simulation_id': self.id, 'simulation_run_id': self.id_simulation_run,
                         'iteration': self.iteration, 'item_type': 'slot', 'id': slot.id,
                         'aircraft': aircraft_id, 'aircraft_type': aircraft_type,
                         'slot_type': slot.remarks, 'execution_state': slot_state,
                         'departure_sched':slot.dateStart_init, 'departure_act': slot.dateStart_final,
                         'arrival_sched': slot.dateEnd_init, 'arrival_act': slot.dateEnd_final,
                         'departure_sched_original': slot.dateStart_original,
                         'arrival_sched_original': slot.dateEnd_original,
                         'delay_departure': delay_departure, 'delay_arrival': delay_arrival,
                         'work_location': slot.location,
                         'work_sched_labor': work_sched_labor,
                         'work_sched_max_duration': work_sched_duration,
                         'work_labor_max': work_labor_max,
                         'slot_filling': slot_filling,
                         'work_non_routine_labor': nr_labor,
                         'nr_reserved_labor': float(getattr(slot, 'reserved_nr_hours', 0.0)),
                         'nr_realized_labor': float(getattr(slot, 'realized_nr_hours', nr_labor)),
                         'nr_uncovered_labor': max(
                             0.0,
                             float(getattr(slot, 'realized_nr_hours', nr_labor))
                             - float(getattr(slot, 'reserved_nr_hours', 0.0))
                         ),
                         'nr_reserve_basis': getattr(slot, 'nr_reserve_basis', None),
                         'nr_reserve_history_count': len(
                             getattr(slot, 'nr_reserve_history', [])),
                         'nr_reserve_history_values': '|'.join(
                             str(round(float(value), 4))
                             for value in getattr(slot, 'nr_reserve_history', [])),
                         'nr_mode': getattr(G, 'NR_MODE', 'static'),
                         'aircraft_cum_fh': getattr(aircraft, 'cum_fh', None) if aircraft is not None else None,
                         'aircraft_cum_fc': getattr(aircraft, 'cum_fc', None) if aircraft is not None else None,
                         'duration_sched': duration_scheduled, 'duration_act': duration_act,
                         'free_fleet_space': slot.free_fleet_space, 'slot_swapped': slot_swap,
                         'cancellation_reason': slot.cancellation_reason,
                         'wp_anticipation': slot.workpackage_anticipation,
                         'clean_days': slot.aircraft_clean_days}
            data_slots.append(slot_dict)

        df_slots = pd.DataFrame(data_slots)
        return df_slots


    def __generate_results_tasks(self):

        data_tasks = []
        for task in self.tasks:
            ##### Task state #####
            if task in self.tasks_executed:
                task_state = 'executed'
            elif task in self.tasks_missed:
                task_state = 'missed'
            else:
                task_state = 'future task'

            ##### Tast execution #####
            if task_state == 'executed':
                # Slot where task executed
                task_slot = next((sl for sl in self.slots_executed if task in sl.tasks), None)
                # Check that slot found
                if task_slot == None:
                    # breakpoint()
                    log_error('SLOT IN WHICH TASK EXECUTED NOT FOUND')
                    task_slot_id = None
                    task_slot_type = None
                else:
                    task_slot_id = task_slot.id
                    task_slot_type = task_slot.remarks
                    # date_execution = task_slot.dateEnd_final
                # Days by which task execution is anticipated with respect to its due date
                date_execution = task.dateExecution
                task_anticipation = self.__days_between_dates(date_execution, task.dateDue)
                # Days between arrival and due date
                task_days_for_execution = self.__days_between_dates(task.dateArrival, task.dateDue)
                task_anticipation_relative = task_anticipation/task_days_for_execution

            else:
                task_slot_id = None
                task_slot_type = None
                date_execution = None
                task_anticipation = None
                task_anticipation_relative = None

            ##### Requirement info #####
            if task.requirement != None:
                task_interval_min = task.requirement.intervalMin.days
                task_interval_CD = task.requirement.intervals['CD']
                task_interval_FH = task.requirement.intervals['FH']
                task_interval_FC = task.requirement.intervals['FC']
                task_requirement = task.requirement.code
                task_requirement_subtype = task.requirement.ac_type
                task_requirement_class = task.requirement.req_class
            else:
                task_interval_min = (task.dateDue.date() - task.dateArrival.date()).days
                task_interval_CD = None
                task_interval_FH = None
                task_interval_FC = None
                task_requirement = None
                task_requirement_subtype = None
                task_requirement_class = None

            try:
                aircraft_id = task.aircraft.id
            except:
                aircraft_id = None

            task_dict = {'simulation_id': self.id, 'simulation_run_id': self.id_simulation_run,
                         'iteration': self.iteration, 'item_type': 'task' , 'id': task.id, 'task_type': task.type,
                         'aircraft': aircraft_id, 'info':task.info,
                         'execution_state': task_state, 'execution_slot': task_slot_id,
                         'execution_slot_type': task_slot_type,
                         'date_arrival': task.dateArrival, 'date_due': task.dateDue, 'date_ready': task.dateReady,
                         'date_execution': date_execution, 'anticipation_days': task_anticipation,
                         'anticipation_relative': task_anticipation_relative,
                         'work_labor': task.laborEst.total_seconds()/3600,
                         'work_duration': task.durationEst.total_seconds()/3600, 'work_type': task.workType,
                         'requirement': task_requirement, 'requirement_ac_type': task_requirement_subtype,
                         'interval_min': task_interval_min, 'interval_CD':task_interval_CD,
                         'interval_FC': task_interval_FC, 'interval_FH':task_interval_FH,
                         'requirement_class': task_requirement_class
                         }
            data_tasks.append(task_dict)

        df_tasks = pd.DataFrame(data_tasks)
        return df_tasks



    def __find_delay_of_item(self, item, arr_dep):
        '''
        Returns the departure or arrival delay of an item instance of classes Flight, Rotation, or Slot.
        :param arr_dep: 'dep' for departure delay and 'arr' for arrival delay
        '''
        if (isinstance(item, Rotation) or isinstance(item, Flight)) and arr_dep=='dep':
            delay = self.__minutes_between_datetimes(item.dep_sched, item.dep_act)
        elif (isinstance(item, Rotation) or isinstance(item, Flight)) and arr_dep=='arr':
            delay = self.__minutes_between_datetimes(item.arr_sched, item.arr_act)
        elif isinstance(item, Slot) and arr_dep=='dep':
            delay = self.__minutes_between_datetimes(item.dateStart_init, item.dateStart_final)
        elif isinstance(item, Slot) and arr_dep=='arr':
            delay = self.__minutes_between_datetimes(item.dateEnd_init, item.dateEnd_final)
        else:
            raise Exception('Case not supported')
        if delay < 0:
            delay = 0
        delay = round(delay)
        return delay

    def __generate_results_overview(self):
        # Dictionary of results gathered dynamically during simulation
        dynamic_results = self.results['dynamic']

        # Rotations
        rot_executed = len(self.rotations_executed)
        rot_cancelled = len(self.rotations_cancelled)
        rot_cancelled_recovery = len(self.rotations_cancelled_recovery)

        # Requirements anticipation
        df_tasks = self.results['tasks']
        df_requirements_executed = df_tasks[(df_tasks['task_type']=='REQUIREMENT') &
                                            (df_tasks['execution_state']=='executed')]
        # TODO note that missed requirements not included
        requirements_anticipation_mean = df_requirements_executed['anticipation_relative'].mean()

        # Task anticipation target in simulation
        if G.LI_HEALTH_ORIENTED == 0:
            requirements_anticipation_target = 0
        elif G.LI_HEALTH_ORIENTED == 1:
            requirements_anticipation_target = G.LI_PREFERRED_ANTICIPATION
        else:
            raise Exception('Value of parameter G.LI_HEALTH_ORIENTED not supported')

        ##### KPIs #####
        # Completion factor
        completion_factor = rot_executed/(rot_executed+rot_cancelled)
        completion_factor_recovery = rot_executed/(rot_executed+rot_cancelled_recovery)

        tasks_executed = len(self.tasks_executed)
        tasks_missed = len(self.tasks_missed)
        tasks_total = tasks_executed + tasks_missed
        if tasks_total > 0:
            tasks_execution_factor = tasks_executed / tasks_total
        else:
            # If there were no tasks to perform, the execution factor is technically 100% (or 0)
            tasks_execution_factor = 1.0

        # Delays
        df_rotations = self.results['rotations']
        rotations_executed = df_rotations[df_rotations['execution_state']=='executed']
        rotations_delay_dep = rotations_executed['delay_departure'].mean()
        rotations_delay_arr = rotations_executed['delay_arrival'].mean()

        df_flights = self.results['flights']
        flights_executed = df_flights[df_flights['execution_state']=='executed']
        flights_delay_dep = flights_executed['delay_departure'].mean()
        flights_delay_arr = flights_executed['delay_arrival'].mean()

        df_slots = self.results['slots']
        slots_executed = df_slots[(df_slots['execution_state'] == 'executed') & (df_slots['slot_type']!='AG')]
        slots_delay_dep = slots_executed['delay_departure'].mean()
        slots_delay_arr = slots_executed['delay_arrival'].mean()

        ##### Maintenance-efficiency KPIs (logged for every rung so Fig 3 can be drawn) #####
        # Interval spillage: operations-side analog of Paper 2's avg_spillage_percentage_per_task.
        # Mean absolute deviation of executed requirements from their due date, as a percentage of
        # each task's execution window. Lower = tasks done closer to due date. anticipation_relative
        # is signed (early positive, late negative); abs() captures both wasted interval and lateness.
        _anticipation = df_requirements_executed['anticipation_relative'].dropna()
        interval_spillage = float(_anticipation.abs().mean() * 100) if len(_anticipation) else float('nan')

        # Ground time: total actual maintenance-slot hours (aircraft downtime), excluding AOG slots.
        ground_time_hours = float(slots_executed['duration_act'].sum() / 60) if not slots_executed.empty else 0.0

        # Non-routine overrun (workforce model): NR-induced A-check delay in clock-hours, i.e. the
        # time the slot ran beyond its 24h plan because realized NR exceeded the reserved workforce
        # allowance -- sum of max(0, duration_final - duration_init) over executed A-check slots.
        # Zero when the buffer/prediction covered the realized NR; grows when it did not.
        nr_overrun_hours = float(sum(
            max(0.0, (sl.duration_final - sl.duration_init).total_seconds() / 3600)
            for sl in self.slots_executed
            if sl.remarks == 'A' and sl.duration_final is not None and sl.duration_init is not None))
        nr_slots = [sl for sl in self.slots_executed if sl.remarks == 'A']
        nr_reserved_hours = float(sum(getattr(sl, 'reserved_nr_hours', 0.0) for sl in nr_slots))
        nr_realized_hours = float(sum(getattr(sl, 'realized_nr_hours', 0.0) for sl in nr_slots))
        nr_uncovered_labor_hours = float(sum(
            max(0.0, getattr(sl, 'realized_nr_hours', 0.0)
                - getattr(sl, 'reserved_nr_hours', 0.0))
            for sl in nr_slots
        ))
        reserve_values = np.asarray(
            [getattr(sl, 'reserved_nr_hours', 0.0) for sl in nr_slots], dtype=float)
        realized_values = np.asarray(
            [getattr(sl, 'realized_nr_hours', 0.0) for sl in nr_slots], dtype=float)
        if (len(nr_slots) >= 2 and np.std(reserve_values) > 0
                and np.std(realized_values) > 0):
            nr_reserve_realized_corr = float(np.corrcoef(reserve_values, realized_values)[0, 1])
        else:
            nr_reserve_realized_corr = float('nan')

        ##### Results dictionary #####
        results_overview = {
            # Simulation info
            'sim_id': self.id,
            'sim_run_id': self.id_simulation_run,
            'sim_iteration': self.iteration,
            'fleet': str(self.scenario['Aircraft_types']),
            'aircraft_removed': str(self.scenario['Aircraft_remove']),
            'aircraft_additional': str(self.scenario['Aircraft_additional']),
            'reserves_per_day': int(self.scenario['Reserves_per_day']),
            'schedule': self.scenario['Rotations_start'],
            'slots_schenario': self.scenario['Slotsnorm_scenario'],
            'sim_duration': G.SIM_DURATION,
            # PAPER_DESIGN factors (logged so each run cell is self-describing)
            'nr_mode': getattr(G, 'NR_MODE', 'static'),
            'nr_buffer_quantile': getattr(G, 'NR_BUFFER_QUANTILE', None),
            'nr_variance_scale': getattr(G, 'NR_VARIANCE_SCALE', None),
            'maintenance_schedule_mode': MODULES.MAINTENANCE_SCHEDULE,
            'run_time': round((timeit.default_timer() - self.timer_start)/60),

            # KPIs
            'completion_factor': completion_factor,
            'completion_factor_recovery': completion_factor_recovery,   # Does not consider rotations cancelled during
                                                                        # tail assignment

            'tasks_execution_factor': tasks_execution_factor,
            'interval_spillage': interval_spillage,   # Paper-2 proxy, operations-side (% of interval)

            'rotations_delay_dep': rotations_delay_dep,
            'rotations_delay_arr': rotations_delay_arr,
            'flights_delay_dep': flights_delay_dep,
            'flights_delay_arr': flights_delay_arr,
            'slots_delay_dep': slots_delay_dep,
            'slots_delay_arr': slots_delay_arr,

            # Rotations
            'rotations_total': rot_executed + rot_cancelled,
            'rotations_executed': rot_executed,
            'rotations_cancelled': rot_cancelled,   # Include rotations cancelled during recovery
            'rotations_cancelled_recovery': rot_cancelled_recovery,

            # Slots
            'slots_executed': len(self.slots_executed),
            'slots_cancelled': len(self.slots_cancelled),
            'slots_AOG': len([sl for sl in self.slots_executed if sl.remarks=='AG']),
            'slots_included_in_aog': dynamic_results['slots_included_in_aog'],
            'ground_time_hours': ground_time_hours,
            'nr_overrun_hours': nr_overrun_hours,
            'nr_reserved_hours': nr_reserved_hours,
            'nr_realized_hours': nr_realized_hours,
            'nr_uncovered_labor_hours': nr_uncovered_labor_hours,
            'nr_reserve_realized_corr': nr_reserve_realized_corr,

            # Recovery
            'aircraft_swaps': dynamic_results['ac_swaps'],
            'aircraft_swaps_2hr': dynamic_results['ac_swaps_2hr'],
            'aircraft_swaps_4hr': dynamic_results['ac_swaps_4hr'],
            'aircraft_swaps_6hr': dynamic_results['ac_swaps_6hr'],
            'slots_delayed_ffs': dynamic_results['slot_delayed_ffs'],
            'slot_swaps': dynamic_results['slot_swaps'],
            'slot_swaps_opportunities': dynamic_results['slot_swaps_opportunities'],
            'recovery_module_disr_call_count': int(dynamic_results['recovery_module_disr_call_count']),
            'recovery_module_AOG_call_count': int(dynamic_results['recovery_module_AOG_call_count']),

            # Tasks
            'tasks_total': tasks_total,
            'tasks_executed': tasks_executed,
            'tasks_missed': tasks_missed,
            'requirements_anticipation_target': requirements_anticipation_target,
            'requirements_executed_anticipation_mean': requirements_anticipation_mean

        }

        # Make a dataframe
        results_overview_df = pd.DataFrame([results_overview])

        return results_overview, results_overview_df


    # =================================================================================#
    # VERIFICATION
    # =================================================================================#
    def __check_results(self):
        ''' Check if there is any inconsistency in results'''

        ##### ROTATION #####
        # Check if rotation not found
        for rotation in self.rotations:
            if rotation not in self.rotations_open \
                    and rotation not in self.rotations_cancelled \
                    and rotation not in self.rotations_executed \
                    and rotation.dep_sched <= self.__get_now():

                log_error('Rotation'+ rotation.id+ 'not simulated')

        # Check if rotation in more than one list
        rotations_cancelled_executed = [rt for rt in self.rotations_cancelled if rt in self.rotations_executed]
        rotations_cancelled_open = [rt for rt in self.rotations_cancelled if rt in self.rotations_open]
        rotations_open_executed = [rt for rt in self.rotations_open if rt in self.rotations_executed]
        if rotations_cancelled_executed!=[] or rotations_cancelled_open!=[] or rotations_open_executed!=[]:
            log_error('Rotation in more than one list')

        # Check if rotation not cancelled but should be
        rotations_should_be_cancelled = [rt for rt in self.rotations_open
                                         if rt.dep_sched < self.__get_now()-timedelta(weeks=1)]
        if rotations_should_be_cancelled!=[]:
            log_error('rotations', rotations_should_be_cancelled)

        ##### TASKS #####
        # Check that tasks not executed not missed have due date later than now - buffer
        tasks_future = [ts for ts in self.tasks if ts not in self.tasks_executed and ts not in self.tasks_missed
                        and ts not in self.tasks_in_progress]
        tasks_future_gone_due = [ts for ts in tasks_future if ts.dateDue<self.__get_now()-timedelta(days=1)]

        for task in tasks_future_gone_due:
            log_error('Task ' + task.id + ' appears as future task but gone due on '+
                      dt.strftime(task.dateDue, '%Y-%m-%d %H:%M:%S')+'. Now is '+self.__now_string())



    # =================================================================================#
    # VALIDATION
    # =================================================================================#

    def __validate_model(self):
        ''' Function that calls the required validation functions'''
        # Validation of disruption events duration
        if 1 in G.VERIFICATION_VALIDATION:
            self.__validate_disruption_events()
        if 2 in G.VERIFICATION_VALIDATION:
            self.__validate_delays_outstations()
        # Show generated plots
        plt.show()

    def __validate_delays_outstations(self):
        ''' Graphically validate delays at outstations by comparing them to historical data'''
        delays = self.disruptions_outstations
        fig, axs = plt.subplots(1)
        plt.suptitle('Distributions of delays at outstations')

        # Print probability of having delays == 0
        p_delay_is_zero_simulated = len([dl for dl in delays['delays_validation'] if dl == 0]) / \
                                    len(delays['delays_validation'])

        p_delay_is_zero_historical = delays['probability_no_delay']
        print('Delays at outstations: probability zero delay is ', round(p_delay_is_zero_simulated, 2),
              'simulated,', round(p_delay_is_zero_historical, 2), 'historical')

        # Historical data histogram
        delays_durations = delays['delays_historical']
        axs.hist(delays_durations, bins=range(floor(min(delays_durations)), ceil(max(delays_durations))),
                       density=True, alpha=0.5, label='Historical')
        # Simulation histogram - only draw delays > 0
        delays_to_draw = delays['delays_validation']
        axs.hist(delays_to_draw,
                       bins=range(floor(min(delays_durations)), ceil(max(delays_durations))),
                       density=True, alpha=0.5, label='Simulated')
        # Fitted distribution
        delays_fitted_dist = delays['delays_fitted_dist']
        x = np.array(range(round(min(delays_durations)), round(max(delays_durations))))
        y = delays_fitted_dist.model['distr'].pdf(x, *delays_fitted_dist.model['arg'],
                                                  loc=delays_fitted_dist.model['loc'],
                                                  scale=delays_fitted_dist.model['scale'])

        axs.plot(x, y)
        axs.grid()
        axs.set(xlabel='[min]')
        axs.legend()

    def __validate_disruption_events(self):
        ''' Graphically validate disruption events duration by comparing them to historical data'''
        distributions_disruptions = self.disruptions_hub
        fig, axs = plt.subplots(len(distributions_disruptions.disruption_levels), 2, sharex='col')
        plt.suptitle('Delays distributions and disruption events distribution for different disruption severity level')


        for i in range(len(distributions_disruptions.disruption_levels)):
            disr = distributions_disruptions.disruption_levels[i]

            # AIRCRAFT DELAYS
            # Print probability of having delays == 0
            p_delay_is_zero_simulated = len([dl for dl in self.disruptions_hub.disruption_levels[i]['delays_validation'] if dl == 0]) / \
                                        len(self.disruptions_hub.disruption_levels[i]['delays_validation'])

            p_delay_is_zero_historical = disr['probability_no_delay']
            print('Disruption level', disr['levelId'], ': probability zero delay is ',round(p_delay_is_zero_simulated,2),
                  'simulated,',round(p_delay_is_zero_historical,2),'historical')

            delays_durations = disr['delays_duration']
            axs[i, 0].set_title(disr['level'])
            # Historical data histogram
            axs[i, 0].hist(delays_durations, bins=range(floor(min(delays_durations)), ceil(max(delays_durations))),
                           density=True, alpha=0.5, label='Historical')
            # Simulation histogram - only draw delays > 0
            delays_to_draw = self.disruptions_hub.disruption_levels[i]['delays_validation']
            axs[i, 0].hist(delays_to_draw,
                           bins=range(floor(min(delays_durations)), ceil(max(delays_durations))),
                           density=True, alpha=0.5, label='Simulated')
            # Fitted distribution
            delays_fitted_dist = disr['delays_fitted_dist']
            x = np.array(range(round(min(delays_durations)), round(max(delays_durations))))
            y = delays_fitted_dist.model['distr'].pdf(x, *delays_fitted_dist.model['arg'], loc=delays_fitted_dist.model['loc'],
                                                      scale=delays_fitted_dist.model['scale'])

            axs[i, 0].plot(x, y)
            axs[i, 0].grid()


            # DISRUPTION EVENTS DURATION
            axs[i, 1].set_title(disr['level'])
            # Historical data histogram
            events_durations = disr['events_duration']
            axs[i, 1].hist(events_durations, bins=range(floor(min(events_durations)), ceil(max(events_durations))),
                           density=True, alpha=0.5, label='Historical')
            # Simulation histogram
            axs[i, 1].hist(self.disruptions_hub.disruption_levels[i]['events_validation'], bins=range(floor(min(events_durations)), ceil(max(events_durations))),
                           density=True, alpha=0.5, label='Simulated')
            # Distributions fitted to historical data
            events_fitted_dist = disr['events_fitted_dist']
            x = np.linspace(0, 30, 1000)
            y = events_fitted_dist.model['distr'].pdf(x, *events_fitted_dist.model['arg'],
                                                      loc=events_fitted_dist.model['loc'],
                                                      scale=events_fitted_dist.model['scale'])

            # Shift distribution to the right by one unit
            x = [val+1 for val in x]
            axs[i, 1].plot(x, y, label= 'Fitted to historical')

            axs[i, 1].grid()


        axs[len(distributions_disruptions.disruption_levels)-1, 0].set(xlabel='[min]')
        axs[len(distributions_disruptions.disruption_levels)-1, 1].set(xlabel='[20 min units]')
        axs[0,1].legend()
