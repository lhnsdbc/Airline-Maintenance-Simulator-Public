import os
import pandas as pd
import pytz
from datetime import datetime as dt
import openpyxl # Necessary to run in Zoe
import gurobipy # Necessary to run in Zoe

class RUN_CONFIG:
        ''' Class that contains info on what to run'''
        MODE = 0                               # 0:simulation, iterations run in SERIES
                                                # 1000: simulation, iterations run in PARALLEL
                                                # 1: map
                                                # 2: generate output for dashboard
                                                # 10: preprocessing DISRUPTIONS distributions from scratch - takes
                                                #       around one hour to run
                                                # 11: preprocessing DISRUPTIONS distributions from pre-extracted
                                                #       empirical distributions
                                                # 12: preprocessing MAINTENANCE distributions from scratch (imports df)
                                                # 13: preprocessing MAINTENANCE distributions from pickled df
                                                # 14: preprocessing AOG distributions
                                                # 20: Compute historical KPIs
                                                # 21: Network validation
                                                # 22: Maintenance validation
                                                # 30: Case reserve aircraft results
                                                # 31: Case health

        MAP_LOG = 'log_20220902_115155'         # name of the log to run in the map

        # List of simulation ids to unify in one file for dashboard
        # RESULTS_DASHBOARD_OUTPUT = ['18_07_22_sw1_20230301_182751', '18_07_22_sw1_rs2_20230301_182845', # Reserve full
        #                             '18_07_22_sw2_20230226_031214', '18_07_22_sw2_rs2_20230226_033219',
        #                             '18_07_22_sw18_20230225_004610', '18_07_22_sw18_rs2_20230225_004919',
        #                             '18_07_22_sw16_20230225_213414', '18_07_22_sw16_rs2_20230225_214025',
        #                             '18_07_22_sw14_20230225_062307', '18_07_22_sw14_rs2_20230225_064104',
        #                             '18_07_22_sw12_20230225_160728', '18_07_22_sw12_rs2_20230225_160509']
        RESULTS_DASHBOARD_OUTPUT = ['18_07_22_sw1_rs0_20230321_094111', '18_07_22_sw1_20230301_182751', # Reserve 0
                                    '18_07_22_sw1_rs2_20230301_182845']#'789_18_07_22_20230310_101920',

        # RESULTS_DASHBOARD_OUTPUT = ['health_18_07_22_c1_20230322_211958', 'health_18_07_22_c2_20230320_100404',
        #                             'health_18_07_22_c3_20230323_003746', 'health_18_07_22_c4_20230320_130240',
        #                             'health_18_07_22_c5_20230323_031831', 'health_18_07_22_c8_20230320_155120',
        #                             'health_18_07_22_wpa2_20230322_212459', 'health_18_07_22_wpa4_20230322_221810',
        #                             'health_18_07_22_wpa6_20230322_232808', 'health_18_07_22_wpa8_20230323_025637']

        REDUCED_MODEL = 0       #NOTE when ==1, then some parameters of the next classes are changed.
                                # See code at the end of script


class MODULES:
        ''' Class that contains the modules to be used in the simulation'''
        TAIL_ASSIGNMENT = 1                     # 0:old model, 1:Standard model
        MAINTENANCE_SCHEDULE = 2                # 0:old model, 1:Standard model, 2:Health
                                                # 5: baseline MRI block scheduler
                                                # 6: SAC-GNN scenario 1 (wide slot combine window)
                                                # 7: SAC-GNN scenario 2 (tight slot combine window)
        DISRUPTIONS_RECOVERY = 0                # 0:pyomo model


class G:
        ''' Class that contains general simulation configuration'''
        PREPROCESSING = 1                       # 0: No preprocessing (import pickles of objects)
                                                # 1: Complete preprocessing
                                                # 2: Preprocess objects (import pickles of dataframes)
                                                # NOTE: Preferably run in preprocessing mode 1. Always use mode 1 when running multiple simulations in parallel
        
        INPUT_SIM_ID = 0                          # 0: Sim id and iteration generated, 1: input from user
        SIM_DURATION = 10                       # [days]
        SIM_ITERATIONS = 2                      # Number of iterations of the full simulation
        PARALLEL_RUN_N_CORES = 8                # Number of cores to be used in mode 1000. max is mp.cpu_count()

        # SCENARIO_SIMULATION = [('18_07_22_sw2_rs2', '20230226_033219', 30),
        #                        ('18_07_22_sw18_rs2', '20230225_004919', 30),
        #                        ('18_07_22_sw16_rs2', '20230225_214025', 30),
        #                        ('18_07_22_sw14_rs2', '20230225_064104', 30),
        #                        ('18_07_22_sw12_rs2', '20230225_160509', 30)]

        SCENARIO_SIMULATION = ['default_run'] # ['health_18_07_22_wpa1', 'health_18_07_22_wpa5', 'health_18_07_22_wpa7',
                                #'health_18_07_22_wpa3',  'health_18_07_22_wpa9']
                                #['health_18_07_22_c2']#'validation_30_05_22']#'789_18_07_22']#18_07_22_sw1
        # [list of str or tuples] List of scenarios to simulate in sequence
        #                        - str: Strings of scenarios ids
        #                        - tuple: [0] scenario id [str]
        #                                 [1] time id of specific simulation run, e.g. '20230224_164555' [str]
        #                                 [2] iteration count to start from when adding iterations to existing scenario [int]


        SAVE_SIMULATION_GANTT = 0               # 0: Simulation Gantt should not be saved, 1 otherwise
        SAVE_RECOVERY_KPIS = 1              # 0: df of recovery solution should not be saved, 1 otherwise
        VERIFICATION_VALIDATION = []#3, 4]        # List of validation codes to run
                                                # 1: validation disruption at hub
                                                # 2: validation delays at outstations
                                                # 3: verification tail and maint slot assignment
                                                # 4: recovery module

        TEST_RECOVERY = False # TODO for testing

        ##### SCHEDULE #####
        SCHEDULE_IMPORT_FROM = 1                # 0: import schedule from BlueLagoon
                                                # 1: import schedule from local file
        A_CHECKS_IMPORT = 0  # 0: From scenario # 1: Historical assignment NOTE:1 not implemented yet


        ROTATIONS_WEEKS = 1                     # Number of weeks to include in the schedule
        EXCLUDE_ROTATIONS_CANCELLED_BEFORE = 14 # [days] rotations cancelled this number of days before
                                                # operations are not considered within the schedule.
        EXCLUDE_ROTATIONS_OVERLAPPING_FLIGHTS = False   # In the imported schedule, some flights are
                                                        # scheduled to overlap other flights in the rotation.
                                                        # If this parameter is False, these rotations are excluded,
                                                        # if True, the rotations are included in the schedule.


        TIMEZONE_LOCAL = pytz.timezone('Europe/Amsterdam')
        TIMEZONE_UTC = pytz.timezone('UTC')
        AIRPORT_BASE = 'AMS'
        ICA_SUBTYPES = ['772', '77W', '789', '781', '332', '333'] # List of IATA subtypes of ICA fleet
        ONLY_ROTATIONS_SELECTED_SUBTYPES = 0    # 1: Only keep rotations that were historical executed with registrations
                                                # and where all the flights are executed by the same aircraft type
                                                # 0: Keep all

        ##### DISCRETE TIME LOG (MAP) #####
        LOG_DISCRETE_TIME = 0                   # 1: Log aircraft locations at discrete time steps. 0: Do not log
                                                #    Needed to show the scenario on the map.
                                                #    Slows down simulation significantly.
        LOG_DISCRETE_TIME_STEP = 5              # [min] Time step for the discretetime log

        ##### RECOVERY_CONTROLLER #####
        CALL_RECOVERY_ROTATION_NEXT_EXPECTED_DELAY_CHANGE = 15         # [min] Minimum delay a rotation must have
        CALL_RECOVERY_ROTATION_EXPECTED_DELAY_CHANGE = 20        # [min] The difference in previously computed expected
                                                                # delay and current estimation must exceed this threshold for calling the recovery module
        SLOT_IN_AOG_DURATION_FACTOR = 2                         # slot overlapping with AOG is included in the AOG if
                                                                # scheduled_duration_slot * factor <= duration AOG

        ##### RECOVERY MODULE #####
        DELAYS_ROTATIONS = [5, 10, 20, 40, 60, 120, 180, 240] # [min] allowed delays for rotations
        DELAYS_SLOTS = [5, 10, 20, 40, 60, 120, 180]     # [min] allowed delays for flex maintenance slots
        RECOVERY_WITHIN_DAYS = 3.5                # [days] within which the original schedule must be recovered
        SLOT_SWAP_MAX_DAYS = 3                  # Max number of days between slots to swap. Max applied is RECOVERY_WITHIN_DAYS
        ASSIGNMENT_CHANGE_MIN_ANTICIPATION = {'subtype_fixed': 2,       # [hours] Min hours between call of recovery and
                                              'subtype_changed': 2.5,    #         rotation assignment change.
                                              'reserve': 2.5,           #         NOTE: excludes TAT/2 or towing
                                              'slot': 2}
        RECOVERY_INCLUDE_MAINTENANCE_ARCS = 0
        RECOVERY_MODULE_ALLOWED_GAP = 1e-5
        # objective function weights
        WEIGHT_ROT_OPERATING_COST_FIRST_PREF = 0    # [MU] Weight of operating a rotation with preferred ac type
        WEIGHT_ROT_OPERATING_COST_SECOND_PREF = 10**4 # [MU] Weight of operating a rotation with second preferred ac type
        WEIGHT_ROT_OPERATING_COST_THIRD_PREF = 2 *10**4  # [MU] Weight of operating a rotation with third preferred ac type
        WEIGHT_ROTATION_DELAYED_PER_MIN = 20    # [MU/min] cost per minute of delay
        WEIGHT_ROTATION_CANCELLED = 10**5      # [MU]
        WEIGHT_ROTATION_CHANGE_ASSIGNMENT_BIG = 2 * 10**2 # [MU] Weight of changing the first flight in rotation
        WEIGHT_ROTATION_CHANGE_ASSIGNMENT_SMALL = 10 ** 2  # [MU] Weight of changing any additional flight in rotation
        WEIGHT_GROUND_ARC = 0                   # [MU]
        WEIGHT_ORIGINAL_SLOTS = 0               # [MU]
        WEIGHT_SLOT_DELAYED_PER_MIN = 25        # [MU/min]
        WEIGHT_SLOT_SWAP = 10**3                  # [MU]
        WEIGHT_SLOT_FREE_FLEET_SPACE = 2000      # [MU]
        WEIGHT_SLOT_TO_CANCELLED = 10 ** 6           # [MU] Cost of cancelling a TO-slot
        WEIGHT_SLOT_ACHECK_CANCELLED = 3* 10 ** 6       # [MU] Cost of cancelling an A-check

        # cancelled, but at this point needed to avoid infeasibility.

        ##### SCHEDULING WINDOWS: TAIL ASSIGNMENT #####
        TAIL_ASSIGNMENT_FIXED_FLEET_ASSIGNMENT = 2      # 0: Free. Takes too much time for use in simulation
                                                        # 1: Fixed subtype assignment as by rotations norm
                                                        # 2: Preferred subtypes allowed
        BUFFER_BEFORE_AFTER_FLIGHT = 55          # [min] Buffer to leave before and after a rotation during tail
                                                # assignment. Note that the total buffer will be two hours, as it is applied to both flies

        TAIL_ASSIGNMENT_WINDOW = 10                     # [days] Number of days for which tail assignment is done
        TAIL_ASSIGNMENT_INTERVAL = 7                    # [days] Interval with which tail assignment is executed
        TAIL_ASSIGNMENT_FIX = RECOVERY_WITHIN_DAYS-0.5  # [days] Window within which the tail assignment
                                                        # is not changed anymore. Due to long (AG) disruption recovery,
                                                        # it must be set to recovery window duration or shorter. For
                                                        # setting longer window, code adaptation is needed.

        ##### SCHEDULING WINDOWS: MAINTENANCE SCHEDULING #####
        SLOTS_ASSIGNMENT_ORDER = 2                      # 0: Free, 1: Drumbeat, 2: Subtype
        MAINTENANCE_SCHEDULE_WINDOW = 21                # [days] Number of days within maintenance is scheduled (rolling mode only)
        MAINTENANCE_SCHEDULE_INTERVAL = 7               # [days] Maintenance scheduling algorithm run every so many days (rolling mode); also the cadence for opening deferred defects and line-maintenance bookkeeping in one-shot mode
        MAINTENANCE_SCHEDULE_ONE_SHOT = True            # If True, the optimiser is called once at the start of the run and plans the whole horizon; the plan is then held fixed (deferred defects folded into slots at execution, disruptions absorbed by the recovery module). If False, the optimiser re-runs every MAINTENANCE_SCHEDULE_INTERVAL days over a MAINTENANCE_SCHEDULE_WINDOW rolling window.
        BLOCK_DUE_GATE_DAYS = None                      # [days] Block baseline (R0) only: if set, a task is eligible for a
                                                        # block slot only when the slot starts within this many days before
                                                        # the task's due date, so tasks execute near due instead of at the
                                                        # first ready block. None = off (original first-ready behaviour).
        SLOTS_ASSIGNMENT_FIX = 3                        # [days] Window within which the assignment of slots is not changed anymore
        DAYS_N_FOR_PENALTY = 0 #7                          # [days] If a task goes due after so many days after end of
                                                        # scheduling window, it does not incur in a penalty when remains unassigned

        ##### RESERVE SLOTS #####
        RESERVE_SLOT_START = TIMEZONE_UTC.localize(dt(3000, 1, 1, 10, 0))
        RESERVE_SLOT_END =TIMEZONE_UTC.localize(dt(3000, 1, 1, 8, 0))
        


        ##### PARAMETERS: MAINTENANCE SCHEDULING #####

        # MODELS PARAMETES
        SOLVER = 1  # 0:CBC, 1:Gurobi, or string
        SOLVER_THREADS = None  # Optional per-solve cap; experiment runner sets this via env.

        ##### PARAMETERS: TAIL ASSIGNMENT #####
        TAIL_ASSIGNMENT_ALLOWED_GAP = 1e-4
        PENALTY_UNASSIGN_ROTATION = 10000
        PENALTY_UNASSIGN_RESERVE_SLOT = 100000
        PENALTY_SUBTYPE_LOW = 200
        PENALTY_SUBTYPE_HIGH = 1000
        PREFERRED_SUBTYPES_GROUPS =[ ['789', '772'],
                                     ['781', '77W'],
                                     ['332', '333']
                                     ]
        SUBTYPES_TYPES = {'789': '787',                 # Correspondance between subtype and type
                          '781': '787',
                          '772': '777',
                          '77W': '777',
                          '332': 'A330',
                          '333': 'A330'}

        # MAINTENANCE SCHEDULING MODEL PARAMETERS
        LM_SLOT_DURATION = 2                            # [hours] Max duration of line maintenance
        LM_SLOT_LABOR_TOTAL = 30                        # [hours] Total labor hours that can be scheduled in a LM bin
        LM_SLOT_LABOR_PER_TASK = 2                      # [hours] Max labor hour of task to be executed in a LM bin
        A_CHECK_LABOR_TOTAL = 30                         # [hours] Extra labor hours that can be scheduled in an A-check
        M = 1000

        if MODULES.MAINTENANCE_SCHEDULE in [1, 5, 6, 7]:
                MAINTENANCE_SCHED_ALLOWED_GAP = 1e-7  # VERSION 1
                # Unassingment of tasks
                PENALTY_UNASSIGN_RECURRING = 10 ** 7  # 10**9 # 10**7 Reserve ac case, scheduler version 1
                PENALTY_UNASSIGN_DD = 5 * 10 ** 7  # 5 * 10**9 # 5 * 10**7 Reserve ac case, scheduler version 1
        elif MODULES.MAINTENANCE_SCHEDULE == 2:
                MAINTENANCE_SCHED_ALLOWED_GAP = 1e-9 # VERSION 2
                PENALTY_AIRCRAFT_CLEAN = 10 ** 7
                PENALTY_SLOT_ANTICIPATION = 10 ** 7
                # Unassingment of tasks
                PENALTY_UNASSIGN_RECURRING = 10**9 #10**9 # 10**7 Reserve ac case, scheduler version 1
                PENALTY_UNASSIGN_DD = 5 * 10**9 #5 * 10**9 # 5 * 10**7 Reserve ac case, scheduler version 1
        else:
                raise Exception('Weights not available for MODULES.MAINTENANCE_SCHEDULE '+
                                str(MODULES.MAINTENANCE_SCHEDULE))

        # Slots activation costs
        PENALTY_SLOT_A_CHECK = -5 * 10**12
        PENALTY_SLOT_GROUND_TIME = 10**3 # 1**3                         # Cost per hour of slot duration
        PENALTY_SLOT_PLATFORM = 10**4 #5**4                            # Cost of activating slot on the platform
        PENALTY_SLOT_HANGAR = 10**5                             # Cost of activating slot in the hangar
        # Aircraft-slot assignment change before next tail assignment
        PENALTY_AIRCRAFT_SLOT_ASSIGNMENT_CHANGE = 10**6
        # Slopes for optimal group-slot assignment
        LI_HEALTH_ORIENTED = 0  # 0: Increasing(faults) or decreasing(requirements) task-slot assignment function
                                # 1: Always increasing after health target passed
        LI_PREFERRED_ANTICIPATION = 5  # Day at which LI weight is minimum if LI is health oriented
        LI_SLOPE_REQ_LOST_INTERVAL = 100
        LI_SLOPE_HEALTH = 200 #100
        LI_SLOPE_FAULTS = -40
        LI_MIN_FAULTS = -PENALTY_UNASSIGN_RECURRING

        # SAC-GNN scheduler integration (validation/inference only)
        SAC_GNN_POLICY_CSV_PATH = r'Data\input\sac_gnn\Maintenance policy data.csv'
        SAC_GNN_INITIAL_STATUS_CSV_PATH = r'Data\input\sac_gnn\final_imputed_maintenance_policy.csv'
        SAC_GNN_MODEL_PREFIX = r'Data\models\sac_gnn\best_model_step_70000'
        SAC_GNN_TARGET_TASK_NODES = 361
        SAC_GNN_K_NEAREST_NEIGHBORS = 10
        SAC_GNN_MAX_EPISODE_LEN = 1500
        SAC_GNN_SLOT_CAPACITY = 500
        SAC_GNN_CPU_THREADS = 4
        SAC_GNN_CONFLICT_RESOLUTION_ENABLED = True
        SAC_GNN_SCENARIO_1_SLOT_COMBINE_WINDOW_DAYS = 60
        SAC_GNN_SCENARIO_2_SLOT_COMBINE_WINDOW_DAYS = 7

        # NON-ROUTINE (NR) PREDICTION INTEGRATION  (see PAPER_DESIGN.md, IMPLEMENTATION_LOG.md)
        # Replaces the fleet-average sampled NR with Paper 1's conditional, feature-driven NR + risk buffer.
        NR_MODE = 'static'                     # 'static'   : fleet-average realization + rolling 5-package total reserve
                                                # 'predicted': conditional NR from offline Paper 1 artifact + risk buffer
        NR_PREDICTED_SIGNAL = 'probability'    # How the prediction enters when NR_MODE=='predicted':
                                                # 'probability': prediction drives the per-slot NR *likelihood*
                                                #     (gate) via per-code p_nr; magnitude from the static
                                                #     distribution. Robust when labour-hour prediction is noisy.
                                                # 'magnitude'  : prediction drives the per-slot NR labour magnitude;
                                                #     gate stays at the fleet probability_NR.
                                                # 'both'       : prediction drives likelihood AND magnitude.
        NR_BUFFER_QUANTILE = 0.5               # Quantile of the conditional NR distribution injected as the risk buffer
                                                # (PAPER_DESIGN Exp. B sweep variable; 0.5 = median, no bias)
        NR_VARIANCE_SCALE = 1.0                # Scales the conditional NR spread around its mean (PAPER_DESIGN Exp. D)
        NR_CALIBRATE_TO_STATIC = True          # Rescale predicted NR so total mass matches static (PAPER_DESIGN sec.4.4)
        NR_STATIC_HISTORY_WINDOW = 5           # Incumbent reserve: mean total NR hours of previous completed A-checks
        NR_STATIC_RESERVE_SCALE = 1.0          # Exp. F: scale incumbent static reserve to match predicted budget
        # Vendored, inference-only artifact (regenerated from real maintenance history when available; see swap note)
        NR_PREDICTION_DIR = 'nr_prediction'    # subfolder under Data/input
        NR_CONDITIONAL_FILE = 'nr_conditional.csv'   # CSV keeps it pandas-only (no pyarrow dependency)
        NR_CALIBRATION_FILE = 'nr_calibration.json'


        # MAINTENANCE TASKS
        REQUIREMENTS_TYPE = 0                                   # 0: Real maintenance requirements, 1: dummy req for testing
        REQUIREMENTS_FRACTION_OF_AIRCRAFT = 0.8                 # Fraction of aircraft within subtype to which the
                                                                # req must be assigned for consideration
        REQUIREMENTS_EXECUTION_LM = 0.9                          # Fraction of the interval at which a requirement is
                                                                # assumed to be executed, if executed in line maint bin
        REQUIREMENTS_MISSED_EXECUTION = 0.9                     # Fraction of the interval at which a requirement is
                                                                # assumed to be executed, if it reaches its due date
        REQUIREMENTS_FIRST_INSTANCE_ARRIVAL_SLACK = 7           # [days] number of day by which the first randomly
                                                                # computed arrival day of the first instance of a
                                                                # requirement is shifted forward

        REQUIREMENTS_EXCLUDE = ['OPR-VER-0101']                 # Requirement codes to exclude
        REQUIREMENTS_MAX_INTERVAL_ALLOWED = 100
        REQUIREMENTS_MIN_INTERVAL_ALLOWED = 15


        LABOR_MAX_FACTOR = 3                    # slotNorm_duration*laborMax_factor = max labor hours that can be
                                                # scheduled in slot
        LABOR_AVAILABLE = 4                     # Available labor hours per hour
        TOWING_HANGAR = 60                      # [min] towing time to and from hangar
        TASK_HANGAR_PREPARATION_LABOR = 230     # [min] Additional task for wp in hangar: labor hours

        AIRCRAFT_UTILIZATION = [{'ac_type': '78', 'season': 'summer', 'FH': 15.6, 'FC': 2}, # TODO modify for 77s and 33s
                                {'ac_type': '77', 'season': 'summer', 'FH': 15.6, 'FC': 2},
                                {'ac_type': '33', 'season': 'summer', 'FH': 15.6, 'FC': 2},
                                {'ac_type': '78', 'season': 'winter', 'FH': 15.4, 'FC': 1.8},
                                {'ac_type': '77', 'season': 'winter', 'FH': 15.4, 'FC': 1.8},
                                {'ac_type': '33', 'season': 'winter', 'FH': 15.4, 'FC': 1.8}]



        SUMMER_PERIOD = {'start': TIMEZONE_LOCAL.localize(dt(3000, 4, 1, 0, 0)),
                         # Summer season: April 2021 - October 2021
                         'end': TIMEZONE_LOCAL.localize(dt(3000, 10, 31, 0, 0))}


        # Operations on input parameters
        # SIMULATION_START = TIMEZONE_UTC.localize(dt.strptime(SIMULATION_START, '%Y-%m-%d'))

class RESULTS:
        HEALTH_MAX = 11
        VALIDATION_SIMULATIONS = ['18_07_22_sw1_20230301_182751'] #'validation_18_07_22_20230318_002819'] #20230222_134249']
        FILE_NAMES = {'flights': 'results_flights',
                         'rotations': 'results_rotations',
                         'slots': 'results_slots',
                         'tasks': 'results_tasks',
                         'overview': 'results_overview',
                         'log_sim': 'log_sim',
                         'tail_assignment': 'tail_assignment',
                         'recovery': 'recovery',
                      'health': 'results_health',
                      'recovery_kpis': 'recovery_kpis'
                      }
        FILES_TO_CONCATENATE_IDENTIFIERS = [FILE_NAMES['flights'], FILE_NAMES['rotations'],
                                            FILE_NAMES['slots'], FILE_NAMES['tasks'], FILE_NAMES['overview'],
                                            FILE_NAMES['tail_assignment'], FILE_NAMES['recovery'],FILE_NAMES['log_sim'],
                                            FILE_NAMES['health'], FILE_NAMES['recovery_kpis']]


        COSTS_CARRIER = {'passengers':350,
                     'cancellation': 125 * 10**3, #150 * 10**3 / 2,
                     'compensation_delay_3hr': 300,
                     'compensation_delay_4hr': 600,
                     'claim_rate': 0.197,
                     'food':15,
                     'food_time': 4*60}

class M:
        ''' This class contains configurations concerning MAINTENANCE DISTRIBUTIONS definition'''
        SLACK_INCLUDE_TASK = 15                 # [min]. When labor hours appear to end after their work package,
        # they are still included in the workpackage if they end within this time

        ####### NON-ROUTINES #######
        MAX_DURATION_NR = 50  # [hours] Max NR labor hours allowed in a work package
        BIN_SIZE_NR = 10
        SAMPLING_SIZE_PER_YEAR_PER_AC_NR = 700

        ####### DEFERRED DEFECTS #######
        MAX_DD_INTER_ARRIVAL = 7
        MAX_DD_COUNT = 5
        MAX_DEFERRAL_DAYS_DD = 180         # [days] DDs deferred by more than this value are excluded from the data
        BIN_SIZE_DD_INTER_ARRIVAL = 1
        SAMPLING_SIZE_PER_YEAR_PER_AC_DD = 1000
        DEFERRAL_CLASSES = {'MELB': 3, 'MELC': 10, 'MELD': 120,
                            'NSRE5': 5, 'NSRE10': 10, 'NSRE20': 20, 'NSRE120': 120,
                            'NSRE150': 150}

        # Exclude DD for inconsistencies between scheduled and actual labor.
        # Task excluded if lab_sched <= multip factor * lab_act and lab_sched > exclude_dd min
        EXCLUDE_DD_LABOR_SCHED_ACT_MULTIPL_FACTOR = 3   # []
        EXCLUDE_DD_LABOR_SCHED_ACT_MIN = 4              # [hours]
        EXCLUDE_DD_MAX_LABOR = 200                      # [hours]

        ####### AOG #######
        AOG_DATA_SOURCE = 1                     # 0: mtop until 2020, 1: BL 2022 data
        AOG_INCLUDED_SLOTS_TYPES = ['AG', 'NR', 'RP', 'AM', '#1', '#2']#,'NB', 'TD']
        # List of slots types to consider when computing AG distributions
                                                # 1: Include AG slots and NR (Non-Release) slots
        MAX_DURATION_AOG = 21                   # [days] Max duration of AOG considered when defining AOG distributions
        BIN_SIZE_AOG_INTER_ARRIVAL = 1
        BIN_SIZE_AOG_DURATION = 0.5
        SAMPLING_SIZE_PER_YEAR_AOG_INTER_ARRIVAL = 365
        SAMPLING_SIZE_PER_YEAR_AOG_DURATION = 365



class P:
        ''' This class contains configurations concerning DISRUPTIONS DISTRIBUTIONS definition'''
        ####### TURN AROUND TIME #######
        SEPARATION_SHORT_LONG_TAT = 80          # [min] Average TAT that defines if a short or long TAT is done at an outer station

        ####### DISRUPTIONS AT AMS #######
        MAX_ANTICIPATION_ALLOWED = -40          # Some flights depart with high departure negative delay (anticipation).
                                                # This value sets the maximum anticipation for which flights are considered.

        DISRUPTION_LEVELS = [{'level': 'norm', 'min': MAX_ANTICIPATION_ALLOWED, 'max': 9, 'levelId': 0},    # Arbitrary definition of disruption levels
                             {'level': 'low', 'min': 10, 'max': 19, 'levelId': 1},
                             {'level': 'mid', 'min': 20, 'max': 39, 'levelId': 2},
                             {'level': 'high', 'min': 40, 'max': 100000, 'levelId': 3}
                             ]
        BRACKETS_N = 54
        BRACKETS_DURATION = 20                   # [min]
        BRACKETS_TIME_START = dt.strptime('06:00', '%H:%M')

        DEPARTURE_TIME_REF = 'ScheduledDepartureTimeLocal'      # Time considered as refence for flight being included in
                                                                # a specific time bracket for disruption events definition


        DELAYS_DISTRIBUTIONS_BIN_SIZE = 5       # [min] Bin sizes for fitting analytical distributions to historical data
        TAT_DISTRIBUTIONS_BIN_SIZE = 5




        DELAYS_SAMPLING_SIZE_PER_YEAR = 30000   # Sample size per year of simulation for aircraft delays, per disruption level and at outstations
        EVENTS_SAMPLING_SIZE_PER_YEAR = 5000    # Sample size per year of simulation for disruption events, per disruption level
        TAT_SAMPLING_SIZE_PER_DAY = 150         # Sample size per day of simulation




# DIRECTORIES
# Find origin folder
# directory_orig = os.path.abspath('')
# directory_orig = os.path.split(directory_orig)[0]
directory_orig = os.path.dirname(os.path.abspath(__file__))
class directories:
        ''' Class that contains directories for input and output'''
        orig = directory_orig
        anemos = directory_orig
        data = os.path.join(directory_orig, 'Data')
        input = os.path.join(data, 'input')
        pickle = os.path.join(data, 'pickle')
        output = os.path.join(data, 'output')

        schedules = os.path.join(input, 'schedules')
        nr_prediction = os.path.join(input, 'nr_prediction')   # vendored NR-prediction artifact (PAPER_DESIGN.md)
        logs_map = os.path.join(output, 'logs_map')
        logs_sim = os.path.join(output, 'logs_sim')
        verification_validation = os.path.join(output,'verification_validation')
        results_KPIs = os.path.join(output, 'results_KPIs')
        aog_distributions = os.path.join(pickle, 'AOG')
        dashboard = os.path.join(output, 'dashboard')


# INPUT FILES
class INPUT_FILES:
        ''' Class that contains all the names of the input files imported'''
        # Aircraft
        AIRCRAFT_REGISTRATIONS = 'AircraftRegistrations'
        AIRCRAFT_ADDITIONAL_REGISTRATIONS = 'AircraftRegistrations_additional'
        AIRCRAFT_TYPE_DETAILS = 'AircraftTypeDetails'
        # Airports
        AIRPORTS = 'Airports'
        AIRPORTS_COORDINATES = 'AirportsCoordinates'
        TIMEZONES = 'TimeZones'
        # Network operations
        SCENARIOS_SIMULATION = 'Scenarios_simulation'
        ROTATIONS_FROM_FILE = 'test_rotations'
        TURNAROUND = 'TurnAround'
        FUTURE_VALUE = 'Future_value'
        # Maintenance operations
        SLOTS_NORM_SCENARIOS = 'Slots_norm_scenarios'
        A_CHECKS = 'A-checks'
        ENGINEERING_REQUIREMENTS_INTERVAL_DUMMY = 'Engineering_Requirements_Interval_dummy'
        ENGINEERING_REQUIREMENTS_INTERVAL = 'Engineering_Requirements_interval'
        ENGINEERING_BLOCKS_INTERVAL = 'Engineering_Blocks_interval'
        ENGINEERING_NON_RECURRING = 'Engineering_non_recurring'
        ENGINEERING_NON_RECURRING_FOLDER = 'Engineering_non_recurring_test'# 'Engineering_non_recurring'
        ENGINEERING_NON_RECURRING_COL_NAMES = 'Engineering_non_recurring_col_names'
        AOG_SLOTS = 'AOG_slots'
        MTOP_SLOTS = 'Mtop_slots_ICA'
        DELAYS_OUTSTATIONS = 'delays_outstations'
        DISRUPTIONS_HUB = 'distributions_disruptions_fitted'
        DISRTRIBUTIONS_AOG = 'distributions_AOG'
        DISTRIBUTIONS_NR = 'distributions_NR'
        DISTRIBUTIONS_DD = 'distributions_DD'
        FLIGHT_DURATION = 'flights_duration'
        TAT = 'TAT'
        TASKS_DDS = 'tasks_DD'
        PICKLE_DATA = 'data'

if RUN_CONFIG.REDUCED_MODEL == 1:
        INPUT_FILES.SLOTS_NORM_SCENARIOS = 'Slots_norm_scenarios_reduced'
        G.SCENARIO_SIMULATION = ['dummy']


# =====================================================================================
# PAPER_DESIGN.md experiment overrides (env-var driven, subprocess-isolated)
# -------------------------------------------------------------------------------------
# The experiment runner (experiments/run_experiments.py) launches one *subprocess per
# experiment cell* and passes the cell's configuration through ANEMOS_* environment
# variables. Reading them here -- at config import -- means import-time-sensitive settings
# (e.g. MODULES.MAINTENANCE_SCHEDULE and its dependent MILP penalty weights) are applied
# cleanly with no cross-run contamination. With no env vars set, defaults are unchanged.
# =====================================================================================
def _recompute_maintenance_weights():
        ''' Re-derive the MAINTENANCE_SCHEDULE-dependent MILP weights (mirrors class G body)
        after MODULES.MAINTENANCE_SCHEDULE is overridden at runtime.'''
        if MODULES.MAINTENANCE_SCHEDULE in [1, 5, 6, 7]:
                G.MAINTENANCE_SCHED_ALLOWED_GAP = 1e-7
                G.PENALTY_UNASSIGN_RECURRING = 10 ** 7
                G.PENALTY_UNASSIGN_DD = 5 * 10 ** 7
        elif MODULES.MAINTENANCE_SCHEDULE == 2:
                G.MAINTENANCE_SCHED_ALLOWED_GAP = 1e-9
                G.PENALTY_AIRCRAFT_CLEAN = 10 ** 7
                G.PENALTY_SLOT_ANTICIPATION = 10 ** 7
                G.PENALTY_UNASSIGN_RECURRING = 10 ** 9
                G.PENALTY_UNASSIGN_DD = 5 * 10 ** 9
        G.LI_MIN_FAULTS = -G.PENALTY_UNASSIGN_RECURRING


def _apply_env_overrides():
        ''' Apply ANEMOS_* environment overrides for batch experiments. Each is optional. '''
        def _get(name):
                return os.environ.get(name)

        # --- NR prediction factor ---
        if _get('ANEMOS_NR_MODE') is not None:
                G.NR_MODE = _get('ANEMOS_NR_MODE')
        if _get('ANEMOS_NR_PREDICTED_SIGNAL') is not None:
                G.NR_PREDICTED_SIGNAL = _get('ANEMOS_NR_PREDICTED_SIGNAL')
        if _get('ANEMOS_NR_BUFFER_QUANTILE') is not None:
                G.NR_BUFFER_QUANTILE = float(_get('ANEMOS_NR_BUFFER_QUANTILE'))
        if _get('ANEMOS_NR_VARIANCE_SCALE') is not None:
                G.NR_VARIANCE_SCALE = float(_get('ANEMOS_NR_VARIANCE_SCALE'))
        if _get('ANEMOS_NR_CALIBRATE') is not None:
                G.NR_CALIBRATE_TO_STATIC = _get('ANEMOS_NR_CALIBRATE') not in ('0', 'false', 'False')
        if _get('ANEMOS_NR_STATIC_HISTORY_WINDOW') is not None:
                G.NR_STATIC_HISTORY_WINDOW = int(_get('ANEMOS_NR_STATIC_HISTORY_WINDOW'))
        if _get('ANEMOS_NR_STATIC_RESERVE_SCALE') is not None:
                G.NR_STATIC_RESERVE_SCALE = float(_get('ANEMOS_NR_STATIC_RESERVE_SCALE'))
        if _get('ANEMOS_DISTRIBUTIONS_NR') is not None:
                INPUT_FILES.DISTRIBUTIONS_NR = _get('ANEMOS_DISTRIBUTIONS_NR')

        # --- Deregulation rung: scheduler + labour caps ---
        sched = _get('ANEMOS_MAINTENANCE_SCHEDULE')
        if sched is not None:
                MODULES.MAINTENANCE_SCHEDULE = int(sched)
                _recompute_maintenance_weights()
        if _get('ANEMOS_BLOCK_DUE_GATE_DAYS') is not None:
                _gate = _get('ANEMOS_BLOCK_DUE_GATE_DAYS')
                G.BLOCK_DUE_GATE_DAYS = None if _gate in ('', 'none', 'None', 'off') else int(_gate)
        if _get('ANEMOS_LM_SLOT_LABOR_TOTAL') is not None:
                G.LM_SLOT_LABOR_TOTAL = float(_get('ANEMOS_LM_SLOT_LABOR_TOTAL'))
        if _get('ANEMOS_A_CHECK_LABOR_TOTAL') is not None:
                G.A_CHECK_LABOR_TOTAL = float(_get('ANEMOS_A_CHECK_LABOR_TOTAL'))
        if _get('ANEMOS_LABOR_MAX_FACTOR') is not None:
                G.LABOR_MAX_FACTOR = float(_get('ANEMOS_LABOR_MAX_FACTOR'))

        # --- Run controls ---
        if _get('ANEMOS_SIM_DURATION') is not None:
                G.SIM_DURATION = int(_get('ANEMOS_SIM_DURATION'))
        if _get('ANEMOS_SIM_ITERATIONS') is not None:
                G.SIM_ITERATIONS = int(_get('ANEMOS_SIM_ITERATIONS'))
        if _get('ANEMOS_MAINTENANCE_ONE_SHOT') is not None:
                G.MAINTENANCE_SCHEDULE_ONE_SHOT = _get('ANEMOS_MAINTENANCE_ONE_SHOT') not in ('0', 'false', 'False', '')
        if _get('ANEMOS_RUN_MODE') is not None:
                RUN_CONFIG.MODE = int(_get('ANEMOS_RUN_MODE'))
        if _get('ANEMOS_PARALLEL_RUN_N_CORES') is not None:
                G.PARALLEL_RUN_N_CORES = int(_get('ANEMOS_PARALLEL_RUN_N_CORES'))
        if _get('ANEMOS_SAC_GNN_CPU_THREADS') is not None:
                G.SAC_GNN_CPU_THREADS = int(_get('ANEMOS_SAC_GNN_CPU_THREADS'))
        if _get('ANEMOS_SOLVER') is not None:   # e.g. 'appsi_highs' / 'gurobi_direct' / 'cbc'
                G.SOLVER = _get('ANEMOS_SOLVER')
        if _get('ANEMOS_SOLVER_THREADS') is not None:
                G.SOLVER_THREADS = int(_get('ANEMOS_SOLVER_THREADS'))
        if _get('ANEMOS_RECOVERY_WITHIN_DAYS') is not None:
                G.RECOVERY_WITHIN_DAYS = float(_get('ANEMOS_RECOVERY_WITHIN_DAYS'))
                G.SLOT_SWAP_MAX_DAYS = min(G.SLOT_SWAP_MAX_DAYS, G.RECOVERY_WITHIN_DAYS)
                G.TAIL_ASSIGNMENT_FIX = G.RECOVERY_WITHIN_DAYS - 0.5
        if _get('ANEMOS_PREPROCESSING') is not None:
                G.PREPROCESSING = int(_get('ANEMOS_PREPROCESSING'))
        if _get('ANEMOS_ROTATIONS_WEEKS') is not None:
                G.ROTATIONS_WEEKS = int(_get('ANEMOS_ROTATIONS_WEEKS'))
        if _get('ANEMOS_SCENARIO') is not None:
                G.SCENARIO_SIMULATION = [_get('ANEMOS_SCENARIO')]


_apply_env_overrides()


# def check_parameter_scenario():
#         filename = INPUT_FILES.SCENARIOS_SIMULATION
#         scenarios = pd.read_excel(os.path.join(directories.input, filename + '.xlsx'))
#         # Filter chosen scenario
#         scenario = scenarios[scenarios['Id']==scenario_id]
