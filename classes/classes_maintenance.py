from datetime import datetime, date, timedelta
from config import G
from math import floor
import pandas as pd
import random

class SlotNorm:
    ''' Class of repetitive maintenance slots given as input to simulation '''
    def __init__(
            self,
            id,
            ac_subtype,
            time_start,
            time_end,
            day_start,
            day_end,
            cycle_duration,
            type = '',
            location = '',
            ac_allowed = None,
            task_hangar_preparation = None,
            remarks = ''
    ):

            self.id = id
            self.subtype = ac_subtype
            self.time_start = time_start
            self.time_end = time_end
            self.day_start = day_start
            self.day_end = day_end
            self.cycle_duration = cycle_duration
            self.type = type
            self.location = location
            self.ac_allowed = ac_allowed
            self.task_hangar_preparation = task_hangar_preparation
            self.remarks = remarks

            self.duration = self.__find_duration()
            self.ac_turn = 0
            self.day_order = None


    def __find_duration(self):
        ''' Given start and end day and time of slot, find duration'''
        # Find number of days between sttart and end
        n_days = self.day_end - self.day_start
        if n_days < 0:
            n_days = n_days + self.cycle_duration

        # Find number of hours between start and end
        hours_diff =  datetime.combine(date.today(), self.time_end) - datetime.combine(date.today(), self.time_start)
        n_hours = hours_diff.total_seconds()/3600

        # Slot duration
        duration = timedelta(hours=(n_days * 24 + n_hours))
        if duration <= timedelta(minutes=0):
            raise Exception('Slot cannot have a negative duration')

        return duration

    def find_acTurn(self):
        '''
        Find aircraft that should be assigned to instance according to drumbeat logic
        '''
        # Find registration
        ac = self.ac_allowed[self.ac_turn]
        # Update turn count
        self.ac_turn += 1
        if self.ac_turn == len(self.ac_allowed):
            self.ac_turn = 0
        return ac


class Slot:
    ''' Class of maintenance slots instanciated from the norm slots '''
    def __init__(
            self,
            id,                               # ID or Sequence number
            subtype,
            dateStart_init,                   # Scheduled maintenance start
            dateEnd_init,                     # Scheduled maintenance end
            remarks,                          # Slot type, i.e. TO or A check slot
            duration,
            cycle = None,                     # Repetition of the slots norm to which slots is assigned
            dateStart_final = None,           # Historical maintenance start
            dateEnd_final = None,             # Historical maintenance end
            aircraft_pre_assigned = None,         # Historical tail assignment
            slotNorm = None,                  # Corresponding norm slot
            kindMnt = '',                     # H/P by type of work included
            location= '',                     # H/P by location
            slotNorm_instance = None,
            task_hangar_preparation = None,    # Task to account for operations to be executed when maint is in hangar
            labor_max = None,                    # Max labor that can be scheduled within the slot
            labor_max_per_task = None           # Max labor hours of single task that can be executed in a slot
                 ):

        self.id = id
        self.subtype = subtype
        self.dateStart_init = dateStart_init
        self.dateEnd_init = dateEnd_init
        self.dateStart_final = dateStart_final
        self.dateEnd_final = dateEnd_final
        self.remarks = remarks
        self.duration_init = duration
        self.cycle = cycle
        self.aircraft_pre_assigned = aircraft_pre_assigned
        self.slotNorm = slotNorm
        self.kindMnt = kindMnt
        self.location = location
        self.slotNorm_instance = slotNorm_instance
        self.task_hangar_preparation = task_hangar_preparation

        self.towing_time = self.__find_towing_time()
        self.tasks = []
        if labor_max == None:
            self.laborMax = self.__find_laborMax()
        else:
            self.laborMax = labor_max
        if labor_max_per_task == None:
            self.laborMax_per_task = self.laborMax
        else:
            self.laborMax_per_task = labor_max_per_task
        self.aircraft = None

        # Original schedule of slot. Needed for slots scheduled in ffs and for slot swap
        self.dateStart_original = dateStart_init
        self.dateEnd_original = dateEnd_init

        self.duration_final = None              # Final duration including non routines
        self.duration_scheduled = None         # Scheduled duration: max(scheduled_work_duration, scheduled_work_labor)
        self.scheduled_work_duration = None     # Total labor scheduled in slot
        self.scheduled_work_labor = None        # Duration of max task scheduled in slot
        self.workpackage_due_date = None        # Earliest due date of the tasks assigned to the slot
        self.swap = None                        # Other slot, if swap between the two is done
        self.free_fleet_space = False           # True if slot is moved to free fleet space
        self.cancellation_reason = None         # When slot is cancelled
        self._val_recovery_prec_assignment = None
        self._val_recovery_ffs = False
        self.postponed_after_aog = False
        self.included_in_aog = False
        self.workpackage_anticipation = None
        self.aircraft_clean_days = None
        self.reserved_nr_hours = 0.0             # Planned NR labour the crew was sized for (workforce model)
        self.realized_nr_hours = 0.0             # Realized NR labour at execution
        self.nr_reserve_basis = None             # predicted_quantile / rolling_total_mean / cold_start_expected
        self.nr_reserve_history = []             # Completed work-package totals used by the static policy


    def __find_towing_time(self):
        ''' Returns the slot towing time'''
        if self.location == 'P':
            towing_time = timedelta(minutes=0)
        elif self.location == 'H':
            towing_time = timedelta(minutes=G.TOWING_HANGAR)
        else:
            raise Exception('Slot location not supported')
        return towing_time


    def __find_laborMax(self):
        '''
        Find labor max that can be included in a slot.
        A-check slot: config max labor
        other slots: duration of the slot multiplied by the config factor
        '''
        if self.kindMnt == 'A':
            laborMax = timedelta(hours=G.A_CHECK_LABOR_TOTAL)
        else:
            laborMax = floor((self.dateEnd_init-self.dateStart_init).total_seconds()/3600)*G.LABOR_MAX_FACTOR
            laborMax = timedelta(hours=laborMax)
        return laborMax

    def find_workpackage_due_date(self):
        ''' Given a slot, finds the assigned workpackage due date as the date on which the first task goes due'''
        # If no tasks are assigned to the workpackage, the due date is set to 5 years from the slot scheduled start date
        if self.tasks == []:
            due_date = self.dateStart_init + timedelta(days=5*365)
        else:
            due_date = min([ts.dateDue for ts in self.tasks])
        self.workpackage_due_date = due_date


    def add_non_routines(self, realized_nr_hours, reserved_nr_hours=0.0):
        ''' Record realized NR labour on the slot and size its duration under the workforce model.

        The A-check crew is sized at planning to complete (routine + reserved NR) within the 24h
        slot; reserved NR is the prediction/buffer allowance. At execution only realized NR ABOVE
        the reserve cannot be absorbed by the fixed crew and so extends the slot (delay). See
        compute_duration('actual'). '''
        self.reserved_nr_hours = float(reserved_nr_hours)
        self.realized_nr_hours = float(realized_nr_hours)
        if realized_nr_hours != 0:
            # Generate task for the realized NR labour (workforce demand / labour accounting)
            NR_task_id = 'NR_'+self.id
            NR_task = Task(id = NR_task_id,
                           durationEst = timedelta(hours=0),
                           laborEst = timedelta(hours=realized_nr_hours),
                           aircraft = self.aircraft,
                           dateArrival = None,
                           dateReady = None,
                           dateDue = None,
                           workType = None,
                           type = 'NON-ROUTINE',
                           info = ''
                           )
            self.tasks.append(NR_task)
        self.compute_duration('actual')


    def compute_duration(self, duration_type):
        '''
        If slot is AG or LM slot, then both the scheduled and final duration as fixed and las as the initial duration

        If slot is A-check, the scheduled duration is computed as the initial duration + the duration of the NR slots

        For TO slots:
        If duration_type == 'scheduled', compute the scheduled labor hours and work duration
        If duration_type == 'actual', estimate the duration of the slot as maximum of:
        - maximum duration of task included in its workpackage
        - total labor hours divided by available labor hours per hour

        '''
        # If slot is AG, duration is fixed
        if self.remarks == 'AG' or self.remarks=='LM':
            self.duration_scheduled = self.duration_init
            self.duration_final = self.duration_init

        elif self.remarks == 'A' and duration_type == 'scheduled':
            self.duration_scheduled = self.duration_init
            self.duration_final = self.duration_init
            tasks = [ts for ts in self.tasks if ts.type != 'NON-ROUTINE']
            _, labor_total, duration_max = self.compute_duration_and_labor_from_tasks_list(tasks,
                                                                                           preparation_task=None)

            self.scheduled_work_duration = duration_max
            self.scheduled_work_labor = labor_total

        elif self.remarks == 'A' and duration_type == 'actual':
            # Workforce model: the crew is sized to finish the PLANNED work (routine + reserved NR)
            # in the 24h slot. Actual duration scales with REALIZED work; only realized NR above the
            # reserve extends the slot (delay). Under-buffering / poor prediction -> reserve too low
            # -> realized > planned -> overrun. Adequate buffer -> realized <= planned -> no delay.
            routine = sum(ts.laborEst.total_seconds()/3600 for ts in self.tasks
                          if ts.type != 'NON-ROUTINE')
            planned = routine + self.reserved_nr_hours
            realized = routine + self.realized_nr_hours
            if planned <= 0:
                self.duration_final = self.duration_init
            else:
                self.duration_final = self.duration_init * (realized / planned)

        # Find scheduled duration
        elif duration_type == 'scheduled':
            # Consider all tasks included in work package apart from non-routines
            tasks = [ts for ts in self.tasks if ts.type != 'NON-ROUTINE']
            duration, labor_total, duration_max = self.compute_duration_and_labor_from_tasks_list(tasks,
                                                                                                  self.task_hangar_preparation)

            # Set values
            self.scheduled_work_duration = duration_max
            self.scheduled_work_labor = labor_total
            self.duration_scheduled = duration

            # Initialize duration final to scheduled duration
            self.duration_final = self.duration_scheduled

        elif duration_type == 'actual':
            # Consider all tasks included in work package apart from non-routines
            tasks = self.tasks
            duration, _, _ = self.compute_duration_and_labor_from_tasks_list(tasks,
                                                                             self.task_hangar_preparation,
                                                                             type='actual')

            self.duration_final = duration

        else:
            raise Exception('duration_type not supported')

    @staticmethod
    def compute_duration_and_labor_from_tasks_list(tasks_list, preparation_task, type='scheduled'):
        '''
        Given a list of tasks, returns:
        - duration: duration of the slot estimated as maximum between max duration and duration computed based on
        included labor hours, by also considering the labor hours required for preparing the airplane for maintenance
        (preparation task)
        - labor_total: total labor hours in work package (excluding preparation task)
        - duration_max: max duration among the tasks included in the work package
        '''

        # If slot is empty, duration is zero
        if tasks_list == []:
            duration = timedelta(seconds=0)
            labor_total = timedelta(seconds=0)
            duration_max = timedelta(seconds=0)
        else:
            if type == 'scheduled':
                labor_total = sum([ts.laborEst for ts in tasks_list], timedelta())
            elif type == 'actual':
                labor_total = sum([ts.laborAct for ts in tasks_list], timedelta())
            else:
                raise Exception('Type parameter not supported')
            duration_max = max([ts.durationEst for ts in tasks_list])

            # Add preparation task for computing slot duration
            if preparation_task == None:
                labor_total_with_preparation = labor_total
            else:
                labor_total_with_preparation = labor_total + preparation_task.laborEst

            duration_labor = labor_total_with_preparation / G.LABOR_AVAILABLE
            duration = max([duration_max, duration_labor])

        return duration, labor_total, duration_max



class Requirement:
    def __init__(
            self,
            subtypes,
            ac_type,
            code,
            info,
            workType,
            flightHours,
            flightCycles,
            calendarDays,
            laborEst,
            durationEst,
            req_class,
            type,
            initial_due_days_by_aircraft=None

    ):

        self.code = code
        self.subtypes = subtypes
        self.ac_type = ac_type
        self.info = info
        self.workType = workType
        self.intervals = self.__make_interval_dict(flightHours, flightCycles, calendarDays)
        self.laborEst = self.__make_timedelta_hours(laborEst)
        self.durationEst = self.__make_timedelta_hours(durationEst)
        self.req_class = req_class
        self.type = type
        self.initial_due_days_by_aircraft = initial_due_days_by_aircraft or {}

        self.id = self.__find_id()
        self.instances = []
        self.inst_idCount = 0
        self.intervalMin = None
        self.simulation_start_weekday = None


    def __make_interval_dict(self, FH, FC, CD):
        ''' This function unifies the intervals within one dictionary'''
        FH = self.nan_to_None(FH)
        FC = self.nan_to_None(FC)
        CD = self.nan_to_None(CD)
        interval_dict = {
            'FH': FH,
            'FC': FC,
            'CD': CD
        }
        return interval_dict

    @staticmethod
    def nan_to_None(x):
        ''' If input x is nan, them make it into None'''
        if pd.isnull(x):
            x = None
        return x


    @staticmethod
    def __make_timedelta_hours(hours):
        ''' Given an integer or float, this function makes it into a timedelta [hours] and removes milliseconds from it'''
        hours_timedelta = timedelta(hours=hours)
        hours_timedelta = hours_timedelta - timedelta(microseconds=hours_timedelta.microseconds)
        return hours_timedelta


    def __find_id(self):
        '''
        A requirement is uniquely defined by the combination of its assembly and requirement code.
        The id is given by the combination of these to factors
        '''
        id = self.ac_type + '|' + self.code
        return id

    def find_shortest_interval(self, season):
        '''
        Find strictest interval based on average utilization
        '''
        utilization = next((ut for ut in G.AIRCRAFT_UTILIZATION
                           if ut['season']==season and ut['ac_type']==self.ac_type),None)
        if utilization == None:
            raise Exception('Utilization not found')

        high_interval = 999999999

        FH_days = high_interval
        if self.intervals['FH'] != None:
            FH_days = floor(self.intervals['FH'] / utilization['FH'])

        FC_days = high_interval
        if self.intervals['FC'] != None:
            FC_days = floor(self.intervals['FC'] / utilization['FC'])

        CD_days = high_interval
        if self.intervals['CD'] != None:
            CD_days = self.intervals['CD']

        shortest_days = min(CD_days, FH_days, FC_days)
        shortest_interval = timedelta(days=shortest_days)

        # Check that interval was found
        if shortest_days == high_interval:
            raise Exception('Interval was not found')

        self.intervalMin = shortest_interval


    def generate_instance(self, execution_type, aircraft=None, today=None, task=None, LM_start=None, sim_start=None):
        '''
        Generate new instance of requirement given previous instance
        Args:
            -execution_type='executed': task executed today
                            'executed_LM': tast executed in line maintenance bin, and assumed to be executed at a
                                            fraction of its interval, or when the slot start if computed execution
                                            date is earlier
                            'missed': due date reached, task assumed to be executed at a fraction of its interval
                            'first': first instance of requirement for a registration
            -aircraft = [Aircraft] aircraft to which the task is assigned
            -today = [datetime] simulation current day, necessary for 'executed' type
            -task = [Task] previous instance. Needed for 'executed_LM' and 'missed' types
            -LM_start: only needed for type 'executed_LM'
        Returns:
             -new generated instance
        '''

        # ARRIVAL AND DUE DATE
        # If first instance, generate due date
        if execution_type == 'first':
            initial_due_days = self.initial_due_days_by_aircraft.get(aircraft.id)
            if initial_due_days is None:
                dateArrival = self.__find_date_arrival_first_instance(sim_start)
            else:
                dateArrival = sim_start + timedelta(days=float(initial_due_days)) - self.intervalMin
        # If prev task executed today, today is the arrival date for the new instance
        elif execution_type == 'executed':
            dateArrival = today
        # If prev task missed or executed in LM bin, assume it was performed at some percentage of its interval,
        # coinciding with the new instance arrival
        elif execution_type == 'executed_LM':
            dateArrival = task.dateDue - (1-G.REQUIREMENTS_EXECUTION_LM)*self.intervalMin
            if dateArrival < LM_start:
                dateArrival = LM_start
        elif execution_type == 'missed':
            dateArrival = task.dateDue - (1-G.REQUIREMENTS_MISSED_EXECUTION)*self.intervalMin
        else:
            raise Exception('Task type not supported')
        # Find due date assuming the found interval to the arrival date
        dateDue = dateArrival + self.intervalMin
        dateDue = dateDue.replace(hour=23, minute=59, second=0)

        if execution_type!='first':
            task.dateExecution = dateArrival

        # GENERATE TASK INSTANCE
        task_generated = Task(
            id = self.__find_new_id(),
            durationEst = self.durationEst,
            laborEst = self.laborEst,
            requirement = self,
            aircraft = aircraft,
            dateArrival = dateArrival,
            dateReady = dateArrival,
            dateDue = dateDue,
            workType = self.workType,
            type = 'REQUIREMENT',
            info=self.info,
            workpackage=None
        )
        # Add generated task to list of generated instances
        self.instances.append(task_generated)

        return task_generated

    def generate_instances_for_horizon(self, aircraft, sim_start, sim_end):
        """Generate every recurring instance due within a one-shot planning horizon."""
        initial_due_days = self.initial_due_days_by_aircraft.get(aircraft.id)
        if initial_due_days is None:
            first_arrival = self.__find_date_arrival_first_instance(sim_start)
            due_date = first_arrival + self.intervalMin
        else:
            due_date = sim_start + timedelta(days=float(initial_due_days))

        tasks = []
        while due_date <= sim_end:
            due_date = due_date.replace(hour=23, minute=59, second=0)
            date_arrival = due_date - self.intervalMin
            task = Task(
                id=self.__find_new_id(),
                durationEst=self.durationEst,
                laborEst=self.laborEst,
                requirement=self,
                aircraft=aircraft,
                dateArrival=date_arrival,
                dateReady=date_arrival,
                dateDue=due_date,
                workType=self.workType,
                type='REQUIREMENT',
                info=self.info,
                workpackage=None
            )
            self.instances.append(task)
            tasks.append(task)
            due_date = due_date + self.intervalMin

        return tasks

    def __find_date_arrival_first_instance(self, sim_start):
        '''
        This function finds the arrival date of the first instance of a requirement.
        The arrival date is chosen randomly in the time that goes from one interval before the simulation start date
        and the simulation start date. The arrival date is then shifted by a fixed number of days so that tasks are not
        due right at the beginning of the simulation.
        :return: The arrival date of the first instance of a requirement [datetime]
        '''
        # min interval as integer [days]
        interval_min_days = self.intervalMin.days
        # sample arrival date
        days_to_arrival = random.randint(1, interval_min_days)
        # Find arrival day by subtracting the sampled days to arrival and adding slack to the selection
        dateArrival = sim_start \
                      - timedelta(days=days_to_arrival) \
                      + timedelta(days=G.REQUIREMENTS_FIRST_INSTANCE_ARRIVAL_SLACK)

        return dateArrival




    def __find_new_id(self):
        '''
        Find new id for requirement instances
        '''
        newId = self.id + '#' + str(self.inst_idCount)
        self.inst_idCount += 1
        return newId




class Task:
    def __init__(
            self,
            id,
            durationEst,
            laborEst,
            aircraft,
            dateArrival,
            dateReady,
            dateDue,
            workType,
            type,                   # 'REQUIREMENT', 'BLOCK', 'MEL', 'ADHOC', 'NSRE'
            info = '',
            requirement = None,
            workpackage = None,
            laborAct = None,
    ):
        self.id = id
        self.durationEst = durationEst
        self.laborEst = laborEst
        if laborAct == None:
            self.laborAct = laborEst
        else:
            self.laborAct = laborAct
        self.requirement = requirement
        self.aircraft = aircraft
        self.dateArrival = dateArrival
        self.dateReady = dateReady
        self.dateDue = dateDue
        self.workType = workType
        self.type = type
        self.info = info
        self.workpackage = workpackage

        self.dateExecution = None
        self.unassign_type = None

