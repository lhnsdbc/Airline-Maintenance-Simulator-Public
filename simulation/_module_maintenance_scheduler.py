import pyomo.environ as pyo
import pandas as pd
from datetime import timedelta, date
import timeit
from config import G
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints
from output.output_functions import log_info, log_error, log_warning
import os
import tempfile
from collections import defaultdict
from config import directories

class MaintenanceSchedulerHealth:
    def __init__(
            self,
            simulation
                ):
        self.simulation = simulation
        self.name = 'schedule_maintenance'+self.simulation.now.strftime('%d/%m/%Y')

        # Init time
        start_time = timeit.default_timer()

        # Init model
        self.model = pyo.ConcreteModel(name='schedule_maintenance')

        mid_time = timeit.default_timer()
        self.__add_sets()
        # print('Time add sets', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__add_decision_variables()
        # print('Time add decision variables', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__add_constraints()
        # print('Time add constraints', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__add_objective()
        # print('Time add objective', timeit.default_timer() - mid_time)

        self.time_initialization = timeit.default_timer() - start_time

    # =================================================================================#
    # SETS
    # =================================================================================#
    def __add_sets(self):
        # General sets
        self.model.set_aircraft = pyo.Set(initialize=self.simulation.aircraft)
        self.model.set_slots = pyo.Set(initialize=self.__initialize_slots_to_and_a())
        self.model.set_slots_to = pyo.Set(initialize=self.__initialize_slots_to())
        self.model.set_slots_lm = pyo.Set(initialize=self.__initialize_slots_lm())
        self.model.set_slots_full = self.model.set_slots | self.model.set_slots_lm
        self.model.set_tasks = pyo.Set(initialize=self.__initialize_set_tasks())

        # Reduced combination sets
        self.model.set_aircraft_slot = pyo.Set(initialize=self.__inialize_set_aircraft_slot(), dimen=2)
        self.model.set_aircraft_slot_orig = pyo.Set(initialize=self.__initialize_set_aircraft_slot_orig(), dimen=2)
        self.model.set_aircraft_orig = pyo.Set(initialize=self.__initialize_set_aircraft_orig())
        self.model.set_task_slot = pyo.Set(initialize=self.__initialize_set_task_slot(), dimen=2)
        self.model.set_task_slot_lm = pyo.Set(initialize=self.__initialize_set_task_slot_lm(), dimen=2)
        self.model.set_task_slot_full = self.model.set_task_slot | self.model.set_task_slot_lm

        # Health sets
        if self.simulation.scenario['MAS_constr_clean'] == 1:
            self.model.set_task_slot_clean = pyo.Set(initialize=self.__initialize_set_task_slot_clean(), dimen=2)
            self.model.set_clean = pyo.Set(initialize=self.__initialize_set_clean(), dimen=3)
        if self.simulation.scenario['MAS_constr_wp_anticipation'] == 1:
            self.model.set_task_slot_anticipation = pyo.Set(initialize=self.__initialize_set_task_slot_anticipation(), dimen=2)

    def __initialize_slots_to_and_a(self):
        ''' Return slots that are not line maintenance slots'''
        return [sl
                for sl in self.simulation.slots_scheduling
                if sl.remarks != 'LM']

    def __initialize_slots_to(self):
        '''Return list of TO slots'''
        return[sl
               for sl in self.model.set_slots
               if sl.remarks == 'TO']

    def __initialize_slots_lm(self):
        ''' Return line maintenance slots'''
        return [sl
                for sl in self.simulation.slots_scheduling
                if sl.remarks == 'LM']

    def __task_penalized_unassignment(self, task):
        '''
        Return True if the unassignment of a task should be penalized, False otherwise
        the final day (included) for which a task is assigned a penalty is computed as:
            end of the scheduling window + the number of days specified by G.DAYS_N_FOR_PENALTY
        '''
        date_until_penalty = self.simulation.maintenance_scheduling_window_end + timedelta(days=G.DAYS_N_FOR_PENALTY)
        return task.dateDue <= date_until_penalty

    def __initialize_set_tasks(self):
        ''' Consider all open tasks, apart from requirements going due after the penalty window. These requirements
        would be postponed anyways to a later opportunity. '''
        return [ts
                for ts in self.simulation.tasks_open
                if ts not in self.simulation.tasks_in_progress
                and (ts.type !='REQUIREMENT'
                     or (ts.type =='REQUIREMENT' and self.__task_penalized_unassignment(ts)))]


    def __inialize_set_aircraft_slot(self):
        ''' In general, an aircraft can be assigned to a slot if there is a match in subtype. However, within the
        fixed assignment window, the assigned slots are fixed to an assigne registration. '''
        return [(ac, sl)
                for ac in self.model.set_aircraft
                for sl in self.model.set_slots
                if self.__find_compatibility_ac_slot(ac, sl) == True]

    def __find_compatibility_ac_slot(self, aircraft, slot):
        '''
        The following cases are considered:
            - Drumbeat (G.SLOTS_ASSIGNMENT_ORDER==1): assign to designated aircraft
            - Fixed subtype (G.SLOTS_ASSIGNMENT_ORDER==2): compatibility is same subtype
            - Fixed subtype or free assignemnt (G.SLOTS_ASSIGNMENT_ORDER==0 or 2) : fix assignment within fixed window
        '''
        # TODO if A-checks should have a fixed assignment, fix it here

        # If slot overlaps with AOG of aircraft, do not assign it
        aog_slot = next((sl for sl in aircraft.slots if sl.remarks == 'AG'), None)
        if aog_slot!=None:
            if slot.dateStart_final - slot.towing_time <= aog_slot.dateEnd_final + aog_slot.towing_time \
                and slot.dateEnd_final + slot.towing_time >= aog_slot.dateStart_final - aog_slot.towing_time:
                return False

        # Drumbeat assignment or fixed subtype
        match(G.SLOTS_ASSIGNMENT_ORDER):
            # DRUMBEAT
            case 1:
                if slot.aircraft_pre_assigned == aircraft:
                   return True
                else:
                    return False

            # FIXED SUBTYPE
            case 2:
                if aircraft.subtype.IATA != slot.subtype:
                    return False

        # FIXED SCHEDULING WINDOW (do not consider if first time running optimization)
        if G.SLOTS_ASSIGNMENT_FIX > 0 and pd.isnull(self.simulation.scheduler_maintenance)==False:
            # End of fixed scheduling window
            fixed_schedule_end = self.simulation.now + timedelta(days=G.SLOTS_ASSIGNMENT_FIX)
            if slot.dateStart_init <= fixed_schedule_end and slot.aircraft != aircraft:
                return False

        # If no incompatibility found, return True
        return True

    def __initialize_set_aircraft_slot_orig(self):
        ''' Set of previous assignment of aircraft to slots for the starting week'''
        # Window to fix is interval with which tail assignment is called
        window_fix_end = self.simulation.now + timedelta(days=G.TAIL_ASSIGNMENT_INTERVAL) \
                         + timedelta(days=G.TAIL_ASSIGNMENT_FIX)
        # Find set of aircraft-slots to fix
        return [(sl.aircraft, sl)
                for sl in self.model.set_slots
                if sl.dateStart_final <= window_fix_end
                and pd.isnull(sl.aircraft) == 0]

    def __initialize_set_aircraft_orig(self):
        ''' Return list of aircraft for which at least one slot assignment should be preferably fixed from previous
        runs of the scheduler'''
        ac_fixed = list(set([ac for (ac,sl) in self.model.set_aircraft_slot_orig]))
        return ac_fixed

    def __initialize_set_task_slot(self):
        return [(ts, sl)
                for ts in self.model.set_tasks
                for sl in self.model.set_slots
                if self.__find_task_compatibility_task_slot(ts, sl) == True]

    def __initialize_set_task_slot_lm(self):
        return [(ts, sl)
                for ts in self.model.set_tasks
                for sl in self.model.set_slots_lm
                if self.__find_task_compatibility_task_slot(ts, sl) == True]

    def __initialize_set_task_slot_clean(self):
        return[(ts, sl)
               for sl in self.model.set_slots
               for ts in self.model.set_tasks
               if self.__task_slot_clean(ts,sl) == True]

    def __task_slot_clean(self, task, slot):
        '''
        A task concerns a slot clean constraint if:
        - The aircraft of the task can be assigned to the slot
        - Arrival date: Falls before the start of the slot
        - Due date: Falls between the slot and G.CLEAN_TARGET days later
        - Ready date: Falls before the start of the slot
        '''
        if (task.aircraft, slot) not in self.model.set_aircraft_slot:
            return False

        # Arrival date
        if task.dateArrival.date() > slot.dateStart_final.date():
            return False

        # Due date
        if (task.dateDue.date() > slot.dateStart_final.date()+timedelta(days=self.simulation.scenario['MAS_clean_target']))\
                | (task.dateDue.date() < slot.dateStart_final.date()):
            return False
        # Ready date
        if task.dateReady.date() > slot.dateStart_final.date():
            return False

        # If due date and ready date within relevant ranges, return True
        return True

    def __initialize_set_clean(self):
        ''' (ts, slh, sl) is included in the set if task ts is relevant for slot sl and can be scheduled in slot slh
        to not activate the clean constraint on slot sl. Note that if task ts can be scheduled in slot sl,
        then set (ts,sl,sl) is also included'''
        return [(ts,slh,sl)
                for (ts,sl) in self.model.set_task_slot_clean
                for slh in self.model.set_slots_full
                if (ts,slh) in self.model.set_task_slot_full
                and slh.dateStart_final <= sl.dateStart_final]

    def __initialize_set_task_slot_anticipation(self):
        ''' Set of tuples (ts, sl), so that task ts is limiting for anticipation of slot sl '''
        return [(ts, sl)
                for (ts,sl) in self.model.set_task_slot
                if ts.dateDue.date() - sl.dateStart_final.date() <= timedelta(days=self.simulation.scenario['MAS_wp_anticipation'])]



    def __find_task_compatibility_task_slot(self, task, slot):
        '''
        A task can be executed in a slot given the following conditions:
             - AIRCRAFT: the aircraft of the task must be compatible with the slot based on aircraft type or specific
                            registration assignment.
             - DUE DATE: the task can be scheduled only before its due date
             - READY DATE: the task can only be executed after its ready date is reached
             - LOCATION: hangar (H) tasks can only be executed in the hangar, while platform (P) tasks can be
                            executed both in the hangar and on the platform
             - DURATION: the duration of the task cannot exceed the duration of the slot
             - LABOR: the total labor needed for the task must not exceed the maximum labor that can be scheduled in
                        a slot
        '''

        # CONDITION AIRCRAFT
        if slot.remarks != 'LM' and (task.aircraft, slot) not in self.model.set_aircraft_slot:
            return False
        if slot.remarks == 'LM' and task.aircraft != slot.aircraft:
            return False

        # CONDITION DUE DATE
        if task.dateDue < slot.dateStart_final:
            return False

        # CONDITION READY DATE
        if task.dateReady > slot.dateEnd_final:
            return False

        # CONDITION LOCATION
        if task.workType == 'H' and slot.location == 'P':
            return False

        # CONDITION DURATION
        if task.durationEst > slot.duration_init:
            return False

        # CONDITION LABOR
        if task.laborEst > slot.laborMax_per_task:
            return False

        # If no incompatibility found, task and slot are compatible
        return True


    # =================================================================================#
    # DECISION VARIABLES
    # =================================================================================#
    def __add_decision_variables(self):
        '''
        Add decision variables to the model
        '''
        # Assign task to slot
        self.model.dv_task_slot = pyo.Var(self.model.set_task_slot_full, domain=pyo.Binary,
                                          initialize=self.__initialize_dv_task_slot())
        # Unassign task
        self.model.dv_task_unassign = pyo.Var(self.model.set_tasks, domain=pyo.Binary,
                                          initialize=self.__initialize_dv_unassign())
        # Assign aircraft to slot
        self.model.dv_aircraft_slot = pyo.Var(self.model.set_aircraft_slot, domain=pyo.Binary,
                                              initialize=self.__init_dv_aircaft_slot())

        # Change of aircraft-slot assignment
        self.model.dv_change_aircraft_slot_assignment = pyo.Var(self.model.set_aircraft_orig,
                                                                domain=pyo.NonNegativeIntegers, initialize=0)

        # Aircraft clean target respected
        # if G.CONSTR_CLEAN == 1:
        if self.simulation.scenario['MAS_constr_clean'] == 1:
            self.model.dv_aircraft_clean_slot = pyo.Var(self.model.set_aircraft_slot, domain=pyo.Binary)

        # Slot workpackage anticipation
        # if G.CONSTR_WP_ANTICIPATION == 1:
        if self.simulation.scenario['MAS_constr_wp_anticipation'] == 1:
            self.model.dv_slot_anticipation = pyo.Var(self.model.set_slots, domain=pyo.Binary)

    def __initialize_dv_task_slot(self):
        '''
        If previus solution exists, return dictionary of initial values for the decision variables for assignment of
        tasks to slots.
        '''
        if self.simulation.scheduler_maintenance != None:
            # Find tuples (group, slot) in common between current and previous model
            task_slot_in_common = [tup for tup in self.model.set_task_slot_full
                                   if tup in self.simulation.scheduler_maintenance.model.set_task_slot_full]
            # Find values corresponding to tuples
            task_slot_values = [round(self.simulation.scheduler_maintenance.model.dv_task_slot[ts, sl]())
                                for (ts, sl) in task_slot_in_common]
            # Generate initialization dictionary
            task_slot_dict = dict(zip(task_slot_in_common, task_slot_values))
        else:
            task_slot_dict = {}

        return task_slot_dict

    def __initialize_dv_unassign(self):
        '''
        If previus solution exists, return dictionary of initial values for the decision variables for task
        unassignment.
        '''
        if pd.isnull(self.simulation.scheduler_maintenance)==0:
            # Find tasks in commond with previous model
            tasks_in_common = [ts for ts in self.model.set_tasks
                               if ts in self.simulation.scheduler_maintenance.model.set_tasks]
            tasks_unassign_value = [round(self.simulation.scheduler_maintenance.model.dv_task_unassign[ts]())
                                    for ts in tasks_in_common]
            tasks_unassing_dict = dict(zip(tasks_in_common, tasks_unassign_value))
        else:
            tasks_unassing_dict = {}

        return tasks_unassing_dict

    def __init_dv_aircaft_slot(self):
        '''
        If previus solution exists, return dictionary of initial values for the decision variables for assignment of
        aircraft to slots.
        '''
        if pd.isnull(self.simulation.scheduler_maintenance) == 0:
            aircraft_slot_in_common = [tup for tup in self.model.set_aircraft_slot
                                       if tup in self.simulation.scheduler_maintenance.model.set_aircraft_slot]
            aircraft_slot_value = [round(self.simulation.scheduler_maintenance.model.dv_aircraft_slot[ac, sl]())
                                   for (ac, sl) in aircraft_slot_in_common]
            aircraft_slot_dict = dict(zip(aircraft_slot_in_common, aircraft_slot_value))
        else:
            aircraft_slot_dict = {}
        return aircraft_slot_dict


    # =================================================================================#
    # OBJECTIVE
    # =================================================================================#
    def __add_objective(self):
        '''
        Add objective function to the model. The objective has three components:
            - UNASSIGNMENT OF TASK: If a task remains unassigned, and it goes due within the postponement penalty
                                    window, then assign a penalty to it
            - TASK-SLOT ASSIGNMENT: Requirements and DDs respectively follow a decreasing and increasing linear cost
                                    function for their assignment to a slot, with respect to their due date.
            - ACTIVATION OF A SLOT: Using a slot causes the use of ground time, so the activation of a slot causes a
                                    cost in the objective function, proportionally to its duration.
            - AC-SL ASSIGNMENT CHANGE: If the aircraft-slot assignment is changed before the next tail assignment is
                                        called, the changes are penalized.
            - AIRCRAFT CLEAN: Penalization if the aircraft is not clean for a number of days equal to the clean
                                target when it finishes a slots
            - SLOT WORK PACKAGE ANTICIPATION: Penalization of including in a work package one or more tasks that go
                                due a number of days lower than the slot anticipation target after the end of the slot
        '''

        # Task is unassigned
        obj_unassign_task = sum(self.__find_weight_unassign_task(ts) * self.model.dv_task_unassign[ts]
                                for ts in self.model.set_tasks if self.__task_penalized_unassignment(ts) == True)

        # Task is assigned to specific slot
        obj_task_allocation = sum(self.__find_weight_allocate_task(ts, sl) * self.model.dv_task_slot[ts, sl]
                                  for (ts, sl) in self.model.set_task_slot_full)

        # Slot activation
        obj_slot_activation = sum(self.__find_weight_slot_activation(sl) * self.model.dv_aircraft_slot[ac, sl]
                                  for (ac, sl) in self.model.set_aircraft_slot)

        # Aircraft-slot assignment change
        obj_aircraft_slot_assignment_change = sum(G.PENALTY_AIRCRAFT_SLOT_ASSIGNMENT_CHANGE \
                                                  * self.model.dv_change_aircraft_slot_assignment[ac]
                                                  for ac in self.model.set_aircraft_orig)

        # Aircraft clean
        # if G.CONSTR_CLEAN == 1:
        if self.simulation.scenario['MAS_constr_clean'] == 1:
            obj_aircraft_clean_slot = sum(G.PENALTY_AIRCRAFT_CLEAN
                                          * self.model.dv_aircraft_clean_slot[ac,sl]
                                          for (ac, sl) in self.model.set_aircraft_slot)
        else:
            obj_aircraft_clean_slot = 0

        # Slot workpackage anticipation
        # if G.CONSTR_WP_ANTICIPATION == 1:
        if self.simulation.scenario['MAS_constr_wp_anticipation'] == 1:
            obj_slot_anticipation = sum(G.PENALTY_SLOT_ANTICIPATION
                                        * self.model.dv_slot_anticipation[sl]
                                        for sl in self.model.set_slots)
        else:
            obj_slot_anticipation = 0

        # Final objective
        self.model.obj = pyo.Objective(expr=obj_unassign_task + obj_task_allocation + obj_slot_activation\
                                            + obj_aircraft_slot_assignment_change \
                                            + obj_aircraft_clean_slot + obj_slot_anticipation)



    def __find_weight_unassign_task(self, task):
        ''' Return the weight associated to leave a task unassigned'''
        # The requirements that do not have a penalty are excluded from the considered set of requirements.
        if task.type == 'REQUIREMENT':
            weight = G.PENALTY_UNASSIGN_RECURRING
        elif task.type == 'ADHOC' or task.type == 'MEL' or task.type == 'NSRE':
            weight = G.PENALTY_UNASSIGN_DD
        else:
            raise Exception('Task type not supported')
        return weight


    def __find_weight_allocate_task(self, task, slot):
        ''' Select a function to find the weight of allocating a task to a slot, based on the type of the task'''
        if slot.remarks == 'LM':
            weight = self.__find_weight_allocate_lm(task, slot)
        elif task.type == 'REQUIREMENT' and slot.remarks!='LM':
            weight = self.__find_weight_allocate_requirement(task, slot.dateStart_final)
        elif (task.type == 'ADHOC' or task.type == 'MEL' or task.type == 'NSRE') and slot.remarks!='LM':
            weight = self.__find_weight_allocate_deferred_defect(task, slot.dateStart_final)
        else:
            raise Exception('Task type not supported')
        return weight

    def __find_weight_allocate_lm(self, task, slot):
        ''' Find the weight of allocating a task to a line maintenance slot'''
        # Weights are found for the start and end date of the slot
        weight_days = [slot.dateStart_final, slot.dateEnd_final]
        # Middle day computed if at least five days between start and end of line maintenance slot
        slot_duration_days = self.__compute_days_between_dates(slot.dateStart_final, slot.dateEnd_final)
        if slot_duration_days >= 5:
            middle_date = slot.dateStart_final + timedelta(days=round(slot_duration_days/2))
            weight_days.append(middle_date)

        # Compute weights for the chosen dates
        weights = []
        for weight_day in weight_days:
            if task.type == 'REQUIREMENT':
                weight = self.__find_weight_allocate_requirement(task, weight_day)
            elif task.type == 'ADHOC' or task.type == 'MEL' or task.type == 'NSRE':
                weight = self.__find_weight_allocate_deferred_defect(task, weight_day)
            else:
                raise Exception('Task type not supported')
            weights.append(weight)

        # Choose the minimum found weight
        weight_min = min(weights)

        return weight_min


    def __find_weight_allocate_requirement(self, task, allocation_date):
        '''
        Given a requirement and a slot, determine the weight of the assignment considering lost interval and health.
        '''

        # If no anticipation requested
        if G.LI_HEALTH_ORIENTED == False:
            weight = G.LI_SLOPE_REQ_LOST_INTERVAL * self.__compute_days_between_dates(allocation_date, task.dateDue)

        # If there is a preference for anticipating tasks
        elif G.LI_HEALTH_ORIENTED == True:
            health = self.__compute_health_assignment(task, allocation_date)
            # If health greater than health target, consider lost interval
            if health >= G.LI_PREFERRED_ANTICIPATION:
                weight = G.LI_SLOPE_REQ_LOST_INTERVAL * (health - G.LI_PREFERRED_ANTICIPATION)
            # If health smaller that health target give penalty
            else:
                weight = G.LI_SLOPE_HEALTH * (G.LI_PREFERRED_ANTICIPATION - health)

        else:
            raise Exception('Config LI_health_oriented not supported')

        return weight


    def __find_weight_allocate_deferred_defect(self, task, allocation_date):
        '''
        Given a deferred defect and the date of allocation, determine the weight of the assignment, facilitating the
        assignment as early as possible.
        '''
        if G.LI_HEALTH_ORIENTED == False:
            weight = G.LI_SLOPE_FAULTS * self.__compute_days_between_dates(allocation_date, task.dateDue)

        elif G.LI_HEALTH_ORIENTED == True:
            # Find health
            health = self.__compute_health_assignment(task, allocation_date)
            # If health greater than target health, weight increasing with time, but slowly
            if health >= G.LI_PREFERRED_ANTICIPATION:
                weight = G.LI_SLOPE_FAULTS * (health - G.LI_PREFERRED_ANTICIPATION)
            # If health smaller that health target give penalty
            else:
                weight = G.LI_SLOPE_HEALTH * (G.LI_PREFERRED_ANTICIPATION - health)
        else:
            raise Exception('Config LI_health_oriented not supported')

        # Make sure the weight is not smaller than the minimum allowed
        weight = max(weight, G.LI_MIN_FAULTS)
        return weight


    def __compute_health_assignment(self, task, allocation_date):
        '''
        Given a task and a slot, returns the health determined by that assignment
        '''
        health = self.__compute_days_between_dates(allocation_date, task.dateDue)
        if health < 0:
            health = 0
        return health

    @staticmethod
    def __compute_days_between_dates(dt1, dt2):
        '''
        Given two datetime objects, compute difference of days between the two (dt2-dt1)
        '''
        return (dt2.date() - dt1.date()).days

    @staticmethod
    def __find_weight_slot_activation(slot):
        '''
        Given a slot, return the cost of activating it:
         - A-checks: fixed negative value
         - TO slot: as a function of the required ground time and location of the slot
        '''

        # A-checks
        if slot.remarks == 'A':
            weight_slot_activation = G.PENALTY_SLOT_A_CHECK
        # TO slots
        else:
            # Weight given by ground time of slot
            weight_slot_ground_time = G.PENALTY_SLOT_GROUND_TIME * slot.duration_init.total_seconds()/3600

            # Weight given by location of slot
            if slot.location == 'H':
                weight_slot_location = G.PENALTY_SLOT_HANGAR
            else:
                weight_slot_location = G.PENALTY_SLOT_PLATFORM

            # Total weight
            weight_slot_activation = weight_slot_ground_time + weight_slot_location

        return weight_slot_activation


    # =================================================================================#
    # CONSTRAINTS
    # =================================================================================#
    def __add_constraints(self):
        '''
        Add constraints to the model
        '''
        mid_time = timeit.default_timer()
        self.__constraint_cover_task()
        # print('\tConstr assign task', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__constraint_one_aircraft_per_slot()
        # print('\tConstr max assignment slot ', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__constraint_task_aircraft_same_slot()
        # print('\tConstr task ac same slot', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__constr_no_empty_slot()
        # print('\tConstr no empty slot', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__constr_max_labor()
        # print('\tConstr max labor', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__constr_one_slot_per_cycle()
        # print('\tConstr one slot per cycle', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__constr_slot_assignment_change()
        # print('\tConstr assignment change', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        # if G.CONSTR_CLEAN == 1:
        if self.simulation.scenario['MAS_constr_clean'] == 1:
            self.__constr_aircraft_clean()
        # print('\tConstr aircraft_clean', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        # if G.CONSTR_WP_ANTICIPATION == 1:
        if self.simulation.scenario['MAS_constr_wp_anticipation'] == 1:
            self.__constr_slot_anticipation()
        # print('\tConstr work package anticipation', timeit.default_timer() - mid_time)

    def __constraint_cover_task(self):
        '''
        Constraint 1: Task must be assigned only to one slot, or unassigned
        '''
        self.model.constr_cover_task = pyo.ConstraintList()
        for task in self.model.set_tasks:
            # Assignment of task to a slot
            expr = sum(self.model.dv_task_slot[ts,sl] for (ts, sl) in self.model.set_task_slot_full if ts==task)
            # Unassignment of task
            expr = expr + self.model.dv_task_unassign[task]
            # Add constraint to list of constraints
            self.model.constr_cover_task.add(expr == 1)


    def __constraint_one_aircraft_per_slot(self):
        '''
        Constraint 2: Maximum one registration can be assigned to a slot
        '''
        self.model.constr_one_aircraft_per_slot = pyo.ConstraintList()
        for slot in self.model.set_slots:
            expr = sum(self.model.dv_aircraft_slot[ac,sl] for (ac, sl) in self.model.set_aircraft_slot if sl==slot)
            if isinstance(expr, int) == 0:
                self.model.constr_one_aircraft_per_slot.add(expr <= 1)


    def __constraint_task_aircraft_same_slot(self):
        '''
        Constraint 3: A task can be assigned to a slot only if that slot is assigned to the same aircraft
        '''
        self.model.constr_task_aircraft_same_slot = pyo.ConstraintList()
        for (aircraft, slot) in self.model.set_aircraft_slot:
            expr = sum(self.model.dv_task_slot[ts,sl]
                       for (ts,sl) in self.model.set_task_slot
                       if sl==slot
                       and ts.aircraft == aircraft)
            self.model.constr_task_aircraft_same_slot.add(expr <= self.model.dv_aircraft_slot[aircraft,slot] * G.M)


    def __constr_no_empty_slot(self):
        '''
        Constraint 4: Ensure that slots (excluded LM and A checks) are not assigned to aircraft if no task is
                    assigned to slot. Note that although the activation of a slot is penalized within the objective,
                    the MIP gap can lead to solutions where a slot is activated without any task assigned to it.
        '''
        self.model.constr_no_empty_slot = pyo.ConstraintList()
        for slot in self.model.set_slots_to:
            expr_left = sum(self.model.dv_aircraft_slot[ac,sl] for (ac,sl) in self.model.set_aircraft_slot
                            if sl==slot)
            expr_right = sum(self.model.dv_task_slot[ts,sl] for (ts,sl) in self.model.set_task_slot
                             if sl==slot)
            if isinstance(expr_left, int) == 0:
                self.model.constr_no_empty_slot.add(expr_left <= expr_right)


    def __constr_max_labor(self):
        '''
        Constraint 5: The total labor that is allocated to a slot cannot exceed the slot total labor
        '''
        self.model.constr_max_labor = pyo.ConstraintList()
        for slot in self.model.set_slots_full:
            expr = sum(ts.laborEst.total_seconds()/3600 * self.model.dv_task_slot[ts,sl]
                       for (ts, sl) in self.model.set_task_slot_full if sl==slot)
            if isinstance(expr,int)==0:
                self.model.constr_max_labor.add(expr <= slot.laborMax.total_seconds()/3600)

    def __constr_one_slot_per_cycle(self):
        '''
        Constraint 6: Each aircraft can be assigned to maximum one slot per cycle. Necessary to avoid assignment to
        two slots which are overlapping. Note that the functio must be changed if the calls of the recovery module
        do not coincide with a cycle. This is because here it is assumed that all the slots within a cycle are
        included within the set of the considered set of slots.
        '''
        self.model.constr_one_slot_per_cycle = pyo.ConstraintList()
        # Find current cycles
        cycles = list(set([sl.cycle for sl in self.model.set_slots]))
        for cycle in cycles:
            for aircraft in self.model.set_aircraft:
                expr = sum(self.model.dv_aircraft_slot[ac,sl] for (ac, sl) in self.model.set_aircraft_slot
                           if ac==aircraft and sl.cycle == cycle)
                if isinstance(expr, int) == 0:
                    self.model.constr_one_slot_per_cycle.add(expr <= 1)


    def __constr_slot_assignment_change(self):
        '''
        Constraint 7: Soft constraint that penalizes the change of aircraft-slot assignment for slots of the starting
        week, with respect to previous schedule
        '''
        self.model.constr_slot_assignment_change = pyo.ConstraintList()
        for aircraft in self.model.set_aircraft_orig:
            # Sum the decision variables of the original aircraft-slot assignment
            expr_left = sum(self.model.dv_aircraft_slot[ac,sl]
                            for (ac,sl) in self.model.set_aircraft_slot_orig
                            if ac == aircraft and (ac,sl) in self.model.set_aircraft_slot)
            # Find slots that should remain fixed for the aircraft
            slots_to_fix = [sl for sl in self.model.set_slots
                            if (aircraft, sl) in self.model.set_aircraft_slot_orig]
            expr_right = len(slots_to_fix) - self.model.dv_change_aircraft_slot_assignment[aircraft]
            # Add constraint
            self.model.constr_slot_assignment_change.add(expr_left >= expr_right)

    def __constr_aircraft_clean(self):
        '''
        Constraint 8: Soft constraint that penalizes the use of a slot without having the aircraft clean for a target
        number of days afterwards
        '''
        self.model.constr_aircraft_clean = pyo.ConstraintList()
        # mid_time_loop = timeit.default_timer()
        for (aircraft,slot) in self.model.set_aircraft_slot:
            # mid_time = timeit.default_timer()
            tasks_clean = [(ts, sl) for (ts, sl) in self.model.set_task_slot_clean
                           if sl==slot and ts.aircraft==aircraft]
            # print('Time tasks_clean list', timeit.default_timer() - mid_time)

            # If no task found, continue
            if tasks_clean == []:
                continue
            # mid_time = timeit.default_timer()
            sum_tasks_assigned = sum(self.model.dv_task_slot[ts,slh] for (ts,slh,sl) in self.model.set_clean
                                     if sl==slot and (ts,sl) in tasks_clean)
            # print('Time sum tasks assigned', timeit.default_timer() - mid_time)

            # mid_time = timeit.default_timer()
            expr_left = self.model.dv_aircraft_slot[aircraft,slot] - 1/len(tasks_clean) * sum_tasks_assigned
            # print('Time expr left', timeit.default_timer() - mid_time)

            # mid_time = timeit.default_timer()
            self.model.constr_aircraft_clean.add(expr_left <= self.model.dv_aircraft_clean_slot[aircraft,slot])
            # print('Time add constraint', timeit.default_timer() - mid_time)
            # print('Time (aircraft_slot for iteration)', timeit.default_timer() - mid_time_loop, '\n')


    def __constr_slot_anticipation(self):
        '''
        Constraint 9: Soft constraint that penalizes the assignment of a task to a slot when the slot execution is
        too close to the task's due date.
        '''
        self.model.constr_slot_anticipation = pyo.ConstraintList()
        for slot in self.model.set_slots:
            tasks_anticip = [(ts,sl) for (ts,sl) in self.model.set_task_slot_anticipation if sl==slot]
            if tasks_anticip == []:
                continue
            expr_left = 1/len(tasks_anticip)* sum(self.model.dv_task_slot[ts, sl] for (ts,sl) in tasks_anticip)
            self.model.constr_slot_anticipation.add(expr_left <= self.model.dv_slot_anticipation[slot])


    # =================================================================================#
    # SOLUTION
    # =================================================================================#
    def solve(self, solver=None):
        start_time = timeit.default_timer()
        if solver == None:
            solver = self.__find_solver()
        solver = pyo.SolverFactory(solver)
        # Specify allowed gap
        solver.options['mipgap'] = G.MAINTENANCE_SCHED_ALLOWED_GAP
        if getattr(G, 'SOLVER_THREADS', None) is not None:
            solver.options['threads'] = G.SOLVER_THREADS
        results = solver.solve(self.model, tee=False)

        self.time_solve = timeit.default_timer() - start_time
        log_info('\n\n##### MAINTENANCE SCHEDULE #####')
        log_info('Time initialization:', self.time_initialization, '\nTime solution:', self.time_solve)

        # Print solver status # TODO Add if solver status should be checked
        if (results.solver.status == SolverStatus.ok) and (
                results.solver.termination_condition == TerminationCondition.optimal):
            log_info("Found a feasible and optimal solution")
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            log_error("WARNING: SOLUTION INFEASIBLE")
            # # Log info on infeasible model
            # logging.basicConfig(filename='maintenance_infeasible.log', force=True,level=logging.INFO,encoding='utf-8')
            # log_infeasible_constraints(self.model, log_expression = True, log_variables = True)
            # TODO uncomment for saving infeasibility log
        else:
            # something else is wrong
            log_error(str(results.solver))

    @staticmethod
    def __find_solver():
        if isinstance(G.SOLVER,str):
            solver = G.SOLVER
        elif G.SOLVER==0:
            solver = 'cbc'
        elif G.SOLVER==1:
            solver = 'gurobi_direct'
        else:
            raise Exception('Optimization solver not supported')
        return solver


    def results(self):
        for task in self.model.set_tasks:
            slot = [sl for (ts,sl) in self.model.set_task_slot_full
                    if ts==task and self.model.dv_task_slot[(task, sl)]()==1 ]
            # if len(slot) == 0 and task.unassign_type == 'unassigned':
            #     print(task.aircraft.id + ' Task'+ task.id + ' ' + task.unassign_type)
            # elif len(slot) == 0 and task.unassign_type == 'postponed':
            #     print(task.aircraft.id + ' Task'+ task.id + ' ' + task.unassign_type)
            if len(slot) == 0:
                print(task.aircraft.id + ' Task' + task.id + ' unassigned or postponed')
            elif len(slot)== 1:
                print(task.aircraft.id + ' Task'+ task.id + ' Slot '+ slot[0].id)
            else:
                print('more than one assignment')



    # =================================================================================#
    # SOLUTION CHECK
    # =================================================================================#

    def check_results(self):
        '''
        Check that all constraints are fulfilled
        '''
        # Constraint 1: Group must be assigned only to one slot, or unassigned
        for task in self.model.set_tasks:
            # Find slots to which task is assigned
            slots = [sl for (ts, sl) in self.model.set_task_slot_full
                    if ts==task and round(self.model.dv_task_slot[(ts, sl)]()) == 1]

            # Check if task unassigned
            if round(self.model.dv_task_unassign[task]()) == 1:
                unassign = 1
            else:
                unassign = 0

            # Check condition
            total_assignments = len(slots) + unassign
            if total_assignments == 0:
                raise Exception('Task never assigned')
            elif total_assignments > 1:
                raise Exception('Task assigned more than once')


        for slot in self.model.set_slots_to:
            # Constraint 2: Maximum one registration can be assigned to a slot
            # Find aircraft to which slot is assigned
            aircraft = [ac for (ac, sl) in self.model.set_aircraft_slot
                        if sl == slot and round(self.model.dv_aircraft_slot[ac,sl]())==1]
            if len(aircraft) > 1:
                raise Exception('More than one registration is assigned to a slot')

            # Constraint 3: Group can be assigned to a slot only if that slot is assigned to the same aircraft
            # Find tasks assigned to slot
            tasks = [ts for (ts, sl) in self.model.set_task_slot
                          if sl==slot and round(self.model.dv_task_slot[(ts, slot)]()) == 1]

            # Find aircraft of tasks assigned to slot
            task_ac = [ts.aircraft for ts in tasks]
            task_ac = list(set(task_ac))
            if G.SLOTS_ASSIGNMENT_ORDER != 1 \
                and ((aircraft==[] and tasks!=[]) or (aircraft!=[] and tasks==[]) \
                or (aircraft!=[] and tasks!=[] and task_ac[0] != aircraft[0])):
                breakpoint()
                raise Exception('Group not assigned to slot of correct registration')

            # Constraint 4: Maximum labor allocated to slot
            # Find labor hours of tasks assigned to slot
            tasks_labor = sum([ts.laborEst.total_seconds() / 3600 for ts in tasks])
            if round(tasks_labor,2) > slot.laborMax.total_seconds()/3600:
                raise Exception('Too much labor is allocated to a slot')






class MaintenanceScheduler:
    def __init__(
            self,
            simulation
                ):
        self.simulation = simulation
        self.name = 'schedule_maintenance'+self.simulation.now.strftime('%d/%m/%Y')

        # Init time
        start_time = timeit.default_timer()

        # Init model
        self.model = pyo.ConcreteModel(name='schedule_maintenance')

        mid_time = timeit.default_timer()
        self.__add_sets()
        # print('Time add sets', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__add_decision_variables()
        # print('Time add decision variables', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__add_constraints()
        # print('Time add constraints', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__add_objective()
        # print('Time add objective', timeit.default_timer() - mid_time)

        self.time_initialization = timeit.default_timer() - start_time

    # =================================================================================#
    # SETS
    # =================================================================================#
    def __add_sets(self):
        # General sets
        self.model.set_aircraft = pyo.Set(initialize=self.simulation.aircraft)
        self.model.set_slots = pyo.Set(initialize=self.__initialize_slots_to_and_a())
        self.model.set_slots_to = pyo.Set(initialize=self.__initialize_slots_to())
        self.model.set_slots_lm = pyo.Set(initialize=self.__initialize_slots_lm())
        self.model.set_slots_full = self.model.set_slots | self.model.set_slots_lm
        self.model.set_tasks = pyo.Set(initialize=self.__initialize_set_tasks())

        # Reduced combination sets
        self.model.set_aircraft_slot = pyo.Set(initialize=self.__inialize_set_aircraft_slot(), dimen=2)
        self.model.set_aircraft_slot_orig = pyo.Set(initialize=self.__initialize_set_aircraft_slot_orig(), dimen=2)
        self.model.set_aircraft_orig = pyo.Set(initialize=self.__initialize_set_aircraft_orig())
        self.model.set_task_slot = pyo.Set(initialize=self.__initialize_set_task_slot(), dimen=2)
        self.model.set_task_slot_lm = pyo.Set(initialize=self.__initialize_set_task_slot_lm(), dimen=2)
        self.model.set_task_slot_full = self.model.set_task_slot | self.model.set_task_slot_lm

    def __initialize_slots_to_and_a(self):
        ''' Return slots that are not line maintenance slots'''
        return [sl
                for sl in self.simulation.slots_scheduling
                if sl.remarks != 'LM']

    def __initialize_slots_to(self):
        '''Return list of TO slots'''
        return[sl
               for sl in self.model.set_slots
               if sl.remarks == 'TO']

    def __initialize_slots_lm(self):
        ''' Return line maintenance slots'''
        return [sl
                for sl in self.simulation.slots_scheduling
                if sl.remarks == 'LM']

    def __task_penalized_unassignment(self, task):
        '''
        Return True if the unassignment of a task should be penalized, False otherwise
        the final day (included) for which a task is assigned a penalty is computed as:
            end of the scheduling window + the number of days specified by G.DAYS_N_FOR_PENALTY
        '''
        date_until_penalty = self.simulation.maintenance_scheduling_window_end + timedelta(days=G.DAYS_N_FOR_PENALTY)
        return task.dateDue <= date_until_penalty

    def __initialize_set_tasks(self):
        ''' Consider all open tasks, apart from requirements going due after the penalty window. These requirements
        would be postponed anyways to a later opportunity. '''
        return [ts
                for ts in self.simulation.tasks_open
                if ts not in self.simulation.tasks_in_progress
                and (ts.type !='REQUIREMENT'
                     or (ts.type =='REQUIREMENT' and self.__task_penalized_unassignment(ts)))]


    def __inialize_set_aircraft_slot(self):
        ''' In general, an aircraft can be assigned to a slot if there is a match in subtype. However, within the
        fixed assignment window, the assigned slots are fixed to an assigne registration. '''
        return [(ac, sl)
                for ac in self.model.set_aircraft
                for sl in self.model.set_slots
                if self.__find_compatibility_ac_slot(ac, sl) == True]

    def __find_compatibility_ac_slot(self, aircraft, slot):
        '''
        The following cases are considered:
            - Drumbeat (G.SLOTS_ASSIGNMENT_ORDER==1): assign to designated aircraft
            - Fixed subtype (G.SLOTS_ASSIGNMENT_ORDER==2): compatibility is same subtype
            - Fixed subtype or free assignemnt (G.SLOTS_ASSIGNMENT_ORDER==0 or 2) : fix assignment within fixed window
        '''
        # TODO if A-checks shoul have a fixed assignment, fix it here

        # If slot overlaps with AOG of aircraft, do not assign it
        aog_slot = next((sl for sl in aircraft.slots if sl.remarks == 'AG'), None)
        if aog_slot!=None:
            if slot.dateStart_final - slot.towing_time <= aog_slot.dateEnd_final + aog_slot.towing_time \
                and slot.dateEnd_final + slot.towing_time >= aog_slot.dateStart_final - aog_slot.towing_time:
                return False

        # Drumbeat assignment or fixed subtype
        match(G.SLOTS_ASSIGNMENT_ORDER):
            # DRUMBEAT
            case 1:
                if slot.aircraft_pre_assigned == aircraft:
                   return True
                else:
                    return False

            # FIXED SUBTYPE
            case 2:
                if aircraft.subtype.IATA != slot.subtype:
                    return False

        # FIXED SCHEDULING WINDOW (do not consider if first time running optimization)
        if G.SLOTS_ASSIGNMENT_FIX > 0 and pd.isnull(self.simulation.scheduler_maintenance)==False:
            # End of fixed scheduling window
            fixed_schedule_end = self.simulation.now + timedelta(days=G.SLOTS_ASSIGNMENT_FIX)
            if slot.dateStart_init <= fixed_schedule_end and slot.aircraft != aircraft:
                return False

        # If no incompatibility found, return True
        return True

    def __initialize_set_aircraft_slot_orig(self):
        ''' Set of previous assignment of aircraft to slots for the starting week'''
        # Window to fix is interval with which tail assignment is called
        window_fix_end = self.simulation.now + timedelta(days=G.TAIL_ASSIGNMENT_INTERVAL) \
                         + timedelta(days=G.TAIL_ASSIGNMENT_FIX)
        # Find set of aircraft-slots to fix
        return [(sl.aircraft, sl)
                for sl in self.model.set_slots
                if sl.dateStart_final <= window_fix_end
                and pd.isnull(sl.aircraft) == 0]

    def __initialize_set_aircraft_orig(self):
        ''' Return list of aircraft for which at least one slot assignment should be preferably fixed from previous
        runs of the scheduler'''
        ac_fixed = list(set([ac for (ac,sl) in self.model.set_aircraft_slot_orig]))
        return ac_fixed

    def __initialize_set_task_slot(self):
        return [(ts, sl)
                for ts in self.model.set_tasks
                for sl in self.model.set_slots
                if self.__find_task_compatibility_task_slot(ts, sl) == True]

    def __initialize_set_task_slot_lm(self):
        return [(ts, sl)
                for ts in self.model.set_tasks
                for sl in self.model.set_slots_lm
                if self.__find_task_compatibility_task_slot(ts, sl) == True]

    def __find_task_compatibility_task_slot(self, task, slot):
        '''
        A task can be executed in a slot given the following conditions:
             - AIRCRAFT: the aircraft of the task must be compatible with the slot based on aircraft type or specific
                            registration assignment.
             - DUE DATE: the task can be scheduled only before its due date
             - READY DATE: the task can only be executed after its ready date is reached
             - LOCATION: hangar (H) tasks can only be executed in the hangar, while platform (P) tasks can be
                            executed both in the hangar and on the platform
             - DURATION: the duration of the task cannot exceed the duration of the slot
             - LABOR: the total labor needed for the task must not exceed the maximum labor that can be scheduled in
                        a slot
        '''

        # CONDITION AIRCRAFT
        if slot.remarks != 'LM' and (task.aircraft, slot) not in self.model.set_aircraft_slot:
            return False
        if slot.remarks == 'LM' and task.aircraft != slot.aircraft:
            return False

        # CONDITION DUE DATE
        if task.dateDue < slot.dateStart_final:
            return False

        # CONDITION READY DATE
        if task.dateReady > slot.dateEnd_final:
            return False

        # CONDITION LOCATION
        if task.workType == 'H' and slot.location == 'P':
            return False

        # CONDITION DURATION
        if task.durationEst > slot.duration_init:
            return False

        # CONDITION LABOR
        if task.laborEst > slot.laborMax_per_task:
            return False

        # If no incompatibility found, task and slot are compatible
        return True


    # =================================================================================#
    # DECISION VARIABLES
    # =================================================================================#
    def __add_decision_variables(self):
        '''
        Add decision variables to the model
        '''
        # Assign task to slot
        self.model.dv_task_slot = pyo.Var(self.model.set_task_slot_full, domain=pyo.Binary,
                                          initialize=self.__initialize_dv_task_slot())
        # Unassign task
        self.model.dv_task_unassign = pyo.Var(self.model.set_tasks, domain=pyo.Binary,
                                          initialize=self.__initialize_dv_unassign())
        # Assign aircraft to slot
        self.model.dv_aircraft_slot = pyo.Var(self.model.set_aircraft_slot, domain=pyo.Binary,
                                              initialize=self.__init_dv_aircaft_slot())

        # Change of aircraft-slot assignment
        self.model.dv_change_aircraft_slot_assignment = pyo.Var(self.model.set_aircraft_orig,
                                                                domain=pyo.NonNegativeIntegers, initialize=0)


    def __initialize_dv_task_slot(self):
        '''
        If previus solution exists, return dictionary of initial values for the decision variables for assignment of
        tasks to slots.
        '''
        if self.simulation.scheduler_maintenance != None:
            # Find tuples (group, slot) in common between current and previous model
            task_slot_in_common = [tup for tup in self.model.set_task_slot_full
                                   if tup in self.simulation.scheduler_maintenance.model.set_task_slot_full]
            # Find values corresponding to tuples
            task_slot_values = [round(self.simulation.scheduler_maintenance.model.dv_task_slot[ts, sl]())
                                for (ts, sl) in task_slot_in_common]
            # Generate initialization dictionary
            task_slot_dict = dict(zip(task_slot_in_common, task_slot_values))
        else:
            task_slot_dict = {}

        return task_slot_dict

    def __initialize_dv_unassign(self):
        '''
        If previus solution exists, return dictionary of initial values for the decision variables for task
        unassignment.
        '''
        if pd.isnull(self.simulation.scheduler_maintenance)==0:
            # Find tasks in commond with previous model
            tasks_in_common = [ts for ts in self.model.set_tasks
                               if ts in self.simulation.scheduler_maintenance.model.set_tasks]
            tasks_unassign_value = [round(self.simulation.scheduler_maintenance.model.dv_task_unassign[ts]())
                                    for ts in tasks_in_common]
            tasks_unassing_dict = dict(zip(tasks_in_common, tasks_unassign_value))
        else:
            tasks_unassing_dict = {}

        return tasks_unassing_dict

    def __init_dv_aircaft_slot(self):
        '''
        If previus solution exists, return dictionary of initial values for the decision variables for assignment of
        aircraft to slots.
        '''
        if pd.isnull(self.simulation.scheduler_maintenance) == 0:
            aircraft_slot_in_common = [tup for tup in self.model.set_aircraft_slot
                                       if tup in self.simulation.scheduler_maintenance.model.set_aircraft_slot]
            aircraft_slot_value = [round(self.simulation.scheduler_maintenance.model.dv_aircraft_slot[ac, sl]())
                                   for (ac, sl) in aircraft_slot_in_common]
            aircraft_slot_dict = dict(zip(aircraft_slot_in_common, aircraft_slot_value))
        else:
            aircraft_slot_dict = {}
        return aircraft_slot_dict


    # =================================================================================#
    # OBJECTIVE
    # =================================================================================#
    def __add_objective(self):
        '''
        Add objective function to the model. The objective has three components:
            - UNASSIGNMENT OF TASK: If a task remains unassigned, and it goes due within the postponement penalty
                                    window, then assign a penalty to it
            - TASK-SLOT ASSIGNMENT: Requirements and DDs respectively follow a decreasing and increasing linear cost
                                    function for their assignment to a slot, with respect to their due date.
            - ACTIVATION OF A SLOT: Using a slot causes the use of ground time, so the activation of a slot causes a
                                    cost in the objective function, proportionally to its duration.
            - AC-SL ASSIGNMENT CHANGE: If the aircraft-slot assignment is changed before the next tail assignment is
                                        called, the changes are penalized.
        '''

        # Task is unassigned
        obj_unassign_task = sum(self.__find_weight_unassign_task(ts) * self.model.dv_task_unassign[ts]
                                for ts in self.model.set_tasks if self.__task_penalized_unassignment(ts) == True)

        # Task is assigned to specific slot
        obj_task_allocation = sum(self.__find_weight_allocate_task(ts, sl) * self.model.dv_task_slot[ts, sl]
                                  for (ts, sl) in self.model.set_task_slot_full)

        # Slot activation
        obj_slot_activation = sum(self.__find_weight_slot_activation(sl) * self.model.dv_aircraft_slot[ac, sl]
                                  for (ac, sl) in self.model.set_aircraft_slot)

        # Aircraft-slot assignment change
        obj_aircraft_slot_assignment_change = sum(G.PENALTY_AIRCRAFT_SLOT_ASSIGNMENT_CHANGE \
                                                  * self.model.dv_change_aircraft_slot_assignment[ac]
                                                  for ac in self.model.set_aircraft_orig)

        # Final objective
        self.model.obj = pyo.Objective(expr=obj_unassign_task + obj_task_allocation + obj_slot_activation\
                                       + obj_aircraft_slot_assignment_change)



    def __find_weight_unassign_task(self, task):
        ''' Return the weight associated to leave a task unassigned'''
        # The requirements that do not have a penalty are excluded from the considered set of requirements.
        if task.type == 'REQUIREMENT':
            weight = G.PENALTY_UNASSIGN_RECURRING
        elif task.type == 'ADHOC' or task.type == 'MEL' or task.type == 'NSRE':
            weight = G.PENALTY_UNASSIGN_DD
        else:
            raise Exception('Task type not supported')
        return weight


    def __find_weight_allocate_task(self, task, slot):
        ''' Select a function to find the weight of allocating a task to a slot, based on the type of the task'''
        if slot.remarks == 'LM':
            weight = self.__find_weight_allocate_lm(task, slot)
        elif task.type == 'REQUIREMENT' and slot.remarks!='LM':
            weight = self.__find_weight_allocate_requirement(task, slot.dateStart_final)
        elif (task.type == 'ADHOC' or task.type == 'MEL' or task.type == 'NSRE') and slot.remarks!='LM':
            weight = self.__find_weight_allocate_deferred_defect(task, slot.dateStart_final)
        else:
            raise Exception('Task type not supported')
        return weight

    def __find_weight_allocate_lm(self, task, slot):
        ''' Find the weight of allocating a task to a line maintenance slot'''
        # Weights are found for the start and end date of the slot
        weight_days = [slot.dateStart_final, slot.dateEnd_final]
        # Middle day computed if at least five days between start and end of line maintenance slot
        slot_duration_days = self.__compute_days_between_dates(slot.dateStart_final, slot.dateEnd_final)
        if slot_duration_days >= 5:
            middle_date = slot.dateStart_final + timedelta(days=round(slot_duration_days/2))
            weight_days.append(middle_date)

        # Compute weights for the chosen dates
        weights = []
        for weight_day in weight_days:
            if task.type == 'REQUIREMENT':
                weight = self.__find_weight_allocate_requirement(task, weight_day)
            elif task.type == 'ADHOC' or task.type == 'MEL' or task.type == 'NSRE':
                weight = self.__find_weight_allocate_deferred_defect(task, weight_day)
            else:
                raise Exception('Task type not supported')
            weights.append(weight)

        # Choose the minimum found weight
        weight_min = min(weights)

        return weight_min


    def __find_weight_allocate_requirement(self, task, allocation_date):
        '''
        Given a requirement and a slot, determine the weight of the assignment considering lost interval and health.
        '''

        # If no anticipation requested
        if G.LI_HEALTH_ORIENTED == False:
            weight = G.LI_SLOPE_REQ_LOST_INTERVAL * self.__compute_days_between_dates(allocation_date, task.dateDue)

        # If there is a preference for anticipating tasks
        elif G.LI_HEALTH_ORIENTED == True:
            health = self.__compute_health_assignment(task, allocation_date)
            # If health greater than health target, consider lost interval
            if health >= G.LI_PREFERRED_ANTICIPATION:
                weight = G.LI_SLOPE_REQ_LOST_INTERVAL * (health - G.LI_PREFERRED_ANTICIPATION)
            # If health smaller that health target give penalty
            else:
                weight = G.LI_SLOPE_HEALTH * (G.LI_PREFERRED_ANTICIPATION - health)

        else:
            raise Exception('Config LI_health_oriented not supported')

        return weight


    def __find_weight_allocate_deferred_defect(self, task, allocation_date):
        '''
        Given a deferred defect and the date of allocation, determine the weight of the assignment, facilitating the
        assignment as early as possible.
        '''
        if G.LI_HEALTH_ORIENTED == False:
            weight = G.LI_SLOPE_FAULTS * self.__compute_days_between_dates(allocation_date, task.dateDue)

        elif G.LI_HEALTH_ORIENTED == True:
            # Find health
            health = self.__compute_health_assignment(task, allocation_date)
            # If health greater than target health, weight increasing with time, but slowly
            if health >= G.LI_PREFERRED_ANTICIPATION:
                weight = G.LI_SLOPE_FAULTS * (health - G.LI_PREFERRED_ANTICIPATION)
            # If health smaller that health target give penalty
            else:
                weight = G.LI_SLOPE_HEALTH * (G.LI_PREFERRED_ANTICIPATION - health)
        else:
            raise Exception('Config LI_health_oriented not supported')

        # Make sure the weight is not smaller than the minimum allowed
        weight = max(weight, G.LI_MIN_FAULTS)
        return weight


    def __compute_health_assignment(self, task, allocation_date):
        '''
        Given a task and a slot, returns the health determined by that assignment
        '''
        health = self.__compute_days_between_dates(allocation_date, task.dateDue)
        if health < 0:
            health = 0
        return health

    @staticmethod
    def __compute_days_between_dates(dt1, dt2):
        '''
        Given two datetime objects, compute difference of days between the two (dt2-dt1)
        '''
        return (dt2.date() - dt1.date()).days

    @staticmethod
    def __find_weight_slot_activation(slot):
        '''
        Given a slot, return the cost of activating it:
         - A-checks: fixed negative value
         - TO slot: as a function of the required ground time and location of the slot
        '''

        # A-checks
        if slot.remarks == 'A':
            weight_slot_activation = G.PENALTY_SLOT_A_CHECK
        # TO slots
        else:
            # Weight given by ground time of slot
            weight_slot_ground_time = G.PENALTY_SLOT_GROUND_TIME * slot.duration_init.total_seconds()/3600

            # Weight given by location of slot
            if slot.location == 'H':
                weight_slot_location = G.PENALTY_SLOT_HANGAR
            else:
                weight_slot_location = G.PENALTY_SLOT_PLATFORM

            # Total weight
            weight_slot_activation = weight_slot_ground_time + weight_slot_location

        return weight_slot_activation


    # =================================================================================#
    # CONSTRAINTS
    # =================================================================================#
    def __add_constraints(self):
        '''
        Add constraints to the model
        '''
        mid_time = timeit.default_timer()
        self.__constraint_cover_task()
        # print('\tConstr assign task', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__constraint_one_aircraft_per_slot()
        # print('\tConstr max assignment slot ', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__constraint_task_aircraft_same_slot()
        # print('\tConstr task ac same slot', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__constr_no_empty_slot()
        # print('\tConstr no empty slot', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__constr_max_labor()
        # print('\tConstr max labor', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__constr_one_slot_per_cycle()
        # print('\tConstr one slot per cycle', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__constr_slot_assignment_change()
        # print('\tConstr assignment change', timeit.default_timer() - mid_time)


    def __constraint_cover_task(self):
        '''
        Constraint 1: Task must be assigned only to one slot, or unassigned
        '''
        self.model.constr_cover_task = pyo.ConstraintList()
        for task in self.model.set_tasks:
            # Assignment of task to a slot
            expr = sum(self.model.dv_task_slot[ts,sl] for (ts, sl) in self.model.set_task_slot_full if ts==task)
            # Unassignment of task
            expr = expr + self.model.dv_task_unassign[task]
            # Add constraint to list of constraints
            self.model.constr_cover_task.add(expr == 1)


    def __constraint_one_aircraft_per_slot(self):
        '''
        Constraint 2: Maximum one registration can be assigned to a slot
        '''
        self.model.constr_one_aircraft_per_slot = pyo.ConstraintList()
        for slot in self.model.set_slots:
            expr = sum(self.model.dv_aircraft_slot[ac,sl] for (ac, sl) in self.model.set_aircraft_slot if sl==slot)
            if isinstance(expr, int) == 0:
                self.model.constr_one_aircraft_per_slot.add(expr <= 1)


    def __constraint_task_aircraft_same_slot(self):
        '''
        Constraint 3: A task can be assigned to a slot only if that slot is assigned to the same aircraft
        '''
        self.model.constr_task_aircraft_same_slot = pyo.ConstraintList()
        for (aircraft, slot) in self.model.set_aircraft_slot:
            expr = sum(self.model.dv_task_slot[ts,sl]
                       for (ts,sl) in self.model.set_task_slot
                       if sl==slot
                       and ts.aircraft == aircraft)
            self.model.constr_task_aircraft_same_slot.add(expr <= self.model.dv_aircraft_slot[aircraft,slot] * G.M)


    def __constr_no_empty_slot(self):
        '''
        Constraint 4: Ensure that slots (excluded LM and A checks) are not assigned to aircraft if no task is
                    assigned to slot. Note that although the activation of a slot is penalized within the objective,
                    the MIP gap can lead to solutions where a slot is activated without any task assigned to it.
        '''
        self.model.constr_no_empty_slot = pyo.ConstraintList()
        for slot in self.model.set_slots_to:
            expr_left = sum(self.model.dv_aircraft_slot[ac,sl] for (ac,sl) in self.model.set_aircraft_slot
                            if sl==slot)
            expr_right = sum(self.model.dv_task_slot[ts,sl] for (ts,sl) in self.model.set_task_slot
                             if sl==slot)
            if isinstance(expr_left, int) == 0:
                self.model.constr_no_empty_slot.add(expr_left <= expr_right)


    def __constr_max_labor(self):
        '''
        Constraint 5: The total labor that is allocated to a slot cannot exceed the slot total labor
        '''
        self.model.constr_max_labor = pyo.ConstraintList()
        for slot in self.model.set_slots_full:
            expr = sum(ts.laborEst.total_seconds()/3600 * self.model.dv_task_slot[ts,sl]
                       for (ts, sl) in self.model.set_task_slot_full if sl==slot)
            if isinstance(expr,int)==0:
                self.model.constr_max_labor.add(expr <= slot.laborMax.total_seconds()/3600)

    def __constr_one_slot_per_cycle(self):
        '''
        Constraint 6: Each aircraft can be assigned to maximum one slot per cycle. Necessary to avoid assignment to
        two slots which are overlapping. Note that the functio must be changed if the calls of the recovery module
        do not coincide with a cycle. This is because here it is assumed that all the slots within a cycle are
        included within the set of the considered set of slots.
        '''
        self.model.constr_one_slot_per_cycle = pyo.ConstraintList()
        # Find current cycles
        cycles = list(set([sl.cycle for sl in self.model.set_slots]))
        for cycle in cycles:
            for aircraft in self.model.set_aircraft:
                expr = sum(self.model.dv_aircraft_slot[ac,sl] for (ac, sl) in self.model.set_aircraft_slot
                           if ac==aircraft and sl.cycle == cycle)
                if isinstance(expr, int) == 0:
                    self.model.constr_one_slot_per_cycle.add(expr <= 1)


    def __constr_slot_assignment_change(self):
        '''
        Constraint 7: Soft constraint that penalizes the change of aircraft-slot assignment for slots of the starting
        week, with respect to previous schedule
        '''
        self.model.constr_slot_assignment_change = pyo.ConstraintList()
        for aircraft in self.model.set_aircraft_orig:
            # Sum the decision variables of the original aircraft-slot assignment
            expr_left = sum(self.model.dv_aircraft_slot[ac,sl]
                            for (ac,sl) in self.model.set_aircraft_slot_orig
                            if ac == aircraft and (ac,sl) in self.model.set_aircraft_slot)
            # Find slots that should remain fixed for the aircraft
            slots_to_fix = [sl for sl in self.model.set_slots
                            if (aircraft, sl) in self.model.set_aircraft_slot_orig]
            expr_right = len(slots_to_fix) - self.model.dv_change_aircraft_slot_assignment[aircraft]
            # Add constraint
            self.model.constr_slot_assignment_change.add(expr_left >= expr_right)


    # =================================================================================#
    # SOLUTION
    # =================================================================================#
    def solve(self, solver=None):
        start_time = timeit.default_timer()
        if solver == None:
            solver = self.__find_solver()
        solver = pyo.SolverFactory(solver)
        # Specify allowed gap
        solver.options['mipgap'] = G.MAINTENANCE_SCHED_ALLOWED_GAP
        if getattr(G, 'SOLVER_THREADS', None) is not None:
            solver.options['threads'] = G.SOLVER_THREADS
        results = solver.solve(self.model, tee=False)

        self.time_solve = timeit.default_timer() - start_time
        log_info('\n\n##### MAINTENANCE SCHEDULE #####')
        log_info('Time initialization:', self.time_initialization, '\nTime solution:', self.time_solve)

        # Print solver status # TODO Add if solver status should be checked
        if (results.solver.status == SolverStatus.ok) and (
                results.solver.termination_condition == TerminationCondition.optimal):
            log_info("Found a feasible and optimal solution")
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            log_error("WARNING: SOLUTION INFEASIBLE")
            # # Log info on infeasible model
            # logging.basicConfig(filename='maintenance_infeasible.log', force=True,level=logging.INFO,encoding='utf-8')
            # log_infeasible_constraints(self.model, log_expression = True, log_variables = True) # TODO must change log
        else:
            # something else is wrong
            log_error(str(results.solver))

    @staticmethod
    def __find_solver():
        if isinstance(G.SOLVER,str):
            solver = G.SOLVER
        elif G.SOLVER==0:
            solver = 'cbc'
        elif G.SOLVER==1:
            solver = 'gurobi_direct'
        else:
            raise Exception('Optimization solver not supported')
        return solver


    def results(self):
        for task in self.model.set_tasks:
            slot = [sl for (ts,sl) in self.model.set_task_slot_full
                    if ts==task and self.model.dv_task_slot[(task, sl)]()==1 ]
            # if len(slot) == 0 and task.unassign_type == 'unassigned':
            #     print(task.aircraft.id + ' Task'+ task.id + ' ' + task.unassign_type)
            # elif len(slot) == 0 and task.unassign_type == 'postponed':
            #     print(task.aircraft.id + ' Task'+ task.id + ' ' + task.unassign_type)
            if len(slot) == 0:
                print(task.aircraft.id + ' Task' + task.id + ' unassigned or postponed')
            elif len(slot)== 1:
                print(task.aircraft.id + ' Task'+ task.id + ' Slot '+ slot[0].id)
            else:
                print('more than one assignment')



    # =================================================================================#
    # SOLUTION CHECK
    # =================================================================================#

    def check_results(self):
        '''
        Check that all constraints are fulfilled
        '''
        # Constraint 1: Group must be assigned only to one slot, or unassigned
        for task in self.model.set_tasks:
            # Find slots to which task is assigned
            slots = [sl for (ts, sl) in self.model.set_task_slot_full
                    if ts==task and round(self.model.dv_task_slot[(ts, sl)]()) == 1]

            # Check if task unassigned
            if round(self.model.dv_task_unassign[task]()) == 1:
                unassign = 1
            else:
                unassign = 0

            # Check condition
            total_assignments = len(slots) + unassign
            if total_assignments == 0:
                raise Exception('Task never assigned')
            elif total_assignments > 1:
                raise Exception('Task assigned more than once')


        for slot in self.model.set_slots_to:
            # Constraint 2: Maximum one registration can be assigned to a slot
            # Find aircraft to which slot is assigned
            aircraft = [ac for (ac, sl) in self.model.set_aircraft_slot
                        if sl == slot and round(self.model.dv_aircraft_slot[ac,sl]())==1]
            if len(aircraft) > 1:
                raise Exception('More than one registration is assigned to a slot')

            # Constraint 3: Group can be assigned to a slot only if that slot is assigned to the same aircraft
            # Find tasks assigned to slot
            tasks = [ts for (ts, sl) in self.model.set_task_slot
                          if sl==slot and round(self.model.dv_task_slot[(ts, slot)]()) == 1]

            # Find aircraft of tasks assigned to slot
            task_ac = [ts.aircraft for ts in tasks]
            task_ac = list(set(task_ac))
            if G.SLOTS_ASSIGNMENT_ORDER != 1 \
                and ((aircraft==[] and tasks!=[]) or (aircraft!=[] and tasks==[]) \
                or (aircraft!=[] and tasks!=[] and task_ac[0] != aircraft[0])):
                breakpoint()
                raise Exception('Group not assigned to slot of correct registration')

            # Constraint 4: Maximum labor allocated to slot
            # Find labor hours of tasks assigned to slot
            tasks_labor = sum([ts.laborEst.total_seconds() / 3600 for ts in tasks])
            if round(tasks_labor,2) > slot.laborMax.total_seconds()/3600:
                raise Exception('Too much labor is allocated to a slot')


def pyomo_maintenance_scheduler(self):
    scheduler_maintenance = MaintenanceScheduler(self)
    # Optimize
    scheduler_maintenance.solve()
    # Print results and check them
    # scheduler_maintenance.results()
    scheduler_maintenance.check_results() # TODO uncomment if results should be checked

    return scheduler_maintenance

def pyomo_maintenance_scheduler_health(self):
    scheduler_maintenance = MaintenanceSchedulerHealth(self)
    # Optimize
    scheduler_maintenance.solve()
    # Print results and check them
    # scheduler_maintenance.results()
    scheduler_maintenance.check_results() # TODO uncomment if results should be checked

    return scheduler_maintenance


# ================================================================================= #
# MAINTENANCE SCHEDULER FROM R-CHECK SIMULATION
# ================================================================================= #
# class MaintenanceScheduler_Rcheck_sim: # TODO uncomment for use. Commented to avoid modifications
#     def __init__(
#             self,
#             simulation
#                 ):
#         self.simulation = simulation
#         self.name = 'schedule_maintenance'+self.simulation.now.strftime('%d/%m/%Y')
#
#         # Init time
#         start_time = timeit.default_timer()
#
#         # Initialize compatibility matrices for model reduction
#         self.matrices = self.__generate_schedule_matrices()
#
#
#         # Init model
#         self.model = pyo.ConcreteModel(name='schedule_maintenance')
#         self.__add_sets()
#         self.__add_variables()
#         self.__add_objective()
#         self.__add_constraints()
#
#         self.time_initialization = timeit.default_timer() - start_time
#
#
#     # =================================================================================#
#     # PRECOMPUTED SCHEDULE MATRICES
#     # =================================================================================#
#     def __generate_schedule_matrices(self):
#         '''
#     Generate matrices for reduction of optimization model.
#     GS: Feasible assignment group-slot.
#         Conditions: - ready date
#                     - due date
#                     - task duration
#                     - task required labor
#     LI: Weights for optimal group-slot assignment
#         Requirements, blocks:   - continuously decreasing or
#                                 - decreasing function while health>=health_target + increasing function afterward
#         DD, ad hoc tasks:   - slowly increasing function while health>=health_target
#                             - fast increasing function afterward
#         '''
#         matrix_task_slot_compatibility = self.__generate_TS_matrix()
#         matrix_assignment_weight = self.__generate_LI_matrix(matrix_task_slot_compatibility)
#         matrices = {'TS': matrix_task_slot_compatibility,
#                     'LI': matrix_assignment_weight}
#         return matrices
#
#     def __generate_TS_matrix(self):
#         ''' Generate matrix of task-slot compatibility '''
#         matrix_TS = pd.DataFrame(0,
#                                  index=[ts.id for ts in self.simulation.tasks_open],
#                                  columns=[sl.id for sl in self.simulation.slots_scheduling])
#         for task in self.simulation.tasks_open:
#             for slot in self.simulation.slots_scheduling:
#                 # CONDITIONS
#                 # Task ready date
#                 condition_ready = task.dateReady == None or task.dateReady <= slot.dateEnd_init
#                 # Due date
#                 condition_due = task.dateDue >= slot.dateStart_init
#                 # Duration
#                 condition_duration = task.durationEst <= slot.duration_init #(slot.dateEnd_final - slot.dateStart_final)
#                 # Total labor
#                 condition_labor = task.laborEst <= slot.laborMax
#                 # If group worktype is #H, group must be executed in the hangar
#                 if task.workType == 'H' and slot.location != 'H':
#                     condition_location = False
#                 else:
#                     condition_location = True
#
#                 if condition_ready and condition_due and condition_duration and condition_labor and condition_location:
#                    matrix_TS.loc[task.id, slot.id] = 1
#
#         return matrix_TS
#
#     def __generate_LI_matrix(self, matrix_TS):
#         ''' Generate matrix of weight task-slot assignment'''
#
#         # Define some functions needed for computation
#         def compute_daysDiff(dt1, dt2):
#             '''
#             Given two datetime, compute difference of days between the two (dt2-dt1)
#             '''
#             # Make sure same timezone used
#             dt1_utc = dt1.astimezone(G.TIMEZONE_UTC)
#             dt2_utc = dt2.astimezone(G.TIMEZONE_UTC)
#             # Find date with no time
#             dt1_day = date(dt1_utc.year, dt1_utc.month, dt1_utc.day)
#             dt2_day = date(dt2_utc.year, dt2_utc.month, dt2_utc.day)
#             # Find days difference
#             diff = (dt2_day - dt1_day).days
#             return diff
#
#         def compute_health_assignment(task, slot):
#             '''
#             Given a group and a slot, returns the health determined by that assignment
#             '''
#             if task.dateDue < slot.dateStart_init:
#                 health = 0
#             else:
#                 health = compute_daysDiff(slot.dateStart_init, task.dateDue)
#             return health
#
#         # Define functions for weight calculations
#         def weight_requirement(task, slot):
#             '''
#             Given a requirement or block and a slot, determines the weight of the assignment
#             considering lost interval and health
#             '''
#             if G.LI_HEALTH_ORIENTED == 1:
#                 # Find health given assignment
#                 health = compute_health_assignment(task, slot)
#                 # If health greater than health target, consider lost interval
#                 if health >= G.LI_PREFERRED_DAY:
#                     weight = G.LI_SLOPE_REQ_LOST_INTERVAL * (health - G.LI_PREFERRED_DAY)
#                 # If health smaller that health target give penalty
#                 else:
#                     weight = G.LI_SLOPE_HEALTH * (G.LI_PREFERRED_DAY - health)
#
#             elif G.LI_HEALTH_ORIENTED == 0:
#                 weight = G.LI_SLOPE_REQ_LOST_INTERVAL * compute_daysDiff(slot.dateStart_init, task.dateDue)
#             else:
#                 raise Exception('Config LI_health_oriented not supported')
#
#             return weight
#
#         def weight_fault(task, slot):
#             '''
#             Given a task and a slot, determines the weight of the assignment
#             facilitating the assignment as earlier as possible
#             '''
#             if G.LI_HEALTH_ORIENTED == 1:
#                 # Find health
#                 health = compute_health_assignment(task, slot)
#                 # If health greater than target health, weight increasing with time, but slowly
#                 if health >= G.LI_PREFERRED_DAY:
#                     weight = G.LI_SLOPE_FAULTS * (health - G.LI_PREFERRED_DAY)
#                 # If health smaller that health target give penalty
#                 else:
#                     weight = G.LI_SLOPE_HEALTH * (G.LI_PREFERRED_DAY - health)
#             elif G.LI_HEALTH_ORIENTED == 0:
#                 weight = G.LI_SLOPE_FAULTS * compute_daysDiff(slot.dateStart_init, task.dateDue)
#             else:
#                 raise Exception('Config LI_health_oriented not supported')
#             # Make sure the weight is not smaller than the minimum allowed
#             weight = max(weight, G.LI_MIN_FAULTS)
#             return weight
#
#
#         # Initialize LI matrix
#         matrix_LI = pd.DataFrame(index=[ts.id for ts in self.simulation.tasks_open],
#                                  columns=[sl.id for sl in self.simulation.slots_scheduling])
#
#         # Find weight associated to each assignment
#         for task in self.simulation.tasks_open:
#             for slot in self.simulation.slots_scheduling:
#                 # Only consider feasible assignments
#                 if matrix_TS.loc[task.id, slot.id] != 0:
#                     if task.requirement != None:
#                         weight = weight_requirement(task, slot)
#                     else:
#                         weight = weight_fault(task, slot)
#
#                     matrix_LI.loc[task.id, slot.id] = weight
#
#         return matrix_LI
#
#     # =================================================================================#
#     # SETS
#     # =================================================================================#
#     def __add_sets(self):
#         # General sets
#         self.model.setTasks = pyo.Set(initialize=self.simulation.tasks_open)
#         self.model.setSlots = pyo.Set(initialize=self.simulation.slots_scheduling)
#         self.model.setAc = pyo.Set(initialize=self.simulation.aircraft)
#
#         # Reduced group-slot feasible assignment set
#         self.model.setTaskSlot = pyo.Set(initialize=self.__assignTaskSlot(), dimen=2)
#         # Reduced groups with unassignment penalty
#         self.model.setTaskPenalty = pyo.Set(initialize=self.__taskPenalty())
#
#     def __assignTaskSlot(self):
#         return [ (task, slot)
#                  for task in self.model.setTasks
#                  for slot in self.model.setSlots
#                  if self.matrices['TS'].loc[task.id, slot.id] == 1 ]
#
#     def __taskPenalty(self):
#         tasks_with_penalty = [task
#                               for task in self.model.setTasks
#                               if task.dateDue <= self.simulation.maintenance_scheduling_window_end + timedelta(days=G.DAYS_N_FOR_PENALTY)]
#         tasks_no_penalty = [task for task in self.model.setTasks if task not in tasks_with_penalty]
#         # Assign attribute to unassigned and postponed tasks
#         for task in tasks_with_penalty:
#             task.unassign_type = 'unassigned'
#         for task in tasks_no_penalty:
#             task.unassign_type = 'postponed'
#
#         return tasks_with_penalty
#
#     # =================================================================================#
#     # DECISION VARIABLES
#     # =================================================================================#
#     def __add_variables(self):
#         '''
#         Add decision variables to the model
#         '''
#         # Assign group to slot
#         self.model.dvTaskSlot = pyo.Var(self.model.setTaskSlot, domain=pyo.Binary, initialize=self.__initDvTaskSlot())
#         # Unassign group
#         self.model.dvTaskUnassign = pyo.Var(self.model.setTasks, domain=pyo.Binary, initialize=self.__initDvTaskUnassign())
#         # Assign registration to slot
#         self.model.dvAcSlot = pyo.Var(self.model.setAc, self.model.setSlots, domain=pyo.Binary, initialize=self.__initDvAcSlot())
#
#     def __initDvTaskSlot(self):
#         '''
#         Returns dictionary of initial values of decision variable for assignment of tasks to slots
#         '''
#         if self.simulation.scheduler_maintenance!=None:
#             # Find tuples (group, slot) in common between current and previous model
#             taskSlot = [tup for tup in self.model.setTaskSlot if tup in self.simulation.scheduler_maintenance.model.setTaskSlot]
#             # Find values corresponding to tuples
#             taskSlot_value = [round(self.simulation.scheduler_maintenance.model.dvTaskSlot[ts, sl]()) for (ts, sl) in taskSlot]
#             # generate initialization dictionary
#             taskSlot_dict = dict(zip(taskSlot, taskSlot_value))
#         else: taskSlot_dict = {}
#         return taskSlot_dict
#
#     def __initDvTaskUnassign(self):
#         if pd.isnull(self.simulation.scheduler_maintenance)==0:
#             taskUnassign = [ts for ts in self.model.setTasks if ts in self.simulation.scheduler_maintenance.model.setTasks]
#             taskUnassign_value = [round(self.simulation.scheduler_maintenance.model.dvTaskUnassign[ts]()) for ts in taskUnassign]
#             taskUnassign_dict = dict(zip(taskUnassign, taskUnassign_value))
#         else: taskUnassign_dict = {}
#         return taskUnassign_dict
#
#     def __initDvAcSlot(self):
#         if pd.isnull(self.simulation.scheduler_maintenance)==0:
#             acSlot = [(ac, sl) for ac in self.model.setAc for sl in self.model.setSlots if (ac,sl) in self.simulation.scheduler_maintenance.model.dvAcSlot]
#             acSlot_value = [round(self.simulation.scheduler_maintenance.model.dvAcSlot[ac, sl]()) for (ac, sl) in acSlot]
#             acSlot_dict = dict(zip(acSlot, acSlot_value))
#         else:
#             acSlot_dict = {}
#         return acSlot_dict
#     # =================================================================================#
#     # OBJECTIVE
#     # =================================================================================#
#     def __add_objective(self):
#         '''
#         Add objective function to the model
#         '''
#         # Slot remained unassigned
#         obj_unassign = sum(self.__choose_penalty(ts) * self.model.dvTaskUnassign[ts] for ts in
#                            self.model.setTaskPenalty)
#         # Optimal group-slot assignment
#         obj_interval = sum(self.matrices['LI'].loc[ts.id,sl.id] * self.model.dvTaskSlot[ts,sl]
#                            for (ts,sl) in self.model.setTaskSlot)
#         # Final objective
#         self.model.obj = pyo.Objective(expr=obj_unassign + obj_interval)
#
#
#     # =================================================================================#
#     # CONSTRAINTS
#     # =================================================================================#
#     def __add_constraints(self):
#         '''
#         Add constraints to the model
#         '''
#         self.__constrAssignTask()
#         self.__constrTaskAcSameSlot()
#         self.__constrAssignSlot()
#         self.__noEmptySlot()
#         self.__constrOncePerCycle()
#         self.__constrMaxLabor()
#         self.__constrMaxDuration() # TODO unnecessary constraint
#         self.__constrFixAcSlot()
#
#
#     def __constrAssignTask(self):
#         '''
#         Constraint 1: Task must be assigned only to one slot, or unassigned
#         '''
#         self.model.constrAssignTask = pyo.ConstraintList()
#         for task in self.model.setTasks:
#             expr = sum(self.model.dvTaskSlot[ts,sl] for (ts,sl) in self.model.setTaskSlot if ts==task)
#             expr = expr + self.model.dvTaskUnassign[task]
#             if isinstance(expr,int)==0:
#                 self.model.constrAssignTask.add(expr == 1)
#
#
#     def __constrAssignSlot(self):
#         '''
#         Constraint 2: Maximum one registration can be assigned to a slot
#         '''
#         self.model.constrAssignSlot = pyo.ConstraintList()
#         for sl in self.model.setSlots:
#             expr = sum(self.model.dvAcSlot[ac,sl] for ac in self.model.setAc)
#             self.model.constrAssignSlot.add(expr <= 1)
#
#     def __constrTaskAcSameSlot(self):
#         '''
#         Constraint 3: Group can be assigned to a slot only if that slot is assigned to the same aircraft
#         '''
#         self.model.constrTaskAcSameSlot = pyo.ConstraintList()
#         for ac in self.model.setAc:
#             for slot in self.model.setSlots:
#                 expr = sum(self.model.dvTaskSlot[(ts,sl)] for (ts,sl) in self.model.setTaskSlot
#                            if sl==slot and ts.aircraft == ac)
#                 self.model.constrTaskAcSameSlot.add(expr <= self.model.dvAcSlot[ac, slot] * G.M)
#
#
#     def __noEmptySlot(self):
#         '''
#         Constraint 4: Ensure that slots are not assigned to aircraft if no task is assigned to slot. This does not
#                       apply when a drumbeat logic is used
#         '''
#         if G.SLOTS_ASSIGNMENT_ORDER != 1:
#             self.model.constrNoEmptySlot = pyo.ConstraintList()
#             for slot in self.model.setSlots:
#                 exprLeft = sum(self.model.dvAcSlot[ac,slot] for ac in self.model.setAc)
#                 exprRight = sum(self.model.dvTaskSlot[ts,sl] for (ts,sl) in self.model.setTaskSlot if sl==slot)
#                 self.model.constrNoEmptySlot.add(exprLeft <= exprRight)
#
#
#     def __constrOncePerCycle(self):
#         '''
#         Constraint 5: Registration can be assigned to a slot only once per cycle. Only if requested in config
#         '''
#         if G.SLOTS_ASSIGNMENT_ONCE_PER_CYCLE == 1:
#             self.model.constrOncePerCycle = pyo.ConstraintList()
#             # Find current cycles
#             cycles = list(set([sl.cycle for sl in self.model.setSlots]))
#             # Find aircraft already assigned to current cycle
#             assignedAc = [sl.aircraft for sl in self.simulation.slots_executed
#                             if sl.cycle == self.simulation.slots_cycle and pd.isnull(sl.aircraft) == 0]
#             for cycle in cycles:
#                 for ac in self.model.setAc:
#                     expr = sum(self.model.dvAcSlot[ac,sl] for sl in self.model.setSlots if sl.cycle == cycle)
#                     # Impose non-assignement of aircraft to current cycle, when already assigned
#                     if cycle == self.simulation.slots_cycle and ac in assignedAc:
#                         expr = expr + 1
#                     self.model.constrOncePerCycle.add(expr <= 1)
#
#     def __constrMaxLabor(self):
#         '''
#         Constraint 6: Maximum labor allocated to slot
#         '''
#         self.model.constrMaxLabor = pyo.ConstraintList()
#         for slot in self.model.setSlots:
#             expr = sum(ts.laborEst.total_seconds() / 3600 * self.model.dvTaskSlot[ts,sl]
#                        for (ts, sl) in self.model.setTaskSlot if sl==slot)
#             if isinstance(expr,int)==0:
#                 self.model.constrMaxLabor.add(expr <= slot.laborMax.total_seconds()/3600)
#
#
#     def __constrMaxDuration(self):
#         '''
#         Constraint 7: Duration of task cannot exceed duration of slot
#         '''
#         self.model.constrMaxDuration = pyo.ConstraintList()
#         for (task, slot) in self.model.setTaskSlot:
#             durationTask = task.durationEst.total_seconds()/3600
#             # Check if duration is not zero, otherwise Pyomo error
#             if durationTask!=0:
#                 self.model.constrMaxDuration.add(durationTask * self.model.dvTaskSlot[task,slot] <= (slot.dateEnd_init - slot.dateStart_init).total_seconds()/3600)
#
#
#     def __constrFixAcSlot(self):
#         '''
#         Constraint 8: Fix assignment of slots   - historically defined / drumbeat
#                                                 - when slots are close to today
#                                                 - A-Check slots
#         '''
#         self.model.constrFixAcSlot = pyo.ConstraintList()
#         for slot in self.model.setSlots:
#             # A-Check slots always fixed
#             if slot.kindMnt == 'A': #TODO fix for A-checks
#                 self.model.constrFixAcSlot.add(self.model.dvAcSlot[slot.aircraft_pre_assigned, slot] == 1)
#             # If drumbeat assignment
#             elif G.SLOTS_ASSIGNMENT_ORDER == 1:
#                 if pd.isnull(slot.aircraft_pre_assigned)==0:
#                     self.model.constrFixAcSlot.add(self.model.dvAcSlot[slot.aircraft_pre_assigned, slot] == 1)
#
#             # If free slot assignment, fix slots assignment within certain window
#             elif G.SLOTS_ASSIGNMENT_FIX > 0:
#                 if slot.dateStart_init<=self.simulation.now+timedelta(days=G.SLOTS_ASSIGNMENT_FIX) and pd.isnull(slot.aircraft)==0:
#                     self.model.constrFixAcSlot.add(self.model.dvAcSlot[slot.aircraft, slot] == 1)
#
#     # =================================================================================#
#     # SOLUTION
#     # =================================================================================#
#     def solve(self, solver=None):
#         start_time = timeit.default_timer()
#         if solver == None:
#             solver = self.__find_solver()
#         solver = pyo.SolverFactory(solver)
#         # Specify allowed gap
#         solver.options['mipgap'] = G.MAINTENANCE_SCHED_ALLOWED_GAP
#         results = solver.solve(self.model, tee=False)
#
#         self.time_solve = timeit.default_timer() - start_time
#         print('\n##### MAINTENANCE SCHEDULE #####')
#         print('Time initialization:', self.time_initialization, '\nTime solution:', self.time_solve, '\n')
#
#         # Print solver status # TODO Add if solver status should be checked
#         if (results.solver.status == SolverStatus.ok) and (
#                 results.solver.termination_condition == TerminationCondition.optimal):
#             print("Found a feasible and optimal solution")
#         elif results.solver.termination_condition == TerminationCondition.infeasible:
#             print("WARNING: SOLUTION INFEASIBLE")
#             # Log info on infeasible model
#             logging.basicConfig(filename='maintenance_infeasible.log', force=True,level=logging.INFO,encoding='utf-8')
#             log_infeasible_constraints(self.model, log_expression = True, log_variables = True)
#         else:
#             # something else is wrong
#             print(str(results.solver))
#
#     @staticmethod
#     def __find_solver():
#         if isinstance(G.SOLVER,str):
#             solver = G.SOLVER
#         elif G.SOLVER==0:
#             solver = 'cbc'
#         elif G.SOLVER==1:
#             solver = 'gurobi_direct'
#         else:
#             raise Exception('Optimization solver not supported')
#         return solver
#
#     @staticmethod
#     def __choose_penalty(task):
#         if task.type == 'REQUIREMENT' or task.type == 'BLOCK':
#             penalty = G.PENALTY_UNASSIGN_RECURRING
#         elif task.type == 'ADHOC' or 'MEL' in task.type or 'NSRE' in task.type:
#             penalty = G.PENALTY_UNASSIGN_DD
#         else:
#             raise Exception('Unassignment penalty not found')
#         return penalty
#
#     def results(self):
#         for task in self.model.setTasks:
#             slot = [sl for (ts,sl) in self.model.dvTaskSlot if ts==task and self.model.dvTaskSlot[(task, sl)]() == 1 ]
#             if len(slot) == 0 and task.unassign_type == 'unassigned':
#                 print(task.aircraft.id + ' Task'+ task.id + ' ' + task.unassign_type)
#             elif len(slot) == 0 and task.unassign_type == 'postponed':
#                 print(task.aircraft.id + ' Task'+ task.id + ' ' + task.unassign_type)
#             elif len(slot)== 1:
#                 print(task.aircraft.id + ' Task'+ task.id + ' Slot '+ slot[0].id)
#             else:
#                 print('more than one assignment')
#
#
#
#     # =================================================================================#
#     # SOLUTION CHECK
#     # =================================================================================#
#
#     def check_results(self):
#         '''
#         Check that all constraints are fulfilled
#         '''
#         # Constraint 1: Group must be assigned only to one slot, or unassigned
#         for task in self.model.setTasks:
#             # Find slots to which task is assigned
#             slots = [sl for (ts, sl) in self.model.setTaskSlot
#                     if ts==task and round(self.model.dvTaskSlot[(task, sl)]()) == 1]
#
#             # Check if task unassigned
#             if round(self.model.dvTaskUnassign[task]()) == 1:
#                 unassign = 1
#             else:
#                 unassign = 0
#
#             # Check condition
#             total_assignments = len(slots) + unassign
#             if total_assignments == 0:
#                 raise Exception('Task never assigned')
#             elif total_assignments > 1:
#                 raise Exception('Task assigned more than once')
#
#         for slot in self.model.setSlots:
#             # Constraint 2: Maximum one registration can be assigned to a slot
#             # Find aircraft to which slot is assigned
#             aircraft = [ac for ac in self.model.setAc if round(self.model.dvAcSlot[ac,slot]())==1]
#             if len(aircraft) > 1:
#                 raise Exception('More than one registration is assigned to a slot')
#
#             # Constraint 3: Group can be assigned to a slot only if that slot is assigned to the same aircraft
#             # Find tasks assigned to slot
#             tasks = [ts for (ts, sl) in self.model.setTaskSlot
#                           if sl==slot and round(self.model.dvTaskSlot[(ts, slot)]()) == 1]
#
#             # Find aircraft of tasks assigned to slot
#             task_ac = [ts.aircraft for ts in tasks]
#             task_ac = list(set(task_ac))
#             if G.SLOTS_ASSIGNMENT_ORDER == 0 \
#                 and ((aircraft==[] and tasks!=[]) or (aircraft!=[] and tasks==[]) \
#                 or (aircraft!=[] and tasks!=[] and task_ac[0] != aircraft[0])):
#                 raise Exception('Group not assigned to slot of correct registration')
#             # elif G.slotsAssignmentOrder == 1 \ # TODO remove historical assignment
#             #     and tasks!=[] and aircraft[0] != task_ac[0]:
#             #     raise Exception('Group not assigned to slot of correct registration')
#
#             # Constraint 5: Maximum labor allocated to slot
#             # Find labor hours of tasks assigned to slot
#             tasks_labor = sum([ts.laborEst.total_seconds() / 3600 for ts in tasks])
#             if round(tasks_labor,2) > slot.laborMax.total_seconds()/3600:
#                 raise Exception('Too much labor is allocated to a slot')
#
#
#         # Constraint 4: Registration can be assigned to a slot only once per cycle
#         cycles = list(set([sl.cycle for sl in self.model.setSlots]))
#         for cycle in cycles:
#             aircraft_list = [ac for ac in self.model.setAc for sl in self.model.setSlots
#                              if round(self.model.dvAcSlot[ac,sl]()) == 1 and sl.cycle==cycle]
#             aircraft_set = list(set(aircraft_list))
#             if len(aircraft_list) != len(aircraft_set):
#                 raise Exception('Aircraft is assigned to cycle more than once')

def pyomo_maintenance_scheduler_Rcheck_sim(self): # TODO old scheduler
    scheduler_maintenance = MaintenanceScheduler_Rcheck_sim(self)
    # Optimize
    scheduler_maintenance.solve()
    # Print results and check them
    # scheduler_maintenance.results()
    scheduler_maintenance.check_results() # TODO uncomment if results should be checked

    return scheduler_maintenance


class _FixedBinaryValue:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class _FixedBinaryVar:
    def __init__(self, pairs_with_one):
        self._values = {pair: _FixedBinaryValue(1) for pair in pairs_with_one}
        self._zero = _FixedBinaryValue(0)

    def __getitem__(self, pair):
        return self._values.get(pair, self._zero)


class _ExternalSchedulerModel:
    def __init__(self, aircraft_slot_pairs, task_slot_pairs, task_slot_lm_pairs=None):
        if task_slot_lm_pairs is None:
            task_slot_lm_pairs = []

        self.set_aircraft_slot = list(aircraft_slot_pairs)
        self.set_task_slot = list(task_slot_pairs)
        self.set_task_slot_lm = list(task_slot_lm_pairs)

        all_task_slot_pairs = list(set(self.set_task_slot + self.set_task_slot_lm))
        self.dv_aircraft_slot = _FixedBinaryVar(self.set_aircraft_slot)
        self.dv_task_slot = _FixedBinaryVar(all_task_slot_pairs)


class _ExternalSchedulerBase:
    def __init__(self, simulation, name):
        self.simulation = simulation
        self.name = name
        self.time_initialization = 0
        self.time_solve = 0
        self.model = _ExternalSchedulerModel([], [], [])

    def solve(self, solver=None):
        return None

    def check_results(self):
        return None

    @staticmethod
    def _task_code(task):
        if task.requirement is not None and hasattr(task.requirement, 'code'):
            return task.requirement.code
        return None

    @staticmethod
    def _slot_compatible_with_aircraft(slot, aircraft):
        if slot.remarks == 'LM':
            return slot.aircraft == aircraft
        if slot.aircraft is not None and slot.aircraft != aircraft:
            return False
        return slot.subtype == aircraft.subtype.IATA


class MaintenanceSchedulerBaselineBlock(_ExternalSchedulerBase):
    def __init__(self, simulation):
        start_time = timeit.default_timer()
        super().__init__(simulation, 'baseline_mri_block_scheduler')
        self.block_sequence = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06',
                               'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
        self.task_blocks = self.__load_mappings()
        aircraft_slot_pairs, task_slot_pairs = self.__build_assignments()
        self.model = _ExternalSchedulerModel(aircraft_slot_pairs, task_slot_pairs, [])
        self.time_initialization = timeit.default_timer() - start_time

    def __load_mappings(self):
        mapping_path = os.path.join(directories.input, 'A-check MRI block mapping.xlsx')
        if not os.path.exists(mapping_path):
            mapping_path = 'A-check MRI block mapping.xlsx'
        if not os.path.exists(mapping_path):
            raise Exception('A-check MRI block mapping.xlsx not found in Data/input or repository root')

        df_map = pd.read_excel(mapping_path)
        if 'MRI Code' not in df_map.columns:
            raise Exception("Column 'MRI Code' not found in A-check MRI block mapping.xlsx")

        task_blocks = {}
        for _, row in df_map.iterrows():
            task_code = str(row['MRI Code']).strip()
            valid_blocks = [block for block in self.block_sequence
                            if block in df_map.columns and pd.notna(row[block]) and float(row[block]) == 1.0]
            if task_code and valid_blocks:
                task_blocks[task_code] = valid_blocks
        return task_blocks

    def __build_assignments(self):
        aircraft_slot_pairs = []
        task_slot_pairs = []
        unassigned_by_aircraft = {
            aircraft: sorted(
                [task for task in self.simulation.tasks_open if task.aircraft == aircraft],
                key=lambda task: task.dateDue
            )
            for aircraft in self.simulation.aircraft
        }
        block_index = {aircraft: 0 for aircraft in self.simulation.aircraft}

        slots = sorted(
            [slot for slot in self.simulation.slots_scheduling if slot.remarks != 'LM'],
            key=lambda slot: slot.dateStart_init
        )
        for slot in slots:
            # Labour does not cap planning: the A-check workforce scales to the planned work
            # (more people if needed), so all of the aircraft's due block tasks are assigned to
            # the slot. The labour total is tracked only as workforce demand / candidate ranking.
            candidates = []

            for aircraft in self.simulation.aircraft:
                if not self._slot_compatible_with_aircraft(slot, aircraft):
                    continue

                block = self.block_sequence[block_index[aircraft] % len(self.block_sequence)]
                gate_days = G.BLOCK_DUE_GATE_DAYS
                ready_tasks = [
                    task for task in unassigned_by_aircraft[aircraft]
                    if task.dateReady <= slot.dateStart_init
                    and (
                        gate_days is None
                        or slot.dateStart_init >= task.dateDue - timedelta(days=gate_days)
                    )
                    and (
                        self._task_code(task) not in self.task_blocks
                        or block in self.task_blocks[self._task_code(task)]
                    )
                ]
                if not ready_tasks:
                    continue

                mapped = [task for task in ready_tasks if self._task_code(task) in self.task_blocks]
                unmapped = [task for task in ready_tasks if self._task_code(task) not in self.task_blocks]
                ordered_tasks = sorted(mapped, key=lambda task: task.dateDue) \
                    + sorted(unmapped, key=lambda task: task.dateDue)

                selected = list(ordered_tasks)
                labor_used = sum(task.laborEst.total_seconds() / 3600 for task in selected)

                if selected:
                    earliest_due = min(task.dateDue for task in selected)
                    candidates.append((earliest_due, -labor_used, aircraft, selected))

            if not candidates:
                continue

            _, _, aircraft, selected = min(candidates, key=lambda item: (item[0], item[1]))
            aircraft_slot_pairs.append((aircraft, slot))
            for task in selected:
                task_slot_pairs.append((task, slot))
                unassigned_by_aircraft[aircraft].remove(task)
            block_index[aircraft] += 1

        return aircraft_slot_pairs, task_slot_pairs


class MaintenanceSchedulerSACGNN(_ExternalSchedulerBase):
    def __init__(self, simulation, slot_combine_window_days):
        start_time = timeit.default_timer()
        super().__init__(simulation, f'sac_gnn_scheduler_window_{slot_combine_window_days}')
        self.slot_combine_window_days = slot_combine_window_days
        aircraft_slot_pairs, task_slot_pairs = self.__build_assignments_from_sac()
        self.model = _ExternalSchedulerModel(aircraft_slot_pairs, task_slot_pairs, [])
        self.time_initialization = timeit.default_timer() - start_time

    @staticmethod
    def __resolve_path(path_value):
        path_value = path_value.replace('\\', os.sep).replace('/', os.sep)
        if os.path.isabs(path_value):
            return path_value
        return os.path.join(directories.anemos, path_value)

    def __load_sac_runtime(self):
        try:
            import torch
            from torch_geometric.data import Batch, Data
            from torch_geometric.nn import global_mean_pool
            from simulation.sac_gnn.VRP_SAC_Agent import AgentSAC
            from simulation.sac_gnn.creat_vrp_revised_new import creat_data
            from simulation.sac_gnn.vrpUpdate_1 import update_state, update_mask
        except Exception as exc:
            raise Exception('Failed importing SAC-GNN runtime dependencies. '
                            'Ensure torch/torch_geometric are installed in this environment.') from exc

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            torch.set_num_threads(max(1, G.SAC_GNN_CPU_THREADS))
        return {
            'torch': torch,
            'Batch': Batch,
            'Data': Data,
            'global_mean_pool': global_mean_pool,
            'AgentSAC': AgentSAC,
            'creat_data': creat_data,
            'update_state': update_state,
            'update_mask': update_mask,
            'device': device
        }

    @staticmethod
    def _validate_recurrence_assignments(task_chains, task_slot_pairs):
        """Reject illegal recurrence chains before exposing the fixed schedule to the simulator."""
        assigned_slot = dict(task_slot_pairs)
        for chain_key, tasks in task_chains.items():
            previous_date = None
            predecessor_missing = False
            for task in tasks:
                slot = assigned_slot.get(task)
                if slot is None:
                    predecessor_missing = True
                    continue
                execution_date = slot.dateStart_final.date()
                if predecessor_missing:
                    raise Exception(
                        'Recurring task leapfrogged an unassigned predecessor: '
                        f'{chain_key}, task={task.id}, slot={slot.id}'
                    )
                if not task.dateReady.date() <= execution_date <= task.dateDue.date():
                    raise Exception(
                        'Recurring task assigned outside its valid window: '
                        f'{chain_key}, task={task.id}, slot={slot.id}'
                    )
                if previous_date is not None and execution_date <= previous_date:
                    raise Exception(
                        'Recurring task assigned before or with its predecessor: '
                        f'{chain_key}, task={task.id}, slot={slot.id}'
                    )
                previous_date = execution_date

    def __build_policy_df_from_sim_state(self):
        initial_status_path = self.__resolve_path(G.SAC_GNN_INITIAL_STATUS_CSV_PATH)
        policy_path = self.__resolve_path(G.SAC_GNN_POLICY_CSV_PATH)
        if os.path.exists(initial_status_path):
            df = pd.read_csv(initial_status_path)
        elif os.path.exists(policy_path):
            df = pd.read_csv(policy_path)
        else:
            raise Exception('SAC policy CSV not found. Checked: ' + initial_status_path + ' and ' + policy_path)

        required = {'Task_code', 'Interval', 'Labour', 'Skill', 'Panel'}
        missing = required - set(df.columns)
        if missing:
            raise Exception('SAC policy CSV missing required columns: ' + str(sorted(missing)))

        now_date = self.simulation.now.date()
        base_due = df['Interval'].fillna(365).astype(float).clip(lower=1).astype(int)

        aircraft_list = list(self.simulation.aircraft)
        for idx, aircraft in enumerate(aircraft_list, start=1):
            due_by_code = {}
            for task in [ts for ts in self.simulation.tasks_open if ts.aircraft == aircraft]:
                code = self._task_code(task)
                if code is None:
                    continue
                due_days = (task.dateDue.date() - now_date).days
                due_days = max(1, due_days)
                if code not in due_by_code or due_days < due_by_code[code]:
                    due_by_code[code] = due_days

            col = f'Aircraft_{idx}'
            df[col] = [due_by_code.get(str(task_code).strip(), int(base_due.iloc[row_idx]))
                       for row_idx, task_code in enumerate(df['Task_code'])]

        return df

    @staticmethod
    def __combine_slots_within_window(aircraft_slots, window_days):
        if not aircraft_slots:
            return []
        sorted_slots = sorted(aircraft_slots, key=lambda x: x['preferred_date'])
        combined = [sorted_slots[0]]
        for slot in sorted_slots[1:]:
            current = combined[-1]
            if slot['preferred_date'] - current['preferred_date'] <= window_days:
                current['task_codes'].extend(slot['task_codes'])
            else:
                combined.append(slot)
        return combined

    @staticmethod
    def __resolve_date_conflicts(slots):
        slots_by_date = defaultdict(list)
        for slot in slots:
            slots_by_date[slot['preferred_date']].append(slot)

        used_dates = set()
        final_slots = []
        for date_key in sorted(slots_by_date.keys()):
            day_slots = sorted(slots_by_date[date_key], key=lambda s: len(s['task_codes']), reverse=True)
            for slot in day_slots:
                assigned_date = slot['preferred_date']
                while assigned_date in used_dates:
                    assigned_date -= 1
                used_dates.add(assigned_date)
                slot['assigned_date'] = assigned_date
                final_slots.append(slot)
        return final_slots

    def __generate_sac_schedule(self, sac_runtime, working_csv_path, policy_df):
        torch = sac_runtime['torch']
        Batch = sac_runtime['Batch']
        Data = sac_runtime['Data']
        global_mean_pool = sac_runtime['global_mean_pool']
        AgentSAC = sac_runtime['AgentSAC']
        creat_data = sac_runtime['creat_data']
        update_state = sac_runtime['update_state']
        update_mask = sac_runtime['update_mask']
        device = sac_runtime['device']

        class VRPEnv:
            def __init__(self, initial_batch_data, n_nodes):
                self.n_nodes = n_nodes
                self.depot_node_idx_local = n_nodes - 1
                self.batch_size = initial_batch_data.num_graphs

            def reset(self, batch_data):
                self.subtour_min_date = torch.full((self.batch_size,), -1.0, device=device)
                self.subtour_max_date = torch.full((self.batch_size,), -1.0, device=device)
                self.subtour_nodes_indices = [[] for _ in range(self.batch_size)]
                self.all_completed_subtour_labours = [[] for _ in range(self.batch_size)]
                self.static_capacity = batch_data.capacity.clone().to(device)
                self.scalar_static_capacity = self.static_capacity[0, 0].item()
                self.demands = batch_data.demand.clone().view(self.batch_size, self.n_nodes).to(device)
                self.capacities = self.static_capacity.clone()
                self.all_due_dates = batch_data.x[:, 4].clone().view(self.batch_size, self.n_nodes).to(device)
                self.dynamic_due_dates = self.all_due_dates.clone()
                self.all_intervals = batch_data.x[:, 1].clone().view(self.batch_size, self.n_nodes).to(device)
                self.visited_mask = torch.zeros((self.batch_size, self.n_nodes), dtype=torch.bool, device=device)

                initial_mask, self.visited_mask = update_mask(
                    self.demands,
                    self.capacities,
                    torch.full((self.batch_size, 1), self.depot_node_idx_local, device=device),
                    self.visited_mask,
                    self.depot_node_idx_local,
                    self.all_due_dates,
                    self.subtour_min_date,
                    self.subtour_max_date,
                    num_completed_slots=torch.zeros(self.batch_size, device=device, dtype=torch.long),
                    current_subtour_size=torch.zeros(self.batch_size, device=device, dtype=torch.long)
                )
                state_list = []
                for i in range(self.batch_size):
                    instance_data = batch_data[i]
                    state_list.append(Data(
                        x=instance_data.x,
                        edge_index=instance_data.edge_index,
                        edge_attr=instance_data.edge_attr,
                        raw_panel=instance_data.raw_panel,
                        demand=instance_data.demand,
                        capacity=instance_data.capacity,
                        mask=initial_mask[i].unsqueeze(-1)
                    ))
                self.current_batch_data = Batch.from_data_list(state_list).to(device)
                return self.current_batch_data

            def step(self, actions_tensor):
                for b_idx in range(self.batch_size):
                    selected_node_idx = actions_tensor[b_idx].item()
                    if selected_node_idx == self.depot_node_idx_local:
                        if len(self.subtour_nodes_indices[b_idx]) > 1:
                            subtour_labor = sum(
                                self.current_batch_data.x[self.current_batch_data.ptr[b_idx] + nid, 2].item()
                                for nid in self.subtour_nodes_indices[b_idx]
                            )
                            self.all_completed_subtour_labours[b_idx].append(subtour_labor)
                        self.subtour_min_date[b_idx] = -1.0
                        self.subtour_max_date[b_idx] = -1.0
                        self.subtour_nodes_indices[b_idx] = []
                    else:
                        due = self.all_due_dates[b_idx, selected_node_idx].item()
                        if self.subtour_min_date[b_idx].item() == -1.0:
                            self.subtour_min_date[b_idx] = due
                            self.subtour_max_date[b_idx] = due
                        else:
                            self.subtour_min_date[b_idx] = min(self.subtour_min_date[b_idx].item(), due)
                            self.subtour_max_date[b_idx] = max(self.subtour_max_date[b_idx].item(), due)
                        self.subtour_nodes_indices[b_idx].append(selected_node_idx)

                self.capacities, self.demands, self.dynamic_due_dates = update_state(
                    self.demands,
                    self.capacities,
                    self.dynamic_due_dates,
                    self.all_intervals,
                    self.subtour_min_date,
                    actions_tensor.unsqueeze(-1),
                    self.scalar_static_capacity,
                    self.depot_node_idx_local
                )

                num_completed_slots = torch.tensor([len(s) for s in self.all_completed_subtour_labours],
                                                   device=device, dtype=torch.long)
                current_subtour_size = torch.tensor([len(s) for s in self.subtour_nodes_indices],
                                                    device=device, dtype=torch.long)
                next_mask, self.visited_mask = update_mask(
                    self.demands,
                    self.capacities,
                    actions_tensor.unsqueeze(-1),
                    self.visited_mask,
                    self.depot_node_idx_local,
                    self.all_due_dates,
                    self.subtour_min_date,
                    self.subtour_max_date,
                    num_completed_slots=num_completed_slots,
                    current_subtour_size=current_subtour_size
                )

                next_state_list = []
                for i in range(self.batch_size):
                    instance_data = self.current_batch_data[i]
                    next_state_list.append(Data(
                        x=instance_data.x,
                        edge_index=instance_data.edge_index,
                        edge_attr=instance_data.edge_attr,
                        raw_panel=instance_data.raw_panel,
                        demand=self.demands[i].unsqueeze(-1),
                        capacity=self.capacities[i].unsqueeze(-1),
                        mask=next_mask[i].unsqueeze(-1)
                    ))
                self.current_batch_data = Batch.from_data_list(next_state_list).to(device)
                dones = self.demands[:, :self.depot_node_idx_local].le(0).all(dim=1)
                return self.current_batch_data, torch.zeros(self.batch_size), dones

        target_task_nodes = G.SAC_GNN_TARGET_TASK_NODES
        n_nodes = target_task_nodes + 1

        agent = AgentSAC(
            raw_node_feature_dim=5,
            demand_feature_dim=1,
            hidden_node_dim=256,
            input_edge_dim=1,
            hidden_edge_dim=32,
            conv_layers=3,
            n_nodes=n_nodes,
            learning_rate=1e-5
        )
        model_prefix = self.__resolve_path(G.SAC_GNN_MODEL_PREFIX)
        encoder_path = model_prefix + '_encoder.pth'
        actor_path = model_prefix + '_actor.pth'
        if not os.path.exists(encoder_path) or not os.path.exists(actor_path):
            raise Exception('SAC model files not found at: ' + model_prefix + '_[encoder|actor].pth')

        agent.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
        agent.encoder.eval()
        agent.actor.eval()

        _, unique_task_codes = pd.factorize(policy_df['Task_code'])
        encoded_to_task_code = {idx: str(code) for idx, code in enumerate(unique_task_codes)}

        n_aircraft = len(self.simulation.aircraft)
        instance_loader, _ = creat_data(
            file_path=working_csv_path,
            num_samples=n_aircraft,
            batch_size=n_aircraft,
            n_aircraft=n_aircraft,
            aircraft_ids=range(1, n_aircraft + 1),
            target_task_nodes=target_task_nodes,
            k_nearest_neighbors=G.SAC_GNN_K_NEAREST_NEIGHBORS
        )
        graph_data_batch = next(iter(instance_loader))
        env = VRPEnv(graph_data_batch, n_nodes)
        state = env.reset(graph_data_batch)
        action_sequences = [[] for _ in range(n_aircraft)]
        for _ in range(G.SAC_GNN_MAX_EPISODE_LEN):
            actions_np, _ = agent.select_action(state)
            for batch_index, action in enumerate(actions_np):
                action_sequences[batch_index].append(int(action))
            state, _, dones = env.step(torch.tensor(actions_np, device=device))
            if dones.all():
                break

        all_slots = []
        for batch_index, actions_sequence in enumerate(action_sequences):
            ac_index = batch_index + 1
            graph_data = graph_data_batch[batch_index]
            depot_node_idx = graph_data.num_nodes - 1
            current_subtour = []
            aircraft_slots = []
            aircraft_out_of_phase_tasks = []
            for node_idx in actions_sequence:
                if node_idx == depot_node_idx:
                    if len(current_subtour) > 1:
                        preferred_date = int(min(task['duedate'] for task in current_subtour))
                        aircraft_slots.append({
                            'aircraft_index': ac_index - 1,
                            'preferred_date': preferred_date,
                            'task_codes': [task['task_code'] for task in current_subtour]
                        })
                    elif len(current_subtour) == 1:
                        task = current_subtour[0]
                        aircraft_out_of_phase_tasks.append({
                            'aircraft_index': ac_index - 1,
                            'preferred_date': int(task['duedate']),
                            'task_codes': [task['task_code']],
                            'out_of_phase': True,
                        })
                    current_subtour = []
                    continue

                if 0 <= node_idx < graph_data.x.size(0):
                    encoded_task_code = int(graph_data.x[node_idx][0].item())
                    task_code = encoded_to_task_code.get(encoded_task_code)
                    if task_code is None:
                        continue
                    current_subtour.append({
                        'task_code': task_code,
                        'duedate': float(graph_data.x[node_idx][4].item())
                    })

            aircraft_slots = self.__combine_slots_within_window(
                aircraft_slots,
                self.slot_combine_window_days
            )
            all_slots.extend(aircraft_slots)
            all_slots.extend(aircraft_out_of_phase_tasks)

        if G.SAC_GNN_CONFLICT_RESOLUTION_ENABLED:
            all_slots = self.__resolve_date_conflicts(all_slots)
        else:
            for slot in all_slots:
                slot['assigned_date'] = slot['preferred_date']

        return all_slots

    def __build_assignments_from_sac(self):
        sac_runtime = self.__load_sac_runtime()
        policy_df = self.__build_policy_df_from_sim_state()

        temp_file = tempfile.NamedTemporaryFile(prefix='sac_gnn_policy_', suffix='.csv', delete=False)
        temp_file.close()
        working_csv_path = temp_file.name
        policy_df.to_csv(working_csv_path, index=False)

        try:
            schedule_slots = self.__generate_sac_schedule(sac_runtime, working_csv_path, policy_df)
        finally:
            try:
                os.remove(working_csv_path)
            except OSError:
                pass

        aircraft_slot_pairs = []
        task_slot_pairs = []
        slot_aircraft = {}
        sac_out_of_phase_count = 0
        recurrence_out_of_phase_count = 0
        out_of_phase_reassigned_count = 0
        out_of_phase_unassigned_count = 0
        predecessor_shifted_count = 0
        predecessor_blocked_count = 0
        # Last assigned execution date for each recurring (aircraft, task-code) chain.
        # A None value means an earlier occurrence could not be assigned, so all later
        # occurrences in that chain must remain blocked rather than leapfrog it.
        predecessor_execution = {}

        tasks_by_aircraft_code = {}
        task_chains = {}
        for aircraft in self.simulation.aircraft:
            task_map = defaultdict(list)
            for task in [ts for ts in self.simulation.tasks_open if ts.aircraft == aircraft]:
                task_code = self._task_code(task)
                if task_code is not None:
                    task_map[task_code].append(task)
            for code in task_map.keys():
                task_map[code] = sorted(task_map[code], key=lambda ts: ts.dateDue)
                task_chains[(aircraft, code)] = list(task_map[code])
            tasks_by_aircraft_code[aircraft] = task_map

        now_date = self.simulation.now.date()
        for slot_plan in sorted(schedule_slots, key=lambda x: x.get('assigned_date', x.get('preferred_date', 0))):
            aircraft = self.simulation.aircraft[slot_plan['aircraft_index']]
            assigned_tasks = []
            for task_code in slot_plan.get('task_codes', []):
                task_candidates = tasks_by_aircraft_code[aircraft].get(task_code, [])
                if len(task_candidates) == 0:
                    continue
                assigned_tasks.append(task_candidates.pop(0))

            if len(assigned_tasks) == 0:
                continue

            planned_date = now_date + timedelta(days=max(1, int(
                slot_plan.get('assigned_date', slot_plan.get('preferred_date', 1)))))

            # Preserve the SAC group at its planned maintenance opportunity. A task is
            # "out-of-phase" only when that opportunity falls outside its recurrence window
            # (previous-execution boundary/dateReady through due date). Such a task is detached
            # from the group and independently placed in this aircraft's latest feasible slot,
            # which is the slot closest to its due date and therefore minimizes spillage.
            group_slots = [
                sl for sl in self.simulation.slots_scheduling
                if sl.remarks != 'LM'
                and self._slot_compatible_with_aircraft(sl, aircraft)
                and (sl not in slot_aircraft or slot_aircraft[sl] == aircraft)
            ]
            planned_slot = (
                min(group_slots, key=lambda sl: abs((sl.dateStart_final.date() - planned_date).days))
                if group_slots else None
            )

            sac_declared_out_of_phase = bool(slot_plan.get('out_of_phase', False))
            if sac_declared_out_of_phase:
                sac_out_of_phase_count += len(assigned_tasks)

            # Process recurring instances in due-date order. This makes the predecessor state
            # deterministic even when the SAC group contains repeated copies of one task code.
            for task in sorted(assigned_tasks, key=lambda ts: ts.dateDue):
                task_code = self._task_code(task)
                chain_key = (aircraft, task_code)
                has_predecessor = chain_key in predecessor_execution
                previous_execution = predecessor_execution.get(chain_key)

                if has_predecessor and previous_execution is None:
                    task.unassign_type = 'predecessor_blocked'
                    predecessor_blocked_count += 1
                    out_of_phase_unassigned_count += 1
                    continue

                earliest_date = task.dateReady.date()
                if previous_execution is not None:
                    predecessor_boundary = previous_execution + timedelta(days=1)
                    if predecessor_boundary > earliest_date:
                        earliest_date = predecessor_boundary
                        predecessor_shifted_count += 1
                latest_date = task.dateDue.date()

                planned_slot_valid = (
                    not sac_declared_out_of_phase
                    and planned_slot is not None
                    and earliest_date <= planned_slot.dateStart_final.date() <= latest_date
                )
                if planned_slot_valid:
                    if planned_slot not in slot_aircraft:
                        slot_aircraft[planned_slot] = aircraft
                        aircraft_slot_pairs.append((aircraft, planned_slot))
                    task_slot_pairs.append((task, planned_slot))
                    predecessor_execution[chain_key] = planned_slot.dateStart_final.date()
                    continue

                if not sac_declared_out_of_phase:
                    recurrence_out_of_phase_count += 1
                feasible_slots = [
                    sl for sl in self.simulation.slots_scheduling
                    if sl.remarks != 'LM'
                    and self._slot_compatible_with_aircraft(sl, aircraft)
                    and (sl not in slot_aircraft or slot_aircraft[sl] == aircraft)
                    and earliest_date <= sl.dateStart_final.date() <= latest_date
                ]
                if not feasible_slots:
                    # Leave explicitly unassigned; check_results/the simulation will count it as
                    # missed rather than silently attaching it outside its recurrence interval.
                    # Mark the chain blocked so later occurrences cannot leapfrog this one.
                    task.unassign_type = 'out_of_phase_no_slot'
                    predecessor_execution[chain_key] = None
                    out_of_phase_unassigned_count += 1
                    continue
                closest_due_slot = max(feasible_slots, key=lambda sl: sl.dateStart_final)
                if closest_due_slot not in slot_aircraft:
                    slot_aircraft[closest_due_slot] = aircraft
                    aircraft_slot_pairs.append((aircraft, closest_due_slot))
                task_slot_pairs.append((task, closest_due_slot))
                predecessor_execution[chain_key] = closest_due_slot.dateStart_final.date()
                out_of_phase_reassigned_count += 1

        log_info(
            'SAC-GNN out-of-phase assignment: '
            f'SAC-declared={sac_out_of_phase_count}, '
            f'recurrence-boundary={recurrence_out_of_phase_count}, '
            f'reassigned={out_of_phase_reassigned_count}, '
            f'unassigned={out_of_phase_unassigned_count}, '
            f'predecessor-shifted={predecessor_shifted_count}, '
            f'predecessor-blocked={predecessor_blocked_count}'
        )

        self._validate_recurrence_assignments(task_chains, task_slot_pairs)
        return aircraft_slot_pairs, task_slot_pairs


def baseline_block_scheduler(self):
    scheduler_maintenance = MaintenanceSchedulerBaselineBlock(self)
    scheduler_maintenance.solve()
    scheduler_maintenance.check_results()
    return scheduler_maintenance


def sac_gnn_scheduler_scenario_1(self):
    scheduler_maintenance = MaintenanceSchedulerSACGNN(
        self,
        slot_combine_window_days=G.SAC_GNN_SCENARIO_1_SLOT_COMBINE_WINDOW_DAYS
    )
    scheduler_maintenance.solve()
    scheduler_maintenance.check_results()
    return scheduler_maintenance


def sac_gnn_scheduler_scenario_2(self):
    scheduler_maintenance = MaintenanceSchedulerSACGNN(
        self,
        slot_combine_window_days=G.SAC_GNN_SCENARIO_2_SLOT_COMBINE_WINDOW_DAYS
    )
    scheduler_maintenance.solve()
    scheduler_maintenance.check_results()
    return scheduler_maintenance



# ================================================================================= #
# DICTIONARIES OF MODULE OPTIONS
# ================================================================================= #
module_maintenance_scheduler_functions = {0: pyomo_maintenance_scheduler_Rcheck_sim,
                                          1: pyomo_maintenance_scheduler,
                                          2: pyomo_maintenance_scheduler_health,
                                          5: baseline_block_scheduler,
                                          6: sac_gnn_scheduler_scenario_1,
                                          7: sac_gnn_scheduler_scenario_2}
