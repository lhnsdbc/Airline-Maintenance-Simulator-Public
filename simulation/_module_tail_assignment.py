from datetime import timedelta
from config import G
import pyomo.environ as pyo
import pandas as pd
from datetime import timedelta, date, datetime
import timeit
from pyomo.opt import SolverStatus, TerminationCondition
from classes.classes_operations import Rotation
from output.output_functions import log_info, log_warning, log_error

'''
This script includes all tail assignment strategies that can be used within the simulation.
The modules take a Simulation instance as input and operate on it.

To add a new module function, define a module function and add the function to the dictionary at the end of this 
script. The function can be then selected from the config file, using the key specified in the dictionary. It is 
advised to add the key option in the config file for clarity of use.
'''

# =================================================================================#
# PYOMO OPTIMIZER FOR TAIL ASSIGNMENT WITH RESERVE AIRCRAFT
# =================================================================================#
class TailAssignmentWithReserveScheduler:
    def __init__(self,
                 simulation):
        self.simulation = simulation
        self.name = 'schedule_tail_assignment_with_reserve' + self.simulation.now.strftime('%Y/%m/%d_%H:%M_')\
                    + datetime.today().strftime('%Y%m%d%H%M%S%f')
        # Init time
        start_time = timeit.default_timer()

        # Init model
        self.model = pyo.ConcreteModel(name='tail_assignment')
        mid_time = timeit.default_timer()
        self.__add_sets()
        # print('Time add sets', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__add_decision_variables()
        # print('Time add decision variables', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__add_objective()
        # print('Time add objective', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__add_constraints()
        # print('Time add constraints', timeit.default_timer() - mid_time)

        self.time_initialization = timeit.default_timer() - start_time

    # =================================================================================#
    # PRECOMPUTED SCHEDULE MATRICES
    # =================================================================================#

    @staticmethod
    def _is_overlap(interval_A, interval_B):
        '''
        Find if there is an overlap between two datetime intervals given as a list of two elements [start, end]
        There is overlap between intervals A and B when (StartA <= EndB)  and  (EndA >= StartB).
        returns: 0 if there is not overlap, 1 if there is overlap
        '''
        # Find start and end of two intervals
        A_start = interval_A[0]
        A_end = interval_A[1]
        B_start = interval_B[0]
        B_end = interval_B[1]

        # Find conditions
        condition1 = A_start <= B_end
        condition2 = A_end >= B_start

        is_overlap = condition1 and condition2

        return is_overlap

    # =================================================================================#
    # SETS
    # =================================================================================#
    def __add_sets(self):
        ''' Add sets to the model'''
        # General sets
        mid_time = timeit.default_timer()
        self.model.setAircraft = pyo.Set(initialize=self.simulation.aircraft)
        self.model.setRotations = pyo.Set(initialize=self.__init_set_rotations())
        self.model.setReserveSlots = pyo.Set(initialize=self.__init_set_reserve_slots())
        self.model.setSegments = self.model.setRotations | self.model.setReserveSlots
        #print('\tTime sets Aircraft and Rotations', timeit.default_timer() - mid_time)

        # Reduced Rotation-Aircraft feasible assignment set
        mid_time = timeit.default_timer()
        self.model.setRotAc = pyo.Set(initialize=self.__assignRotAc(), dimen=2)
        self.model.setReserveAc = pyo.Set(initialize=self.__assignReserveAc(), dimen=2)
        #print('\tTime set Aircraft-Rotation', timeit.default_timer() - mid_time)

        # Set of overlapping rotations
        mid_time = timeit.default_timer()
        self.model.setSegmentsOverlap = pyo.Set(initialize=self.__overlapping_segments(), ordered=False)
        #print('\tTime set Overlapping rotations', timeit.default_timer() - mid_time)


    def __init_set_rotations(self):
        '''
        Set of rotation that must be assigned to aircraft.
        If it is the first time that the model is called, then all open rotations are to be assigned. If the module
        has already been called, then the recovery model only assigns rotations ARRIVING at AMS after a specified
        number of days.
        '''
        window_start = self.__find_window_start()
        return [rot for rot in self.simulation.rotations_open
                if rot.arr_sched > self.simulation.now + timedelta(days=window_start)
                and self.__rotation_is_current_duty(rot) == False]

    def __init_set_reserve_slots(self):
        ''' Set of reserve slots that must be assigned to aircraft. '''
        window_start = self.__find_window_start()
        return [rs for rs in self.simulation.reserve_slots
                if rs.arr_sched > self.simulation.now + timedelta(days=window_start)
                and rs.dep_sched <= self.simulation.now + timedelta(days=G.TAIL_ASSIGNMENT_WINDOW)]

    def __find_window_start(self):
        ''' Return the start of the tail assignment window'''
        if self.simulation.scheduler_tail_assignment == None:
            window_start = 0
        else:
            window_start = G.TAIL_ASSIGNMENT_FIX
        return window_start

    def __rotation_is_current_duty(self, rot):
        ''' Return True if a rotation is the current duty of an aircraft, False if it is not or if it is now assigned
        to any aircraft'''
        if pd.isnull(rot.aircraft):
            return False
        elif rot == rot.aircraft.duty_current:
            return True
        else:
            return False

    def __assignRotAc(self):
        return [(rot, ac)
                for rot in self.model.setRotations
                for ac in self.model.setAircraft
                if self.__segm_ac_compatibility(rot, ac) == True]

    def __assignReserveAc(self):
        return [(res, ac)
                for res in self.model.setReserveSlots
                for ac in self.model.setAircraft
                if self.__segm_ac_compatibility(res, ac) == True]

    def __segm_ac_compatibility(self, segment, aircraft):
        '''
        Find if a segment (rotation or reserve slot) can be assigned to an aircraft or not, considering:
        - fleet assignment (only for rotations)
        - maintenance slots assigned to the aircraft
        - rotations previously assigned to an aircraft that are now fixed
        :return: True if rotation can be assigned to an aircraft, False otherwise
        '''
        ##### AIRCRAFT SUBTYPE COMPATIBILITY #####
        # Check for subtype compatibility only if rotation input is not a reserve slot
        if type(segment) == Rotation:
            # Full subtype compatibility
            if G.TAIL_ASSIGNMENT_FIXED_FLEET_ASSIGNMENT == 1 \
                    and (aircraft.subtype not in segment.rotation_norm.subtypes):
                return False
            # Preferred subtype compatibility
            elif G.TAIL_ASSIGNMENT_FIXED_FLEET_ASSIGNMENT == 2:
                subtypes_rotation = list(set([st.IATA for st in segment.rotation_norm.subtypes]))
                # Find second choices for rotation-aircraft assignment
                subtypes_second_choice = []
                for st in subtypes_rotation:
                    second_choices = next(sc for sc in G.PREFERRED_SUBTYPES_GROUPS if st in sc)
                    subtypes_second_choice = subtypes_second_choice + second_choices
                subtypes_second_choice = list(set(subtypes_second_choice))
                if aircraft.subtype.IATA not in subtypes_second_choice:
                    return False

        ##### MAINTENANCE SLOTS COMPATIBILITY #####
        ac_maintenance_slots = aircraft.slots
        # Find the buffer time to keep before and after the segment
        if type(segment) == Rotation:
            buffer_segment = timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)
        else:
            buffer_segment = timedelta(minutes=0)

        for slot in ac_maintenance_slots:
            # The expected departure and arrival time is used for slots instead of the scheduled to account for long AOG
            is_overlap = self._is_overlap([segment.dep_sched - buffer_segment, segment.arr_sched + buffer_segment],
                                          [slot.dateStart_final-slot.towing_time, slot.dateEnd_final+slot.towing_time])

            if is_overlap == True:
                return False

        ##### LAST ROTATION ASSIGNED TO AIRCRAFT #####
        assigned_rotations = [rt for rt in aircraft.rotations if rt not in self.model.setRotations]
        assigned_reserve_slots = [rs for rs in aircraft.reserve_slots if rs not in self.model.setReserveSlots]
        assigned_segments = assigned_rotations + assigned_reserve_slots
        if assigned_segments!=[]:
            assigned_segments = sorted(assigned_segments, key=lambda x:x.dep_sched)
            last_segment = assigned_segments[-1]
            if type(last_segment) == Rotation:
                buffer_segment_last = timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)
            else:
                buffer_segment_last = timedelta(minutes=0)
            if segment.dep_sched - buffer_segment < last_segment.arr_sched+buffer_segment_last:
                return False

        ##### NO INCOMPATIBILITY FOUND #####
        return True


    def __overlapping_segments(self):
        old_scheduler = self.simulation.scheduler_tail_assignment

        # If tail assignment is running for the first time, generate lists from scratch
        if old_scheduler == None:
            # List of rotations for which overlap must be found
            segments_new = [rt for rt in self.model.setSegments]
            # List of previously overlapping rotations
            overlapping_segments_old = []
        # If tail assignment was run before, use part of previous lists
        else:
            # List of rotations for which overlap must be found
            segments_new = [rt for rt in self.model.setSegments
                            if rt not in old_scheduler.model.setSegments]
            # List of previously overlapping rotations
            overlapping_segments_old = [rr for rr in old_scheduler.model.setSegmentsOverlap
                                         if rr.issubset(self.model.setSegments)]

        # Find overlap between new segments and all other segments
        overlapping_rotations_new = set()
        segments_checked = set()
        for segm1 in segments_new:
            # Find buffer for segm1
            if type(segm1) == Rotation:
                buffer1 = timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)
            else:
                buffer1 = timedelta(minutes=0)

            # Find segments to check for overlap
            segments_to_check = [sg for sg in self.model.setSegments
                                 if sg!=segm1 and frozenset([segm1, sg]) not in segments_checked]

            for segm2 in segments_to_check:
                # Find buffer for segm2
                if type(segm2) == Rotation:
                    buffer2 = timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)
                else:
                    buffer2 = timedelta(minutes=0)
                if self._is_overlap([segm1.dep_sched - buffer1, segm1.arr_sched + buffer1],
                                    [segm2.dep_sched - buffer2, segm2.arr_sched + buffer2]) == True:
                    overlapping_rotations_new.add(frozenset([segm1, segm2]))

                segments_checked.add(frozenset([segm1, segm2]))

        overlapping_rotations = overlapping_rotations_new.union(set(overlapping_segments_old))

        return overlapping_rotations

    # =================================================================================#
    # DECISION VARIABLES
    # =================================================================================#
    def __add_decision_variables(self):
        '''
        Add decision variables to the model
        '''
        # Assign rotation to Aircraft
        self.model.dvRotAc = pyo.Var(self.model.setRotAc, domain=pyo.Binary, initialize=self.__initDvRotAc())
        # Assign reserve slot to Aircraft
        self.model.dvReserveAc = pyo.Var(self.model.setReserveAc, domain=pyo.Binary, initialize=self.__initDvResAc())

        # Unassign rotation
        self.model.dvRotUnassign = pyo.Var(self.model.setRotations, domain=pyo.Binary,
                                           initialize=self.__initDvRotUnassign())
        # Unassign reserve slot
        self.model.dvReserveUnassign = pyo.Var(self.model.setReserveSlots, domain=pyo.Binary,
                                               initialize=self.__initDvReserveUnassign())



    def __initDvRotAc(self):
        '''
        Returns dictionary of initial values of decision variable for assignment of rotations to aircraft
        '''
        old_scheduler = self.simulation.scheduler_tail_assignment
        if  old_scheduler != None:
            # Find tuples (rotation, aircraft) in common between current and previous model
            rotAc = [rotAc for rotAc in self.model.setRotAc if
                     rotAc in old_scheduler.model.setRotAc]
            # Find values corresponding to tuples
            rotAc_value = [round(old_scheduler.model.dvRotAc[rt, ac]())
                           for (rt, ac) in rotAc]
            # generate initialization dictionary
            rotAc_dict = dict(zip(rotAc, rotAc_value))
        else:
            rotAc_dict = {}
        return rotAc_dict

    def __initDvResAc(self):
        '''
        Returns dictionary of initial values of decision variable for assignment of reserve slots to aircraft
        '''
        old_scheduler = self.simulation.scheduler_tail_assignment
        if  old_scheduler!= None:
            # Find tuples (reserve slot, aircraft) in common between current and previous model
            rs_aircraft = [rsAc for rsAc in self.model.setReserveAc
                           if rsAc in old_scheduler.model.setReserveAc]
            # Find values corresponding to tuples
            rs_aircraft_value = [round(old_scheduler.model.dvReserveAc[rs, ac]())
                                 for (rs, ac) in rs_aircraft]
            # generate initialization dictionary
            rotAc_dict = dict(zip(rs_aircraft, rs_aircraft_value))
        else:
            rotAc_dict = {}
        return rotAc_dict

    def __initDvRotUnassign(self):
        '''
        Returns dictionary of initial values of decision variable for leaving a rotation unassigned
        '''
        old_scheduler = self.simulation.scheduler_tail_assignment
        if pd.isnull(old_scheduler) == 0:
            rotUnassign = [rot for rot in self.model.setRotations if
                           rot in old_scheduler.model.setRotations]
            rotUnassign_value = [round(old_scheduler.model.dvRotUnassign[rt]()) for rt in
                                 rotUnassign]
            rotUnassign_dict = dict(zip(rotUnassign, rotUnassign_value))
        else:
            rotUnassign_dict = {}
        return rotUnassign_dict

    def __initDvReserveUnassign(self):
        '''
        Returns dictionary of initial values of decision variable for leaving a reserve slot unassigned
        '''
        old_scheduler = self.simulation.scheduler_tail_assignment
        if pd.isnull(old_scheduler) == 0:
            reserve_unassign = [res for res in self.model.setReserveSlots
                                if res in old_scheduler.model.setReserveSlots]
            reserve_unassign_value = [round(old_scheduler.model.dvReserveUnassign[rs]())
                                      for rs in reserve_unassign]
            reserve_unassign_dict = dict(zip(reserve_unassign, reserve_unassign_value))
        else:
            reserve_unassign_dict = {}
        return reserve_unassign_dict

    # =================================================================================#
    # OBJECTIVE
    # =================================================================================#
    def __add_objective(self):
        '''
        Add objective function to the model
        '''
        obj_unassign_rotation = sum(G.PENALTY_UNASSIGN_ROTATION * self.model.dvRotUnassign[rt]
                                    for rt in self.model.setRotations)
        obj_unassign_reserve_slot = sum(G.PENALTY_UNASSIGN_RESERVE_SLOT * self.model.dvReserveUnassign[rs]
                                    for rs in self.model.setReserveSlots)
        obj_rotation_preference_group = sum(self.__choose_penalty_rotation_aircraft_subtype(rot, ac) * self.model.dvRotAc[rot, ac]
                                            for (rot, ac) in self.model.setRotAc)

        self.model.obj = pyo.Objective(expr=obj_unassign_rotation + obj_unassign_reserve_slot +
                                            obj_rotation_preference_group)

    @staticmethod
    def __choose_penalty_rotation_aircraft_subtype(rotation, aircraft):
        '''
        Find the penalty for assigning a rotation to an aircraft of a certain subtype.
        The penalty can be equal to:
        - 0: preferred assignment
        - PENALTY_SUBTYPE_LOW: assignment different from preferred, but within preferred group
        - PENALTY_SUBTYPE_HIGH: assignment out of preference
        '''
        subtypes_rotation = [st.IATA for st in rotation.rotation_norm.subtypes]
        # Find second choices for rotation-aircraft assignment
        subtypes_second_choice = []
        for st in subtypes_rotation:
            second_choices = next(sc for sc in G.PREFERRED_SUBTYPES_GROUPS if st in sc)
            subtypes_second_choice = subtypes_second_choice + second_choices
        subtypes_second_choice = list(set(subtypes_second_choice))
        if aircraft.subtype.IATA in subtypes_rotation:
            penalty = 0
        elif aircraft.subtype.IATA in subtypes_second_choice:
            penalty = G.PENALTY_SUBTYPE_LOW
        else:
            penalty = G.PENALTY_SUBTYPE_HIGH
        return penalty

    # =================================================================================#
    # CONSTRAINTS
    # =================================================================================#
    def __add_constraints(self):
        '''
        Add constraints to the model
        '''
        mid_time = timeit.default_timer()
        self.__constrAssignSegment()
        # print('\tTime constraint Rot assignment', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__constrSegmentsOverlap()
        # print('\tTime constraint Rot overlap', timeit.default_timer() - mid_time)

    def __constrAssignSegment(self):
        '''
        Constraint 1: Rotations and Reserve Slots must be assigned only to one aircraft, or unassigned
        '''
        # Initialize constraints list
        self.model.constrAssignSegment = pyo.ConstraintList()

        # Rotations
        for rotation in self.model.setRotations:
            expr = sum(self.model.dvRotAc[rt,ac] for (rt,ac) in self.model.setRotAc if rt==rotation)
            expr = expr + self.model.dvRotUnassign[rotation]
            if isinstance(expr, int)==0:
                self.model.constrAssignSegment.add(expr == 1)

        # Reserve slots
        for reserve_slot in self.model.setReserveSlots:
            expr = sum(self.model.dvReserveAc[rs,ac] for (rs,ac) in self.model.setReserveAc if rs==reserve_slot)
            expr = expr + self.model.dvReserveUnassign[reserve_slot]
            if isinstance(expr, int)==0:
                self.model.constrAssignSegment.add(expr == 1)


    def __constrSegmentsOverlap(self):
        '''
        Constraint 2: Only rotations that do not overlap can be assigned to an aircraft
        '''
        self.model.constrRotationsOverlap = pyo.ConstraintList()

        time_start = timeit.default_timer()
        for (segm1, segm2) in self.model.setSegmentsOverlap:
            mid_time = timeit.default_timer()
            # Find aircraft that can fly both segments
            compatible_aircraft = [ac for ac in self.model.setAircraft
                                   if ([segm1, ac] in self.model.setRotAc or [segm1, ac] in self.model.setReserveAc)
                                   and ([segm2, ac] in self.model.setRotAc or [segm2, ac] in self.model.setReserveAc)]
            time_execution = timeit.default_timer() - mid_time
            # print(':', time_execution)


            mid_time = timeit.default_timer()
            # Add constraint
            for aircraft in compatible_aircraft:
                # Find decision variables for the two segments
                if type(segm1) == Rotation:
                    dv1 = self.model.dvRotAc[segm1, aircraft]
                else:
                    dv1 = self.model.dvReserveAc[segm1, aircraft]

                if type(segm2) == Rotation:
                    dv2 = self.model.dvRotAc[segm2, aircraft]
                else:
                    dv2 = self.model.dvReserveAc[segm2, aircraft]

                self.model.constrRotationsOverlap.add(dv1 + dv2 <= 1)
            time_execution = timeit.default_timer() - mid_time
            # print(':', time_execution)

        time_total = timeit.default_timer() - time_start
        #print(':', time_total)

    # =================================================================================#
    # SOLUTION
    # =================================================================================#
    def solve(self, solver=None):
        start_time = timeit.default_timer()
        if solver == None:
            solver = self.__find_solver()
        solver = pyo.SolverFactory(solver)
        # Specify allowed gap
        solver.options['mipgap'] = G.TAIL_ASSIGNMENT_ALLOWED_GAP
        if getattr(G, 'SOLVER_THREADS', None) is not None:
            solver.options['threads'] = G.SOLVER_THREADS
        results = solver.solve(self.model, tee=False)

        self.time_solve = timeit.default_timer() - start_time
        log_info('\n##### TAIL ASSIGNMENT #####')
        log_info('Time initialization:', self.time_initialization, '\nTime solution:', self.time_solve,'\n')

        if results.solver.termination_condition == TerminationCondition.infeasible:
            log_info("WARNING: SOLUTION INFEASIBLE")

        # Print solver status # TODO Add if solver status should be checked
        if (results.solver.status == SolverStatus.ok) \
                and (results.solver.termination_condition == TerminationCondition.optimal):
            log_info("Found a feasible and optimal solution")
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            log_error("WARNING: SOLUTION INFEASIBLE")
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
        ''' Print results'''
        for aircraft in self.model.setAircraft:
            rotations = [rt for (rt, ac) in self.model.dvRotAc
                         if ac==aircraft and round(self.model.dvRotAc[rt, ac]())==1]
            # Order rotations
            rotations = sorted(rotations, key=lambda x: x.dep_sched)
            for rotation in rotations:
                time_dep = rotation.dep_sched.strftime('%Y-%m-%d %H:%M:%S')
                time_arr = rotation.arr_sched.strftime('%Y-%m-%d %H:%M:%S')
                print(aircraft.id, rotation.id,' Dep:',time_dep,'Arr:',time_arr)

        for rotation in [rt for rt in self.model.setRotations if round(self.model.dvRotUnassign[rt]())==1]:
                print(rotation.id,' unassigned')


def tail_assignment_with_reserve_pyomo(self):
    scheduler_tail_assignment = TailAssignmentWithReserveScheduler(self)
    # Optimize
    scheduler_tail_assignment.solve()
    # Print results and check them
    # scheduler_tail_assignment.results() # TODO uncomment line if results should be printed in console
    # scheduler_tail_assignment.check_results()

    return scheduler_tail_assignment



# =================================================================================#
# PYOMO OPTIMIZER FOR TAIL ASSIGNMENT
# =================================================================================#
class TailAssignmentScheduler:
    '''
    This tail assignment optimization class assigns rotations to aircraft based on the following:
    OBJECTIVE (MINIMIZE):
    - Number of cancelled rotations
    - Assignment of a rotation to a non-preferred aircraft subtype

    CONSTRAINTS:
    - A rotation can either be assigned to one aircraft, or cancelled
    - Any pair of two rotations can be assigned to an aircraft only if there is no overlap between them and none of
    them overlap with an assigned maintenance slot
    '''
    def __init__(
            self,
            simulation):
        self.simulation = simulation
        self.name = 'schedule_tail_assignment'+self.simulation.now.strftime('%d/%m/%Y')
        # Init time
        start_time = timeit.default_timer()

        # Flights compatibility dataframe
        mid_time = timeit.default_timer()
        self.matrix_compatibility = self.__generate_compatibility_matrix()
        print('Time matrix init',  timeit.default_timer() - mid_time)

        # Init model
        self.model = pyo.ConcreteModel(name='tail_assignment')
        mid_time = timeit.default_timer()
        self.__add_sets()
        print('Time add sets', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__add_decision_variables()
        print('Time add decision variables', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__add_objective()
        print('Time add objective', timeit.default_timer() - mid_time)

        mid_time = timeit.default_timer()
        self.__add_constraints()
        print('Time add constraints', timeit.default_timer() - mid_time)

        self.time_initialization = timeit.default_timer() - start_time

    # =================================================================================#
    # PRECOMPUTED SCHEDULE MATRICES
    # =================================================================================#
    def __generate_compatibility_matrix(self):
        '''
        Generate the compatibility matrix between rotations.
        When element [i,j] of the matrix is equal to 1, two rotations can be assigned to the same aircraft since
        there is no overlap between them. One hour slack before and after the rotations is considered when computing
        the overlap.
        '''
        rotations_open = self.simulation.rotations_open
        # Generate empty compatibility matrix
        matrix = pd.DataFrame(None,
                              index=[rt.id for rt in rotations_open],
                              columns=[rt.id for rt in rotations_open])

        # Fill matrix
        for ind1 in range(len(rotations_open)):
            for ind2 in range(ind1+1):
                rot1 = rotations_open[ind1]
                rot2 = rotations_open[ind2]

                # Find if there is an overlap between rotations
                is_overlap = self.__is_overlap([rot1.dep_sched - timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT),
                                                rot1.arr_sched + timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)],
                                               [rot2.dep_sched - timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT),
                                                rot2.arr_sched + timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)])

                # Rotations are compatibile if there is no overlap
                matrix.loc[rot1.id, rot2.id] = int(not (is_overlap))
                matrix.loc[rot2.id, rot1.id] = int(not (is_overlap))

        return matrix

    @staticmethod
    def __is_overlap(interval_A, interval_B):
        '''
        Find if there is an overlap between two datetime intervals given as a list of two elements [start, end]
        There is overlap between intervals A and B when (StartA <= EndB)  and  (EndA >= StartB).
        returns: 0 if there is not overlap, 1 if there is overlap
        '''
        # Find start and end of two intervals
        A_start = interval_A[0]
        A_end = interval_A[1]
        B_start = interval_B[0]
        B_end = interval_B[1]

        # Find conditions
        condition1 = A_start <= B_end
        condition2 = A_end >= B_start

        is_overlap = condition1 and condition2

        return is_overlap



    # =================================================================================#
    # SETS
    # =================================================================================#
    def __add_sets(self):
        ''' Add sets to the model'''
        # General sets
        self.model.setAircraft = pyo.Set(initialize=self.__aircraft_available())#self.simulation.aircraft)
        self.model.setRotations = pyo.Set(initialize=self.__init_set_rotations())

        # Reduced Rotation-Aircraft feasible assignment set
        self.model.setRotAc = pyo.Set(initialize=self.__assignRotAc(), dimen=2)

    def __aircraft_available(self):
        # return [ac for ac in self.simulation.aircraft if ac.id not in self.simulation.aircraft_reserve]
        return [ac for ac in self.simulation.aircraft]



    def __init_set_rotations(self):
        '''
        Set of rotation that must be assigned to aircraft.
        If it is the first time that the model is called, then all open rotations are to be assigned. If the module
        has already been called, then the recovery model only assigns rotations ARRIVING at AMS after a specified
        number of days.
        '''
        if self.simulation.scheduler_tail_assignment == None:
            window_start = 0
        else:
            window_start = G.TAIL_ASSIGNMENT_FIX
        return [rot for rot in self.simulation.rotations_open
                if rot.arr_sched > self.simulation.now + timedelta(days=window_start)]

    def __assignRotAc(self):
        return [(rot, ac)
                for rot in self.model.setRotations
                for ac in self.model.setAircraft
                if self.__rot_ac_compatibility(rot, ac) == True]

    def __rot_ac_compatibility(self, rotation, aircraft):
        '''
        Find if a rotation can be assigned to an aircraft or not, considering
        - fleet assignment
        - maintenance slots assigned to the aircraft
        - rotations previously assigned to an aircraft that are now fixed
        :return: True if rotation can be assigned to an aircraft, False otherwise
        '''
        # Check compatibility with ac subtypetype
        # Full subtype compatibility
        if G.TAIL_ASSIGNMENT_FIXED_FLEET_ASSIGNMENT == 1 \
                and (aircraft.subtype not in rotation.rotation_norm.subtypes):
            return False
            # Preferred subtype compatibility
        elif G.TAIL_ASSIGNMENT_FIXED_FLEET_ASSIGNMENT == 2:
            subtypes_rotation = [st.IATA for st in rotation.rotation_norm.subtypes]
            # Find second choices for rotation-aircraft assignment
            subtypes_second_choice = []
            for st in subtypes_rotation:
                second_choices = next(sc for sc in G.PREFERRED_SUBTYPES_GROUPS if st in sc)
                subtypes_second_choice = subtypes_second_choice + second_choices
            subtypes_second_choice = list(set(subtypes_second_choice))
            if aircraft.subtype.IATA not in subtypes_second_choice:
                return False

        # Check compatibility with maintenance slots
        ac_maintenance_slots = aircraft.slots
        for slot in ac_maintenance_slots:
            # The expected departure and arrival time is used for slots instead of the scheduled to account for long AOG
            is_overlap = self.__is_overlap([rotation.dep_sched-timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT),
                                            rotation.arr_sched+timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)],
                                           [slot.dateStart_final-slot.towing_time,
                                            slot.dateEnd_final+slot.towing_time])
            if is_overlap == True:
                return False

        # Check compatibility with last aircraft assigned rotation
        assigned_rotations = [rt for rt in aircraft.rotations if rt not in self.model.setRotations]
        if assigned_rotations!=[]:
            assigned_rotations = sorted(assigned_rotations, key=lambda x:x.dep_sched)
            last_rot = assigned_rotations[-1]
            if (rotation.dep_sched-timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)) \
                    < (last_rot.arr_sched+timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)):
                return False

        # If not incompatibility found, return True
        return True

    # =================================================================================#
    # DECISION VARIABLES
    # =================================================================================#
    def __add_decision_variables(self):
        '''
        Add decision variables to the model
        '''
        # Assign rotation to Aircraft
        self.model.dvRotAc = pyo.Var(self.model.setRotAc, domain=pyo.Binary,
                                     initialize=self.__initDvRotAc())

        # Unassign rotation
        self.model.dvRotUnassign = pyo.Var(self.model.setRotations, domain=pyo.Binary,
                                           initialize=self.__initDvRotUnassign())

    def __initDvRotAc(self):
        '''
        Returns dictionary of initial values of decision variable for assignment of rotations to aircraft
        '''
        if self.simulation.scheduler_tail_assignment != None:
            # Find tuples (rotation, aircraft) in common between current and previous model
            rotAc = [rotAc for rotAc in self.model.setRotAc if
                     rotAc in self.simulation.scheduler_tail_assignment.model.setRotAc]
            # Find values corresponding to tuples
            rotAc_value = [round(self.simulation.scheduler_tail_assignment.model.dvRotAc[rt, ac]())
                           for (rt, ac) in rotAc]
            # generate initialization dictionary
            rotAc_dict = dict(zip(rotAc, rotAc_value))
        else:
            rotAc_dict = {}
        return rotAc_dict

    def __initDvRotUnassign(self):
        '''
        Returns dictionary of initial values of decision variable for leaving a rotation unassigned
        '''
        if pd.isnull(self.simulation.scheduler_tail_assignment) == 0:
            rotUnassign = [rot for rot in self.model.setRotations if
                           rot in self.simulation.scheduler_tail_assignment.model.setRotations]
            rotUnassign_value = [round(self.simulation.scheduler_tail_assignment.model.dvRotUnassign[rt]()) for rt in
                                 rotUnassign]
            rotUnassign_dict = dict(zip(rotUnassign, rotUnassign_value))
        else:
            rotUnassign_dict = {}
        return rotUnassign_dict

    # =================================================================================#
    # OBJECTIVE
    # =================================================================================#
    def __add_objective(self):
        '''
        Add objective function to the model
        '''
        obj_unassign = sum(G.PENALTY_UNASSIGN_ROTATION * self.model.dvRotUnassign[rt]
                           for rt in self.model.setRotations)
        obj_preference_group = sum(self.__choose_penalty_rotation_aircraft_subtype(rot, ac) * self.model.dvRotAc[rot, ac]
                                   for (rot, ac) in self.model.setRotAc)
        self.model.obj = pyo.Objective(expr=obj_unassign + obj_preference_group)

    @staticmethod
    def __choose_penalty_rotation_aircraft_subtype(rotation, aircraft):
        '''
        Find the penalty for assigning a rotation to an aircraft of a certain subtype.
        The penalty can be equal to:
        - 0: preferred assignment
        - PENALTY_SUBTYPE_LOW: assignment different from preferred, but within preferred group
        - PENALTY_SUBTYPE_HIGH: assignment out of preference
        '''
        subtypes_rotation = [st.IATA for st in rotation.rotation_norm.subtypes]
        # Find second choices for rotation-aircraft assignment
        subtypes_second_choice = []
        for st in subtypes_rotation:
            second_choices = next(sc for sc in G.PREFERRED_SUBTYPES_GROUPS if st in sc)
            subtypes_second_choice = subtypes_second_choice + second_choices
        subtypes_second_choice = list(set(subtypes_second_choice))
        if aircraft.subtype.IATA in subtypes_rotation:
            penalty = 0
        elif aircraft.subtype.IATA in subtypes_second_choice:
            penalty = G.PENALTY_SUBTYPE_LOW
        else:
            penalty = G.PENALTY_SUBTYPE_HIGH
        return penalty

    # =================================================================================#
    # CONSTRAINTS
    # =================================================================================#
    def __add_constraints(self):
        '''
        Add constraints to the model
        '''
        self.__constrAssignRot()
        self.__constrRotCompatib()

    def __constrAssignRot(self):
        '''
        Constraint 1: Rotations must be assigned only to one aircraft, or unassigned
        '''
        self.model.constrAssignRot = pyo.ConstraintList()
        for rotation in self.model.setRotations:
            expr = sum(self.model.dvRotAc[rt,ac] for (rt,ac) in self.model.setRotAc if rt==rotation)
            expr = expr + self.model.dvRotUnassign[rotation]
            if isinstance(expr, int)==0:
                self.model.constrAssignRot.add(expr == 1)


    def __constrRotCompatib(self):
        '''
        Constraint 2: Only compatible rotations can be assigned to an aircraft
        '''
        self.model.constrCompatibleRot = pyo.ConstraintList()
        for (rotation1, aircraft) in self.model.setRotAc:
            for rotation2 in [rot for [rot, ac] in self.model.setRotAc if (ac==aircraft) and (rot!=rotation1)]:
                expr_left = self.model.dvRotAc[rotation1, aircraft] + self.model.dvRotAc[rotation2, aircraft] - 1
                expr_right = self.matrix_compatibility.loc[rotation1.id, rotation2.id]
                self.model.constrCompatibleRot.add(expr_left <= expr_right)


    # =================================================================================#
    # SOLUTION
    # =================================================================================#
    def solve(self, solver=None):
        start_time = timeit.default_timer()
        if solver == None:
            solver = self.__find_solver()
        solver = pyo.SolverFactory(solver)
        # Specify allowed gap
        solver.options['mipgap'] = G.TAIL_ASSIGNMENT_ALLOWED_GAP
        if getattr(G, 'SOLVER_THREADS', None) is not None:
            solver.options['threads'] = G.SOLVER_THREADS
        results = solver.solve(self.model, tee=False)

        self.time_solve = timeit.default_timer() - start_time
        log_info('\n##### TAIL ASSIGNMENT #####')
        log_info('Time initialization:', self.time_initialization, '\nTime solution:', self.time_solve)

        if results.solver.termination_condition == TerminationCondition.infeasible:
            log_error("WARNING: SOLUTION INFEASIBLE")

        # Print solver status # TODO Add if solver status should be checked
        if (results.solver.status == SolverStatus.ok) \
                and (results.solver.termination_condition == TerminationCondition.optimal):
            log_info("Found a feasible and optimal solution")
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            log_error("WARNING: SOLUTION INFEASIBLE")
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
        ''' Print results'''
        for aircraft in self.model.setAircraft:
            rotations = [rt for (rt, ac) in self.model.dvRotAc
                         if ac==aircraft and round(self.model.dvRotAc[rt, ac]())==1]
            # Order rotations
            rotations = sorted(rotations, key=lambda x: x.dep_sched)
            for rotation in rotations:
                time_dep = rotation.dep_sched.strftime('%Y-%m-%d %H:%M:%S')
                time_arr = rotation.arr_sched.strftime('%Y-%m-%d %H:%M:%S')
                print(aircraft.id, rotation.id,' Dep:',time_dep,'Arr:',time_arr)

        for rotation in [rt for rt in self.model.setRotations if round(self.model.dvRotUnassign[rt]())==1]:
                print(rotation.id,' unassigned')


def tail_assignment_pyomo(self):
    scheduler_tail_assignment = TailAssignmentScheduler(self)
    # Optimize
    scheduler_tail_assignment.solve()
    # Print results and check them
    # scheduler_tail_assignment.results() # TODO uncomment line if results should be printed in console
    # scheduler_tail_assignment.check_results()

    return scheduler_tail_assignment



# ================================================================================= #
# DICTIONARIES OF MODULE OPTIONS
# ================================================================================= #
module_tail_assignment_functions = {0: tail_assignment_pyomo,
                                    1: tail_assignment_with_reserve_pyomo}
