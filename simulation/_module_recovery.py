from config import G
import timeit
import pyomo.environ as pyo
from datetime import timedelta
from datetime import datetime as dt
from pyomo.opt import SolverStatus, TerminationCondition
from classes.classes_operations import Rotation
from classes.classes_maintenance import Slot
import logging
from pyomo.util.infeasible import log_infeasible_constraints
from output.output_functions import log_info, log_warning, log_error



class LongAOGDutyIncompatibility(Exception):
    '''
    Error class raised when a long AOG is incompatible with a duty that fall out of the recovery window. In order
    to fix this incompatibility, the tail assignment module must be called before the recovery module can be called.
    '''
    pass


class Node:
    '''
    Class definition for a network node.
    Parameter balance must be equal to:
        1: origin node
        0: middle node
        -1: terminal node
    '''
    def __init__(
            self,
            id,
            time,
            balance,
            arcs_orig,
            arcs_termin,
            aircraft = None
    ):
        self.id = id
        self.time = time
        self.balance = balance
        self.aircraft = aircraft

        self.arcs_orig = arcs_orig
        self.arcs_termin = arcs_termin


class GroundArc:
    ''' Class definition for ground arcs.'''
    def __init__(
            self,
            id,
            node_start,
            node_end,
            aircraft
    ):
        self.id = id
        self.node_start = node_start
        self.node_end = node_end
        self.aircraft = [aircraft]

class MaintenanceArc:
    ''' Class of maintenance arcs'''
    def __init__(
            self,
            id,
            node_start,
            node_end,
            slot
    ):
        self.id = id
        self.node_start = node_start
        self.node_end = node_end
        self.slot = slot

class DelayedRotation:
    def __init__(
            self,
            id,
            rotation,
            delay,
            dep_time,
            arr_time
    ):
        self.id = id
        self.rotation = rotation
        self.delay = delay
        self.dep_time = dep_time
        self.arr_time = arr_time

class DelayedSlot:
    def __init__(
            self,
            id,
            slot,
            delay,
            start_time,
            end_time
    ):
        self.id = id
        self.slot = slot
        self.delay = delay
        self.start_time = start_time
        self.end_time = end_time


class DisruptionRecoveryOptimizer:
    def __init__(
            self,
            simulation):
        self.simulation = simulation
        self.name = 'disruption_recovery' + self.simulation.now.strftime('%Y/%m/%d_%H:%M_')\
                    + dt.today().strftime('%Y%m%d%H%M%S%f')
        self.recovery_window_start = self.simulation.now
        self.recovery_window_end = self.simulation.now + timedelta(days=G.RECOVERY_WITHIN_DAYS)
        self.__id_count_node = -1
        self.__id_count_ground_arc = -1
        self.__id_count_maintenance_arc = -1

        # Init time
        start_time = timeit.default_timer()

        # Init model
        self.model = pyo.ConcreteModel(name='disruption_recovery')

        # mid_time = timeit.default_timer()
        self.__add_sets()
        #print('TOTAL ADDING SETS:', timeit.default_timer() - mid_time)

        # mid_time = timeit.default_timer()
        self.__add_decision_variables()
        #print('TOTAL ADDING DECISION VARIABLES:', timeit.default_timer() - mid_time)

        # mid_time = timeit.default_timer()
        self.__add_objective()
        #print('TOTAL ADDING OBJECTIVE:', timeit.default_timer() - mid_time)

        # mid_time = timeit.default_timer()
        self.__add_constraints()
        #print('TOTAL ADDING CONSTRAINTS:', timeit.default_timer() - mid_time)

        self.time_initialization = timeit.default_timer() - start_time

    def __get_node_id(self):
        self.__id_count_node += 1
        return self.__id_count_node

    def __get_ground_arc_id(self):
        self.__id_count_ground_arc += 1
        return self.__id_count_ground_arc

    def __get_maintenance_arc_id(self):
        self.__id_count_maintenance_arc += 1
        return self.__id_count_maintenance_arc

    # TODO Currently all aircraft are allowed on every route. The model could be improved by reducing the allowed
    #  pairs (rotation, aircraft)

    # =================================================================================#
    # SETS
    # =================================================================================#
    def __add_sets(self):
        ''' Add sets to the model'''
        # Full sets from simulation
        self.model.set_aircraft = pyo.Set(initialize=self.__init_set_aircraft())
        self.model.set_slots_full = pyo.Set(initialize=self.__init_set_slots_flex_full())
        self.model.set_slots_TO = pyo.Set(initialize=self.__init_set_slots_TO())
        self.model.set_slots_free = pyo.Set(initialize=self.__init_set_slots_free())

        # Rotations
        # mid_time = timeit.default_timer()
        self.model.set_rotations = pyo.Set(initialize=self.__init_set_rotations())
        self.model.set_rotations_delayed = pyo.Set(initialize=self.__init_set_rotations_delayed())
        #print('\tSETS Rotations :', timeit.default_timer() - mid_time)


        # Origin and termination nodes
        self.model.set_nodes_orig = pyo.Set(initialize=self.__init_set_nodes_orig())
        self.model.set_nodes_termin = pyo.Set(initialize=self.__init_set_nodes_termin())
        # Check for long AOG-duty incompatibility
        self.__check_AOG_duty_incompatibility()

        # Slots
        # mid_time = timeit.default_timer()
        self.model.set_slots = pyo.Set(initialize=self.__init_set_slots_flex())
        self.model.set_slots_flex_delayed = pyo.Set(initialize=self.__init_set_slots_flex_delayed())
        self.model.set_slots_flex_swaps = pyo.Set(initialize=self.__init_set_slots_flex_swaps(), dimen=2)
        #print('\tSETS Slots :', timeit.default_timer() - mid_time)

        # Combination sets
        # mid_time = timeit.default_timer()
        self.model.set_rotation_aircraft = pyo.Set(initialize=self.__init_set_rotation_aircraft(), dimen=2)
        self.model.set_rotation_delayed_aircraft = pyo.Set(initialize=self.__init_set_rotation_delayed_aircraft(),dimen=2)
        self.model.set_rotation_aircraft_original = pyo.Set(initialize=self.__init_set_rotation_aircraft_original(),dimen=2)
        # print('\tSETS Combine Rotation-Ac :', timeit.default_timer() - mid_time)

        # Central nodes
        # mid_time = timeit.default_timer()
        self.model.set_nodes_central = pyo.Set(initialize=self.__init_set_nodes_central())
        self.model.set_nodes = self.model.set_nodes_central | self.model.set_nodes_orig | self.model.set_nodes_termin
        # print('\tSETA Nodes :', timeit.default_timer() - mid_time)

        # Other Arcs
        # mid_time = timeit.default_timer()
        self.model.set_ground_arcs = pyo.Set(initialize=self.__init_set_ground_arcs())
        if G.RECOVERY_INCLUDE_MAINTENANCE_ARCS == 1:
            self.model.set_maintenance_arcs = pyo.Set(initialize=self.__init_set_maintenance_arcs())
        else:
            self.model.set_maintenance_arcs = pyo.Set(initialize=[])
        #print('\tSETS Arcs :', timeit.default_timer() - mid_time)


        # mid_time = timeit.default_timer()
        self.model.set_nodes_aircraft = pyo.Set(initialize=self.__init_set_nodes_aircraft(), dimen=2)
        self.model.set_ground_arcs_aircraft = pyo.Set(initialize=self.__init_set_ground_arcs_aircraft(), dimen=2)
        #print('\tSETS Combine Nodes and Arcs - Ac :', timeit.default_timer() - mid_time)


    def __init_set_aircraft(self):
        ''' Set of aircraft that can be used for recovery.'''
        # return [ac for ac in self.simulation.aircraft if ac not in self.simulation.aircraft_reserve]
        return [ac for ac in self.simulation.aircraft]

    # def __init_set_aircraft_reserve(self):
    #     ''' Set of reserve aircraft available at the hub. '''
    #     return [ac for ac in self.simulation.aircraft if ac.id in self.simulation.aircraft_reserve]

    def __init_set_rotations(self):
        '''Return a list of rotations departing within the recovery window, that can be reassigned '''
        rotations = []
        for aircraft in self.model.set_aircraft:
            aircraft_rotations = [rt for rt in aircraft.rotations if rt != aircraft.duty_current
                                  and rt.arr_sched <= self.recovery_window_end]
            rotations = rotations + aircraft_rotations

        return rotations

    def __init_set_rotations_delayed(self):
        ''' Return a list of delayed rotations arcs. The rotation is delayed with respect to the currently
        expected departure time, and it can be delayed by a certain list of allowed delays specified in
        G.DELAYS_ROTATIONS. The maximum value in the list also establishes the maximum delay that a rotation can be
        assigned with respect to its originally scheduled departure time. '''
        rotations_delayed = []
        max_delay = max(G.DELAYS_ROTATIONS)
        for rotation in self.model.set_rotations:
            # Max departure allowed for the rotation
            dep_max = rotation.dep_sched + timedelta(minutes=max_delay)
            for delay in G.DELAYS_ROTATIONS:
                # Compute departure delay with respect to currently expected departure time
                dep_time = rotation.dep_act + timedelta(minutes=delay)
                # Rotation can be delayed only if it does not exceed maximum departure time
                if dep_time <= dep_max:
                    arr_time = rotation.arr_act + timedelta(minutes=delay)
                    # Compute delay with respect to scheduled departure time
                    delay_total = dep_time - rotation.dep_sched
                    rotation_delayed = DelayedRotation(id=rotation.id+'delay'+str(delay), rotation=rotation,
                                                       delay=delay_total, dep_time=dep_time,
                                                       arr_time=arr_time)
                    rotations_delayed.append(rotation_delayed)

        return rotations_delayed

    def __init_set_slots_flex_full(self):
        ''' Return list of slots within the recovery window. These slots can be delayed and can be swapped.'''
        slots = []
        for aircraft in self.model.set_aircraft:
            aircraft_slots = [sl for sl in aircraft.slots if sl != aircraft.duty_current
                              and sl.dateEnd_init <= self.recovery_window_end
                              and sl.remarks!='AG'
                              ]
            slots = slots + aircraft_slots

        return slots

    def __init_set_slots_TO(self):
        ''' Return the subset of slots that are TO slots'''
        return [sl for sl in self.model.set_slots_full
                if sl.remarks == 'TO']

    def __init_set_slots_free(self):
        ''' Returns a list of slots within the recovery window that are not assigned to any aircraft'''
        return [sl for sl in self.simulation.slots_scheduling
                if sl.aircraft == None
                and sl.remarks == 'TO'
                and sl.dateEnd_init <= self.recovery_window_end
                and sl.dateStart_init >= self.recovery_window_start]

    def __init_set_slots_flex(self):
        ''' Return list of slots than fall within recovery window of their aircraft.'''
        return [sl for sl in self.model.set_slots_full
                if self.__arc_aircraft_node_compatibility(sl, sl.aircraft) == True]

    def __init_set_slots_flex_delayed(self):
        ''' Return a list of delayed maitnenance slots. The delay is computed with respect to the currently expected
         departure time, using the values chosen in G.DELAYS_SLOTS. In any case, a slot cannot be delayed by more
         than the maximum delay in the delay list with respect to the originally scheduled start time. Furthermore,
         the delayed slot must fall within the recovery window of its aircraft. '''
        slots_delayed = []
        delay_max = max(G.DELAYS_SLOTS)
        for slot in self.model.set_slots_full:
            start_max = slot.dateStart_init + timedelta(minutes=delay_max)
            for delay in G.DELAYS_SLOTS:
                start_time = slot.dateStart_final + timedelta(minutes=delay)
                if start_time <= start_max:
                    end_time = slot.dateEnd_final + timedelta(minutes=delay)
                    delay_total = start_time - slot.dateStart_init
                    slot_delayed = DelayedSlot(id=slot.id +'delay'+str(delay), slot=slot,
                                               delay=delay_total, start_time=start_time, end_time=end_time)

                    # Check that delayed slot fits within the recovery window of its aircraft
                    if self.__arc_aircraft_node_compatibility(slot_delayed, slot.aircraft) == True:
                        slots_delayed.append(slot_delayed)

        return slots_delayed

    def __init_set_slots_flex_swaps(self):
        '''
        Return a list of alternatives for maintenance slot swap. The swap is possible when:
            - the two slots have the same location (H/P)
            - both slots are TO slots
            - the two slots are maximum G.SLOT_SWAP_MAX_DAYS apart from each other
            - the planned labor hours and duration of both swapping slots fit in the other slot
            - no task goes due before the new slot is reached
            - the final assignment is feasible with the aircraft origin and termination node
        '''
        set_slots_swaps = []
        slots_allowed_swap = [sl for sl in self.model.set_slots_TO | self.model.set_slots_free]
        for slot in [sl for sl in self.model.set_slots_TO]:
            # Filtering must be done in steps otherwise debugging mode will not run.
            swap_candidates = [sl for sl in slots_allowed_swap
                               if sl!=slot and (sl, slot) not in set_slots_swaps]
            # Subtype, location, slot type
            swap_candidates = [sl for sl in swap_candidates
                               if sl.subtype == slot.subtype
                               and sl.location == slot.location
                               and sl.remarks == slot.remarks]
            # Swap window
            swap_candidates = [sl for sl in swap_candidates
                               if abs((sl.dateStart_final.date() - slot.dateStart_final.date()).days) <= G.SLOT_SWAP_MAX_DAYS]
            # Max labor
            swap_candidates = [sl for sl in swap_candidates
                               if slot.scheduled_work_labor <= sl.laborMax
                                and (sl in self.model.set_slots_free
                                     or sl.scheduled_work_labor <= slot.laborMax)]
            # Max duration
            swap_candidates = [sl for sl in swap_candidates
                               if slot.scheduled_work_duration <= sl.duration_init
                               and (sl in self.model.set_slots_free
                                    or sl.scheduled_work_duration <= slot.duration_init)]
            # Work package due date
            swap_candidates = [sl for sl in swap_candidates
                               if slot.workpackage_due_date >= sl.dateStart_final
                               and (sl in self.model.set_slots_free
                                    or sl.workpackage_due_date >= slot.dateStart_final)]

            # Slots are included in the recovery window of the given aircraft
            swap_candidates = [sl for sl in swap_candidates
                               if self.__arc_aircraft_node_compatibility(sl, slot.aircraft)==True
                               and (sl in self.model.set_slots_free
                                    or self.__arc_aircraft_node_compatibility(slot, sl.aircraft)==True)]

            # Append swap alternative if not already in list
            for slot_swap in swap_candidates:
                if (slot, slot_swap) not in set_slots_swaps:
                    set_slots_swaps.append((slot, slot_swap))
                    set_slots_swaps.append((slot_swap, slot))

        return set_slots_swaps

    def __find_central_nodes_times(self):
        ''' Find a list of times for the central nodes '''
        # List of nodes time as start and end time of rotations and slots (with buffer)
        times_departure = list(set([rt.dep_act - timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)
                                    for rt in self.model.set_rotations]))
        times_arrival = list(set([rt.arr_act + timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)
                                  for rt in self.model.set_rotations]))
        # Delayed rotations
        times_departure_delayed = list(set([rt.dep_time - timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)
                                          for rt in self.model.set_rotations_delayed]))
        times_arrival_delayed = list(set([rt.arr_time + timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)
                                          for rt in self.model.set_rotations_delayed]))
        # Maintenance slots
        times_slot_start = list(set([sl.dateStart_final - sl.towing_time
                                     for sl in self.model.set_slots | self.model.set_slots_free]))
        times_slot_end = list(set([sl.dateEnd_final + sl.towing_time
                                   for sl in self.model.set_slots | self.model.set_slots_free]))

        # Delayed maintenance slots
        times_slot_start_delayed = list(set([sl.start_time - sl.slot.towing_time
                                     for sl in self.model.set_slots_flex_delayed]))
        times_slot_end_delayed = list(set([sl.end_time + sl.slot.towing_time
                                   for sl in self.model.set_slots_flex_delayed]))

        # Put together all the sets found
        times_nodes = list(set(times_departure + times_arrival + times_departure_delayed + times_arrival_delayed +
                               times_slot_start + times_slot_end + times_slot_start_delayed + times_slot_end_delayed))
        return times_nodes


    def __init_set_nodes_central(self): # TODO add reserve nodes and arcs
        ''' Find central nodes in the network given the rotations and slots that need to be accommodated'''
        ##### FIND LIST OF TIMES FOR THE NODES #####
        times_nodes = self.__find_central_nodes_times()

        ##### GENERATE LIST OF NODES
        set_nodes = []
        # For each node, find corresponding originating and terminating rotations and slots
        for node_time in times_nodes:
            ##### ARCS ORIGINIATING OR TERMINATING AT THE NODE #####
            # Rotations
            rot_orig = [rt for rt in self.model.set_rotations
                        if rt.dep_act - timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT) == node_time]
            rot_termin = [rt for rt in self.model.set_rotations
                          if rt.arr_act + timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT) == node_time]
            # Delayed rotations
            rot_delayed_orig = [rt for rt in self.model.set_rotations_delayed
                                if rt.dep_time - timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT) == node_time]
            rot_delayed_termin = [rt for rt in self.model.set_rotations_delayed
                                  if rt.arr_time + timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT) == node_time]

            # Slots
            slots_orig = [sl for sl in self.model.set_slots | self.model.set_slots_free
                          if sl.dateStart_final - sl.towing_time == node_time]
            slots_termin = [sl for sl in self.model.set_slots | self.model.set_slots_free
                            if sl.dateEnd_final + sl.towing_time == node_time]
            # Delayed slots
            slots_delayed_orig = [sl for sl in self.model.set_slots_flex_delayed
                                  if sl.start_time - sl.slot.towing_time == node_time]
            slots_delayed_termin = [sl for sl in self.model.set_slots_flex_delayed
                                    if sl.end_time + sl.slot.towing_time == node_time]

            # Find lists of origin and terminal arcs for nodes
            arcs_orig = rot_orig + rot_delayed_orig + slots_orig + slots_delayed_orig
            arcs_termin = rot_termin + rot_delayed_termin + slots_termin + slots_delayed_termin

            ##### LIST OF AIRCRAFT THAT USE THE NODE #####
            if rot_orig == [] and rot_termin == []:
                ac_rot = []
            else:
                ac_rot = [ac for rt in rot_orig+rot_termin for ac in self.model.set_aircraft
                          if (rt, ac) in self.model.set_rotation_aircraft]

            if rot_delayed_orig == [] and rot_delayed_termin == []:
                ac_rot_delayed = []
            else:
                ac_rot_delayed = [ac for rt in rot_delayed_orig + rot_delayed_termin for ac in self.model.set_aircraft
                                  if (rt, ac) in self.model.set_rotation_delayed_aircraft]

            if slots_orig == [] and slots_termin == []:
                ac_slots = []
            else:
                ac_slots = [sl.aircraft for sl in slots_orig+slots_termin if sl.aircraft != None]

            if slots_delayed_orig == [] and slots_delayed_termin == []:
                ac_slots_delayed = []
            else:
                ac_slots_delayed = [sl.slot.aircraft for sl in slots_delayed_orig + slots_delayed_termin]

            # Slots swap
            slots_swaps = [slots_swap for slots_swap in self.model.set_slots_flex_swaps
                           if slots_swap[1] in slots_orig + slots_termin]
            ac_slots_swap = [sl[0].aircraft for sl in slots_swaps if sl[0].aircraft!=None]

            # Full list of interested aircraft
            aircraft_interested = list(set(ac_rot + ac_rot_delayed + ac_slots + ac_slots_delayed + ac_slots_swap))

            ##### GENERATE NODE #####
            node_id = self.__get_node_id()
            next_node = Node(id=node_id, time=node_time, balance=0,
                             arcs_orig=arcs_orig, arcs_termin=arcs_termin, aircraft=aircraft_interested)
            set_nodes.append(next_node)

        return set_nodes


    def __init_set_nodes_orig(self):
        '''
        Initialize set of origin nodes, which are aircraft-specific, based on the current duty of the aircraft (last
        duty is considered if aircraft has no current duty).
        - If an aircraft has a last or current duty: origin node is set to last or current duty expected arrival +
        TAT/2 or towing time
        - If no last or current duty present: origin node is set to start of recovery window

        If an AOG slot is assigned to an aircraft, the origin node is postponed by the duration of
        the AOG slot. If the origin node then falls after the end of the recovery window, then the origin node is
        anticipated to the end of the recovery window itself.

        The computed node times are always anticipated by one minute to avoid origin and central nodes overlap.
        '''
        nodes_orig  = []
        aircraft_no_orig = []
        for aircraft in self.model.set_aircraft:
            # If the aircraft has an AG slot, then the aircraft is available after the end of the AOG
            # NOTE this done instead of considering orig_node + AOG because uses same constraint as tail assignment.
            #  This requires AOG end date to be updated every time duty end date is updated in simulation.
            AG_slot = next((sl for sl in aircraft.slots if sl.remarks == 'AG'), None)
            if AG_slot != None:
                node_aircraft = AG_slot
            # Find origin duty for each aircraft
            elif aircraft.duty_current != None:
                node_aircraft = aircraft.duty_current
            # Only consider last duty if the minimum turn around time is within recovery window
            elif aircraft.duty_last != None:
                node_aircraft = aircraft.duty_last
            else:
                node_aircraft = None
                aircraft_no_orig.append(aircraft)

            ##### ORIGIN NODE WAS FOUND #####
            if node_aircraft != None:
                # If origin duty found, find the node time
                if isinstance(node_aircraft, Rotation):
                    node_time = node_aircraft.arr_act + timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)
                elif isinstance(node_aircraft, Slot):
                    node_time = node_aircraft.dateEnd_final + node_aircraft.towing_time
                else:
                    raise Exception('Duty type not supported')

                ##### AOG slots #####
                # If the origin node comes after the end of the recovery window (long AOG slots),
                # set the node to the end of the recovery window
                if node_time > self.recovery_window_end:
                    node_time = self.recovery_window_end

                ##### NODE GENARATION #####
                # Anticipate the node time by one minute in order to try to avoid the overlap of orig node and central nodes
                node_time = node_time - timedelta(minutes=1)

                # Generate node
                node_id = 'orig'+str(self.__get_node_id())
                node = Node(id=node_id, time=node_time, balance=1,
                            arcs_orig = [], arcs_termin = [], aircraft=[aircraft])
                nodes_orig.append(node)

        ##### ORIGIN NODE WAS NOT FOUND #####
        # For aircraft with no later origin, create a node at current simulation time
        if aircraft_no_orig != []:
            node_time_general = self.recovery_window_start - timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT+1)
            node_general = Node(id='orig_general', time=node_time_general, balance=1,
                        arcs_orig=[], arcs_termin = [], aircraft = aircraft_no_orig)
            nodes_orig.append(node_general)

        return nodes_orig

    def __init_set_nodes_termin(self):
        '''
        Initialize the set of termination nodes, based on the first duty assigned to the aircraft but not included in
        the recovery action space.
        - If the future duty is found: Termination node is set to the start of the future duty, anticipated by TAT/2
        or towing time.
        - If no future duty found: Termination node is set to the one day after the end of the recovery window. This
        is done to allow the delay of rotations scheduled to arrive back in AMS close to the end of the recovery
        window. If an aircraft is assigned a long AOG slot that is scheduled to end after the end of the recovery
        window, this case also applies.
        '''
        nodes_termin = []

        # If aircraft current duty ends after the end of the recovery window (long AOG), assign it to no termination
        aircraft_no_termin = []
        for aircraft in self.model.set_aircraft:
            AOG_slot = next((sl for sl in aircraft.slots if sl.remarks=='AG'
                             and sl.dateEnd_final>self.recovery_window_end), None)
            if AOG_slot != None:
                aircraft_no_termin.append(aircraft)

        # For each aircraft, find first assigned duty not included in recovery
        for aircraft in [ac for ac in self.model.set_aircraft if ac not in aircraft_no_termin]:
            # Rotations and slots out of recovery window
            ac_rotations = [rt for rt in aircraft.rotations if rt not in self.model.set_rotations
                            and rt != aircraft.duty_current]
            ac_slots = [sl for sl in aircraft.slots if sl not in self.model.set_slots_full
                        and sl != aircraft.duty_current and sl.remarks!='AG']
            # Sort lists
            ac_rotations = sorted(ac_rotations, key=lambda x: x.dep_sched)
            ac_slots = sorted(ac_slots, key=lambda  x: x.dateStart_init)

            if ac_rotations != [] and ac_slots != []:
                if ac_rotations[0].dep_sched < ac_slots[0].dateStart_init:
                    duty_termin = ac_rotations[0]
                else:
                    duty_termin = ac_slots[0]
            elif ac_rotations != []:
                duty_termin = ac_rotations[0]
            elif ac_slots != []:
                duty_termin = ac_slots[0]
            else:
                duty_termin = None
                aircraft_no_termin.append(aircraft)

            # Find corresponding node
            if duty_termin != None:
                # If terminal duty found, find the node time
                if isinstance(duty_termin, Rotation):
                    node_time = duty_termin.dep_act - timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)
                elif isinstance(duty_termin, Slot):
                    node_time = duty_termin.dateStart_final - duty_termin.towing_time
                else:
                    raise Exception('Duty type not supported')

                # Delay node time by one minute to avoid overlap with central nodes with short TAT
                node_time = node_time + timedelta(minutes=1)

                # Generate termination node
                node_id = 'termin' + str(self.__get_node_id())
                node = Node(id=node_id, time=node_time, balance=-1,
                            arcs_orig=[], arcs_termin=[], aircraft=[aircraft])
                nodes_termin.append(node)

        # For aircraft with no later origin, create a node at the end of the recovery window + 1 day.
        # the day is added because the aircraft do not need to be at the airport at the end of the recovery window,
        # if there is no duty scheduled afterwards
        if aircraft_no_termin != []:
            node_time_general = self.recovery_window_end + timedelta(days=1)
            node_general = Node(id='termin_general', time=node_time_general, balance=-1,
                                arcs_orig=[], arcs_termin=[], aircraft=aircraft_no_termin)
            nodes_termin.append(node_general)

        return nodes_termin

    def __check_AOG_duty_incompatibility(self):
        '''
        Check that there is no incompatibility between long AOG and a duty that falls out of the recovery window.
        Do so by checking that for each aircraft the origin node falls before the termination node. Note that despite
        this condition usually applies only in the cases where long AOG exist, if the schedule includes long
        rotationa (possibly due to errors in the data), this can apply too.
        '''
        for aircraft in self.model.set_aircraft:
            # Find origing and termination node
            node_orig = next(nd for nd in self.model.set_nodes_orig if aircraft in nd.aircraft)
            node_termin = next(nd for nd in self.model.set_nodes_termin if aircraft in nd.aircraft)

            if node_orig.time>node_termin.time:
                log_info('TEST CALLING TAIL ASSIGNMENT DURING RECOVERY (RECOVERY MODULE)')
                raise LongAOGDutyIncompatibility()#'Recovery module: Incompatibility between long AOG and fixed duty.')


    def __init_set_nodes_aircraft(self):
        ''' Return set of (node, aircraft) if a node can be covered by an aircraft'''
        return [(node, aircraft)
                for node in self.model.set_nodes
                for aircraft in self.model.set_aircraft
                if aircraft in node.aircraft]

    def __init_set_ground_arcs(self):

        # Function to generate a ground arc
        def generate_ground_arc(node_start, node_end, aircraft):
            ground_arc = GroundArc(id=self.__get_ground_arc_id(), node_start=node_start, node_end=node_end,
                                   aircraft=aircraft)
            node_start.arcs_orig.append(ground_arc)
            node_end.arcs_termin.append(ground_arc)
            return ground_arc

        ground_arcs = []
        for aircraft in self.model.set_aircraft:
            # Find relevant nodes for the aircraft
            node_orig = next(nd for nd in self.model.set_nodes_orig if aircraft in nd.aircraft)
            node_termin = next(nd for nd in self.model.set_nodes_termin if aircraft in nd.aircraft)
            nodes = [nd for nd in self.model.set_nodes_central if aircraft in nd.aircraft]
            # Must order a list before adding the origin and termination slot for those cases where the origin and
            # termination nodes happen at the same time as a central node
            nodes = sorted(nodes, key=lambda x: x.time)
            nodes = [node_orig] + nodes + [node_termin]

            # check that no central node happen before the origin node or the termination node
            if  nodes[1].time < node_orig.time or nodes[-2].time > node_termin.time:
                breakpoint()
                raise Exception('A central node is outside the boundaries of the origin and termination nodes for an aircraft')

            # Generate ground arcs
            for node_index in range(len(nodes)-1):
                node_current = nodes[node_index]
                node_next = nodes[node_index+1]
                # Check if ground arc already exists
                ground_arc_found = next((ga for ga in ground_arcs
                                         if ga.node_start == node_current
                                         and ga.node_end == node_next), None)
                # If ground arc already exists, add aircraft to list of aircraft of ground arc
                if ground_arc_found != None:
                    ground_arc_found.aircraft.append(aircraft)
                # If ground arc not found, generate it
                else:
                    ground_arc_generated = generate_ground_arc(node_start=node_current, node_end=node_next,
                                                               aircraft=aircraft)
                    ground_arcs.append(ground_arc_generated)

        return ground_arcs

    def __init_set_ground_arcs_aircraft(self):
        ''' Return set of ground arc - aircraft pairs'''
        return [(ground_arc, aircraft)
                for ground_arc in self.model.set_ground_arcs
                for aircraft in self.model.set_aircraft
                if aircraft in ground_arc.aircraft]#node_start.aircraft and aircraft in ground_arc.node_end.aircraft]

    def __init_set_maintenance_arcs(self):
        ''' Maintenance arcs allow to move a maintenance TO slot to the available free fleet space. A maintenance slot 
        can be moved to free fleet space when enough time is avaiable to fit its ORIGINALLY SCHEDULED duration''' #
        # TODO now maintenance arcs created with initial slot duration. Can be changed to scheduled duration (task
        #  based)

        maintenance_arcs = []
        # Find list of ordered central nodes
        central_nodes = [nd for nd in self.model.set_nodes_central]
        central_nodes = sorted(central_nodes, key=lambda x: x.time)

        for slot in self.model.set_slots_TO:
            # List of nodes after maintenance slot start
            nodes_after_slot = [nd for nd in central_nodes if nd.time >= slot.dateStart_final
                                and slot.aircraft in nd.aircraft]
            # Add terminal node
            terminal_node = next(nd for nd in self.model.set_nodes_termin if slot.aircraft in nd.aircraft)
            nodes_after_slot.append(terminal_node)
            # Find free fleet space maintenance arcs
            for node_start in nodes_after_slot:
                nodes_end = [nd for nd in nodes_after_slot if nd.time >= node_start.time + slot.duration_init]
                if nodes_end != []:
                    node_end = nodes_end[0]

                    # Maintenance arc should be generated only if there is no overlap with reserve slots in the coming days
                    compatibility_rs = self.__arc_compatible_with_rs_slots(slot.aircraft, node_start.time,node_end.time)
                    if compatibility_rs == False:
                        continue

                    maintenance_arc = MaintenanceArc(id=self.__get_maintenance_arc_id(),
                                                     node_start=node_start, node_end=node_end, slot=slot)
                    maintenance_arcs.append(maintenance_arc)
                    node_start.arcs_orig.append(maintenance_arc)
                    node_end.arcs_termin.append(maintenance_arc)


        return maintenance_arcs

    def __init_set_rotation_aircraft(self):
        ''' A rotation can be assigned to an aircraft if its origin and termination nodes are assigned to the
        aircraft. In the future, specific aircraft-route compatibility might also be implemented. '''
        return [(rotation, aircraft)
                for rotation in self.model.set_rotations
                for aircraft in self.model.set_aircraft
                if self.__arc_aircraft_node_compatibility(rotation,aircraft) == True]

    def __init_set_rotation_delayed_aircraft(self):
        return [(rotation_delayed, aircraft)
                for rotation_delayed in self.model.set_rotations_delayed
                for aircraft in self.model.set_aircraft
                if self.__arc_aircraft_node_compatibility(rotation_delayed, aircraft) == True]

                # Condition changed because if rotation is excluded due to nodes positioning delayed rotations might be included
                #if (rotation_delayed.rotation, aircraft) in self.model.set_rotation_aircraft]

    def __arc_compatible_with_rs_slots(self, aircraft, arc_time_dep, arc_time_arr):
        ''' Given an aircraft, an arc start time and end time, return False if the arc overlaps with at least one
        reserve slots scheduled for in aircraft in the coming days. If no incompatibility is found, return True '''
        # Find the reserve slots of the aircraft assigned for the coming days
        rs_slots_no_overlap = [rs for rs in aircraft.reserve_slots if rs.dep_sched.date() > self.simulation.now.date()]
        # Check if arc overlaps with aircraft RS slots in the coming days
        for rs_slot in rs_slots_no_overlap:
            if self.__is_overlap(rs_slot.dep_sched, rs_slot.arr_sched, arc_time_dep, arc_time_arr):
                return False
        # If no incompatibility found, return True
        return True

    @staticmethod
    def __is_overlap(dep1, arr1, dep2, arr2):
        return dep1 <= arr2 and arr1 >= dep2

    def __arc_compatible_with_anticipation(self, arc, aircraft, time_dep, time_arr):
        ''' If the assignment of an arc changes, a minimum anticipation must be guaranteed. '''
        # Original assignment of duty
        if isinstance(arc, DelayedRotation):
            ac_orig = arc.rotation.aircraft
        elif isinstance(arc, Rotation):
            ac_orig = arc.aircraft
        elif isinstance(arc, Slot):
            ac_orig = arc.aircraft
        elif isinstance(arc, DelayedSlot):
            ac_orig = arc.slot.aircraft
        else:
            raise Exception('Arc type not supported')

        # If assignment does not change, no check must be done
        if ac_orig == aircraft:
            return True

        # MIN ANTICIPATION: ROTATIONS
        if isinstance(arc, DelayedRotation) or isinstance(arc, Rotation):
            # Aircraft is reserve slot
            rs_slot = next((rs for rs in aircraft.reserve_slots
                                  if rs.dep_sched.date() == self.simulation.now.date()), None)
            if rs_slot != None and self.__is_overlap(time_dep, time_arr, rs_slot.dep_sched, rs_slot.arr_sched):
                min_anticipation = G.ASSIGNMENT_CHANGE_MIN_ANTICIPATION['reserve']
            # Aircraft is of same subtype
            elif ac_orig.subtype.IATA == aircraft.subtype.IATA:
                min_anticipation = G.ASSIGNMENT_CHANGE_MIN_ANTICIPATION['subtype_fixed']
            # Aircraft is of different subtype
            else:
                min_anticipation = G.ASSIGNMENT_CHANGE_MIN_ANTICIPATION['subtype_changed']

        # MIN ANTICIPATION: SLOTS
        else:
            min_anticipation = G.ASSIGNMENT_CHANGE_MIN_ANTICIPATION['slot']

        # Make anticipation into timedelta and check if guaranteed
        min_anticipation = timedelta(hours=min_anticipation)
        if time_dep < self.simulation.now + min_anticipation:
            return False

        # If no incompatibility found, assignment is allowed.
        return True

    def __arc_aircraft_node_compatibility(self, arc, aircraft):
        ''' An arc and aircraft are compatible if the origin and termination node of the arc fall between the
        origin and termination node of the aircraft, and if the arc does not overlap with RS slots assigned to the
        aircraft for the next days (the overlap is allowed on the day of operations).'''
        # Find the origin and termination node of the aircraft
        ac_node_orig = next(nd for nd in self.model.set_nodes_orig if aircraft in nd.aircraft)
        ac_node_termin = next(nd for nd in self.model.set_nodes_termin if aircraft in nd.aircraft)

        buffer_flight = timedelta(minutes=G.BUFFER_BEFORE_AFTER_FLIGHT)
        # Find the limiting nodes for the arc

        if isinstance(arc, DelayedRotation):
            time_dep = arc.dep_time - buffer_flight
            time_arr = arc.arr_time + buffer_flight
        elif isinstance(arc, Rotation):
            time_dep = arc.dep_act - buffer_flight
            time_arr = arc.arr_act + buffer_flight
        elif isinstance(arc, Slot):
            time_dep = arc.dateStart_final - arc.towing_time
            time_arr = arc.dateEnd_final + arc.towing_time
        elif isinstance(arc, DelayedSlot):
            time_dep = arc.start_time - arc.slot.towing_time
            time_arr = arc.end_time + arc.slot.towing_time
        else:
            raise Exception('Arc type not supported')

        # Check if arc is compatible with reserve slots of aircraft
        arc_compatible_with_rs_slots = self.__arc_compatible_with_rs_slots(aircraft, time_dep, time_arr)
        if arc_compatible_with_rs_slots == False:
            return False

        # Check if arc assignment change would allow enough time for reassignment. Only check if close departure
        limiting_anticipation = max(G.ASSIGNMENT_CHANGE_MIN_ANTICIPATION.values())
        if time_dep < self.simulation.now + timedelta(hours=limiting_anticipation):
            arc_compatible_with_anticipation = self.__arc_compatible_with_anticipation(arc, aircraft,
                                                                                       time_dep, time_arr)
            if arc_compatible_with_anticipation == False:
                return False


        # Aircraft can execute arc if it is within its origin and termination nodes
        if time_dep >= ac_node_orig.time and time_arr <= ac_node_termin.time:
            return True
        else:
            return False


    def __init_set_rotation_aircraft_original(self):
        return [(rotation, rotation.aircraft)
                for rotation in self.model.set_rotations
                if (rotation, rotation.aircraft) in self.model.set_rotation_aircraft
                or ((rotation, rotation.aircraft) in [(rd.rotation, ac) for (rd, ac) in self.model.set_rotation_delayed_aircraft])]


    # =================================================================================#
    # DECISION VARIABLES
    # =================================================================================#
    def __add_decision_variables(self):
        '''
        Add decision variables to the model
        '''
        ##### ROTATIONS #####
        # Assign rotation to aircraft
        self.model.dv_rotation_aircraft = pyo.Var(self.model.set_rotation_aircraft, domain=pyo.Binary)
        # Assign delayed rotation to aircraft
        self.model.dv_rotation_delayed_aircraft = pyo.Var(self.model.set_rotation_delayed_aircraft, domain=pyo.Binary)
        # Cancel rotation
        self.model.dv_rotation_cancelled = pyo.Var(self.model.set_rotations, domain=pyo.Binary)

        ##### SLOTS #####
        self.model.dv_slot_orig = pyo.Var(self.model.set_slots, domain=pyo.Binary)
        self.model.dv_slot_delayed = pyo.Var(self.model.set_slots_flex_delayed, domain=pyo.Binary)
        self.model.dv_slot_free_fleet_space = pyo.Var(self.model.set_maintenance_arcs, domain=pyo.Binary)
        self.model.dv_slot_swap = pyo.Var(self.model.set_slots_flex_swaps, domain=pyo.Binary)
        self.model.dv_slot_cancelled = pyo.Var(self.model.set_slots_full, domain=pyo.Binary)

        ##### GROUND #####
        self.model.dv_ground_arc_aircraft = pyo.Var(self.model.set_ground_arcs_aircraft, domain=pyo.Binary)

        ##### ARTIFICIAL #####
        self.model.dv_change_rotation_assignment_weight_big = pyo.Var(self.model.set_aircraft, domain=pyo.Binary)
        self.model.dv_change_rotation_assignment_weight_small = pyo.Var(self.model.set_aircraft, domain=pyo.NonNegativeIntegers)


    # =================================================================================#
    # OBJECTIVE
    # =================================================================================#
    def __add_objective(self):
        '''
        Add objective function to the model
        '''
        # NOTE: the find_operating_cost function only returns value zero
        obj_rotation_original = sum(self.__find_operating_cost(rt, ac) * self.model.dv_rotation_aircraft[rt, ac]
                                    for (rt, ac) in self.model.set_rotation_aircraft)
        obj_rotation_delayed = sum((self.__find_operating_cost(rd.rotation,ac) + self.__find_rotation_delay_cost(rd))
                                   * self.model.dv_rotation_delayed_aircraft[rd,ac]
                                   for (rd,ac) in self.model.set_rotation_delayed_aircraft)
        obj_rotation_cancelled = sum(G.WEIGHT_ROTATION_CANCELLED * self.model.dv_rotation_cancelled[rt]
                                     for rt in self.model.set_rotations)
        # NOTE: weight of original slots is now set to zero
        obj_slot_original = sum(G.WEIGHT_ORIGINAL_SLOTS * self.model.dv_slot_orig[sl]
                                for sl in self.model.set_slots)
        obj_slot_delayed = sum((G.WEIGHT_ORIGINAL_SLOTS + self.__find_slot_delay_cost(sd))
                                * self.model.dv_slot_delayed[sd]
                               for sd in self.model.set_slots_flex_delayed)
        # Multiply by 0.5 to account for the fact that the swap variables are doubled swap(sl1,sl2)==swap(sl2,sl1)
        obj_slot_swap = sum(0.5 * G.WEIGHT_SLOT_SWAP * self.model.dv_slot_swap[sl1,sl2]
                            for (sl1, sl2) in self.model.set_slots_flex_swaps)

        obj_slot_ffs = sum(G.WEIGHT_SLOT_FREE_FLEET_SPACE * self.model.dv_slot_free_fleet_space[ffs]
                           for ffs in self.model.set_maintenance_arcs)
        obj_slot_cancelled = sum(self.__find_weight_slot_cancellation(sl) * self.model.dv_slot_cancelled[sl]
                                 for sl in self.model.set_slots_full)
        # NOTE: weight of ground arcs is now set to zero
        obj_ground_arc = sum(G.WEIGHT_GROUND_ARC * self.model.dv_ground_arc_aircraft[gr,ac]
                             for (gr,ac) in self.model.set_ground_arcs_aircraft)
        obj_rotation_change_assignment_weight_big = sum(G.WEIGHT_ROTATION_CHANGE_ASSIGNMENT_BIG *
                                                        self.model.dv_change_rotation_assignment_weight_big[ac]
                                                        for ac in self.model.set_aircraft)
        obj_rotation_change_assignment_weight_small = sum(G.WEIGHT_ROTATION_CHANGE_ASSIGNMENT_SMALL *
                                                          self.model.dv_change_rotation_assignment_weight_small[ac]
                                                          for ac in self.model.set_aircraft)

        objective = obj_rotation_original + obj_rotation_delayed + obj_rotation_cancelled +\
                    obj_slot_original + obj_slot_delayed + obj_slot_swap + obj_slot_ffs +\
                    obj_ground_arc +\
                    obj_rotation_change_assignment_weight_big + obj_rotation_change_assignment_weight_small +\
                    obj_slot_cancelled

        self.model.obj = pyo.Objective(expr=objective)

    @staticmethod
    def __find_weight_slot_cancellation(slot):
        ''' Find penalty of cancelling a slot based on the slot type '''
        if slot.remarks == 'A':
            cost_slot_cancellation = G.WEIGHT_SLOT_ACHECK_CANCELLED
        elif slot.remarks == 'TO':
            cost_slot_cancellation = G.WEIGHT_SLOT_TO_CANCELLED
        else:
            raise Exception('Slot type not supported')
        return cost_slot_cancellation

    @staticmethod
    def __find_operating_cost(rotation, aircraft):
        '''
        Find the penalty for assigning a rotation to an aircraft of a certain subtype.
        The penalty can be equal to:
        - G.WEIGHT_ROT_OPERATING_COST_FIRST_PREF: assigned to aircraft of same subtype as original
        - G.WEIGHT_ROT_OPERATING_COST_SECOND_PREF: assignment different from original, but within preferred group
        - G.WEIGHT_ROT_OPERATING_COST_THIRD_PREF: any other assignment
        '''
        subtype_orig = rotation.aircraft.subtype.IATA
        # Find second choices for rotation-aircraft assignment
        subtypes_second_choices = next(sc for sc in G.PREFERRED_SUBTYPES_GROUPS if subtype_orig in sc)
        subtypes_second_choices = [st for st in subtypes_second_choices if st!=subtype_orig]
        if aircraft.subtype.IATA == subtype_orig:
            penalty = G.WEIGHT_ROT_OPERATING_COST_FIRST_PREF
        elif aircraft.subtype.IATA in subtypes_second_choices:
            penalty = G.WEIGHT_ROT_OPERATING_COST_SECOND_PREF
        else:
            penalty = G.WEIGHT_ROT_OPERATING_COST_THIRD_PREF
        return penalty


    def __find_rotation_delay_cost(self, rotation_delayed):
        ''' Returns the weight of delaying a rotation given a cost per minute'''
        delay_int = round(rotation_delayed.delay.total_seconds()/60)
        weight = delay_int * G.WEIGHT_ROTATION_DELAYED_PER_MIN
        return weight

    def __find_slot_delay_cost(self, slot_delayed):
        ''' Returns the weight of delaying a rotation as G.OBJ_ROTATION_DELAYED_BASE * (position of delay + 1)'''
        delay_int = round(slot_delayed.delay.total_seconds()/60)
        weight = delay_int * G.WEIGHT_SLOT_DELAYED_PER_MIN
        return weight

    # =================================================================================#
    # CONSTRAINTS
    # =================================================================================#
    def __add_constraints(self):
        '''
        Add constraints to the model
        '''
        # mid_time = timeit.default_timer()
        self.__constr_cover_rotations()
        #print('\tCONSTR Cover rot', timeit.default_timer() - mid_time)

        # mid_time = timeit.default_timer()
        self.__constr_cover_slots()
        #print('\tCONSTR Cover slots', timeit.default_timer() - mid_time)

        # mid_time = timeit.default_timer()
        self.__constr_balance()
        # print('\tCONSTR Balance', timeit.default_timer() - mid_time)

        # mid_time = timeit.default_timer()
        self.__constr_slot_swap()
        #print('\tCONSTR Slot Swap', timeit.default_timer() - mid_time)

        # Constraint swaps free slots
        self.__constr_simple_swap_free_slots()
        # print('\tCONSTR swap free slots', timeit.default_timer() - mid_time)

        # mid_time = timeit.default_timer()
        self.__constr_rotation_assignment_change()
        #print('\tCONSTR Rot assignment change', timeit.default_timer() - mid_time)



    def __constr_cover_rotations(self):
        '''
        ROTATION COVER CONSTRAINT: Rotation must be assigned to an aircraft at its original time, delayed,
        or it must be cancelled
        '''
        self.model.constr_cover_rotations = pyo.ConstraintList()
        for rotation in self.model.set_rotations:
            expr_rotation_orig = sum(self.model.dv_rotation_aircraft[rt,ac]
                        for (rt, ac) in self.model.set_rotation_aircraft if rt==rotation)
            expr_rotation_delayed = sum(self.model.dv_rotation_delayed_aircraft[rt,ac]
                         for (rt, ac) in self.model.set_rotation_delayed_aircraft if rt.rotation==rotation)
            expr_rotation_cancelled = sum(self.model.dv_rotation_cancelled[rt] for rt in self.model.set_rotations
                                          if rt==rotation)
            expr = expr_rotation_orig + expr_rotation_delayed + expr_rotation_cancelled
            self.model.constr_cover_rotations.add(expr == 1)

    def __constr_cover_slots(self):
        '''
        SLOT COVER CONSTRAINT: slot must be executed at its original time, delayed, assigned to a free fleet space,
        or swapped with another maintenance slot.
        '''
        self.model.constr_cover_slots = pyo.ConstraintList()
        for slot in self.model.set_slots_full:
            expr_slot_orig = sum(self.model.dv_slot_orig[sl] for sl in self.model.set_slots if sl==slot)
            expr_slot_delayed = sum(self.model.dv_slot_delayed[sl]
                                    for sl in self.model.set_slots_flex_delayed if sl.slot==slot)
            expr_slot_free_fleet_space = sum(self.model.dv_slot_free_fleet_space[ffs]
                                             for ffs in self.model.set_maintenance_arcs if ffs.slot==slot)
            expr_slot_swap = sum(self.model.dv_slot_swap[sl1,sl2]
                                 for (sl1, sl2) in self.model.set_slots_flex_swaps if sl1==slot)
            expr_slot_cancelled = sum(self.model.dv_slot_cancelled[sl]
                                      for sl in self.model.set_slots_full if sl==slot)
            expr = expr_slot_orig + expr_slot_delayed + expr_slot_free_fleet_space + expr_slot_swap + expr_slot_cancelled
            self.model.constr_cover_slots.add(expr==1)

    def __constr_balance(self):
        '''
        BALANCE CONSTRAINT: flow that enters the node must be equal to flow that exits
        '''

        def find_dv(arc, ac):
            '''
            Return a list of decision variables associated to an arc and aircraft. While the list usually only
            comprises one decision variable, in the case of slot swaps more than one decision variable can be returned.
            '''
            # Ground arc
            if type(arc) == GroundArc and ac in arc.aircraft:
                dv = self.model.dv_ground_arc_aircraft[arc, ac]
            # Delayed rotation
            elif type(arc) == DelayedRotation and (arc,ac) in self.model.set_rotation_delayed_aircraft:
                dv = self.model.dv_rotation_delayed_aircraft[arc, ac]
            # Slot to free fleet space arc
            elif type(arc) == MaintenanceArc and arc.slot.aircraft == ac:
                dv = self.model.dv_slot_free_fleet_space[arc]
            # Rotation original
            elif type(arc) == Rotation and (arc, ac) in self.model.set_rotation_aircraft:
                dv = self.model.dv_rotation_aircraft[arc, ac]
            # Slot delayed
            elif type(arc) == DelayedSlot and arc.slot.aircraft == ac:
                dv = self.model.dv_slot_delayed[arc]
            # Slot original
            elif type(arc) == Slot and arc.aircraft == ac:
                dv = self.model.dv_slot_orig[arc]
            else:
                dv = []

            # Make dv into a list
            if type(dv)!=list:
                dv = [dv]

            # If dv not found, check if slot swap exists
            if dv == []:
                dv = [self.model.dv_slot_swap[slot_init, slot_final]
                      for (slot_init, slot_final) in self.model.set_slots_flex_swaps
                      if slot_init.aircraft == ac and slot_final == arc ]

            return dv

        def sum_arcs(arcs_list, aircraft):
            ''' Given a list of arcs and an aircraft, return the sum of the decision variables associated to the
            given arcs that concern the given aircraft. '''
            dv_list = []
            for arc in arcs_list:
                dv = find_dv(arc, aircraft)
                dv_list.extend(dv)
            expr = sum(dv_list)
            return expr

        # Empty list of constraints
        self.model.constr_balance = pyo.ConstraintList()
        # Add balance constraint at each (node, aircraft pair)
        for (node, aircraft) in self.model.set_nodes_aircraft:
            expr_node_out = sum_arcs(node.arcs_orig, aircraft)
            expr_node_in = sum_arcs(node.arcs_termin, aircraft)
            expr = expr_node_out - expr_node_in

            # Must use a try-except statement because pyomo cannot check the expression expr==0 if dv in expr
            try:
                if expr==0:
                    pass
            except:
                # mid_time = timeit.default_timer()
                self.model.constr_balance.add(expr == node.balance)
                # time_execution = timeit.default_timer() - mid_time
                # print('Time add constraint:', time_execution)


    def __constr_slot_swap(self):
        '''
        SWAP CONSTRAINT: if a swap of slot assignment is made, then both a aircraft must have their assigned slot
        swapped
        '''
        self.model.constr_slots_swap = pyo.ConstraintList()
        # Consider either the pair (slot1, slot2) or (slot2, slot1)
        for (slot1, slot2) in [(sl1, sl2) for (sl1, sl2) in self.model.set_slots_flex_swaps if sl1.id<sl2.id]:
            self.model.constr_slots_swap.add(self.model.dv_slot_swap[slot1,slot2]==self.model.dv_slot_swap[slot2,slot1])

    def __constr_rotation_assignment_change(self):
        '''
        CONSTRAINT ORIGINAL ROTATION ASSIGNMENT: this constraint aims at keeping the rotation assignment fixed to the
        original assignment. An artificial variable is activated whenever one rotation originally assigned to an
        aircraft is not assigned to a different one, or cancelled.
        '''
        self.model.constr_rotation_assignment_change_weight_big = pyo.ConstraintList()
        self.model.constr_rotation_assignment_change_weight_small = pyo.ConstraintList()
        for aircraft in self.model.set_aircraft:
            expr_rotations = sum(self.model.dv_rotation_aircraft[rt, ac]
                                     for (rt, ac) in self.model.set_rotation_aircraft_original
                                     if ac == aircraft and (rt, ac) in self.model.set_rotation_aircraft)
            expr_rotations_delayed = sum(self.model.dv_rotation_delayed_aircraft[rd, ac]
                                         for (rd, ac) in self.model.set_rotation_delayed_aircraft
                                         if ac == aircraft
                                         and (rd.rotation, ac) in self.model.set_rotation_aircraft_original)
            expr_rotations_cancelled = sum(self.model.dv_rotation_cancelled[rt]
                                           for (rt, ac) in self.model.set_rotation_aircraft_original
                                           if ac == aircraft)
            expr_left = expr_rotations + expr_rotations_delayed + expr_rotations_cancelled
            rotations_original = [rt for rt in self.model.set_rotations
                                  if rt.aircraft == aircraft]


            expr_right_big = len(rotations_original) * (1 - self.model.dv_change_rotation_assignment_weight_big[aircraft])
            expr_right_small = len(rotations_original) - self.model.dv_change_rotation_assignment_weight_small[aircraft]

            # Add constraint big weight
            try:
                self.model.constr_rotation_assignment_change_weight_big.add(expr_left >= expr_right_big)
            except:
                # If both the expressions are equal to zero, it means that the aircraft is excluded from recovery
                if expr_right_big == 0 and expr_left == 0:
                    pass
                else:
                    raise Exception('Something is wrong with original rotation assignment constraint (big weight)')

            # Add constraint small weight
            try:
                self.model.constr_rotation_assignment_change_weight_big.add(expr_left >= expr_right_small)
            except:
                # If both the expressions are equal to zero, it means that the aircraft is excluded from recovery
                if expr_right_small == 0 and expr_left == 0:
                    pass
                else:
                    raise Exception('Something is wrong with original rotation assignment constraint (small weight)')

    def __constr_simple_swap_free_slots(self):
        '''
        CONSTRAINT SIMPLE SWAP OF FREE SLOTS: Each free slot can only be swapped with one other slot.
        '''
        self.model.constr_simple_swap_free_slots = pyo.ConstraintList()
        for slot in self.model.set_slots_free:
            expr = sum(self.model.dv_slot_swap[sl1, sl2]
                       for (sl1, sl2) in self.model.set_slots_flex_swaps if sl1 == slot)

            if isinstance(expr, int) == 0:
                self.model.constr_simple_swap_free_slots.add(expr <= 1)

    def solve(self, solver=None):
        start_time = timeit.default_timer()
        if solver == None:
            solver = self.__find_solver()
        solver = pyo.SolverFactory(solver)
        # Specify allowed gap
        solver.options['mipgap'] = G.RECOVERY_MODULE_ALLOWED_GAP
        if getattr(G, 'SOLVER_THREADS', None) is not None:
            solver.options['threads'] = G.SOLVER_THREADS
        results = solver.solve(self.model, tee=False)

        self.time_solve = timeit.default_timer() - start_time
        log_info('\n##### DISRUPTIONS RECOVERY #####')
        log_info('Time initialization:', self.time_initialization, '\nTime solution:', self.time_solve, '\n')

        # Print solver status
        if (results.solver.status == SolverStatus.ok) and (
                results.solver.termination_condition == TerminationCondition.optimal):
            log_info("Found a feasible and optimal solution")
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            log_error('SOLUTION INFEASIBLE')
            # Log info on infeasible model
            logging.basicConfig(filename='infeasible'+self.simulation.id+self.name+'.log', force=True,
                                level=logging.INFO,encoding='utf-8')
            log_infeasible_constraints(self.model, log_expression=True, log_variables=True)
            breakpoint()
        else:
            # something else is wrong
            log_error(str(results.solver))

    @staticmethod
    def __find_solver():
        if isinstance(G.SOLVER, str):
            solver = G.SOLVER
        elif G.SOLVER == 0:
            solver = 'cbc'
        elif G.SOLVER == 1:
            solver = 'gurobi_direct'
        else:
            raise Exception('Optimization solver not supported')
        return solver

    def results(self):
        for rotation in self.model.set_rotations:
            rot_ac = next((ac for (rt,ac) in self.model.dv_rotation_aircraft
                           if rt == rotation
                           and round(self.model.dv_rotation_aircraft[rt,ac]())==1), None)
            rot_delayed_ac = next(((rd, ac) for (rd,ac) in self.model.dv_rotation_delayed_aircraft
                                  if rd.rotation == rotation
                                  and round(self.model.dv_rotation_delayed_aircraft[rd,ac]())==1), None)
            rot_cancelled = next((rt for rt in self.model.dv_rotation_cancelled
                                  if rt == rotation
                                  and round(self.model.dv_rotation_cancelled[rt]())==1), None)

            # check that rotation assigned ones
            found_none = [rot_ac, rot_delayed_ac,rot_cancelled].count(None)
            if found_none != 2:
                raise Exception('Rotation assigned multiple times or unassigned')

            # Rotation has original assignment, on time
            if rot_ac != None and rot_ac == rotation.aircraft:
                print('Rotation '+rotation.id+ ' Assignment unchanged: '+rotation.aircraft.id)
            # Rotation is delayed with original assignment
            elif rot_delayed_ac != None and rot_delayed_ac[1] == rot_delayed_ac[0].rotation.aircraft:
                print('Rotation ' + rotation.id + ' Assignment unchanged: ' + rotation.aircraft.id +\
                      ' Delay [min]: ' + str(rot_delayed_ac[0].delay.total_seconds()/60))
            # Rotation assignment changed
            elif rot_ac != None and rot_ac != rotation.aircraft:
                print('Rotation ' + rotation.id + ' Assign orig: ' +rotation.aircraft.id+ ' Assign final: '+rot_ac.id )
            # Rotation assignment changed and rotation is delayed
            elif rot_delayed_ac != None and rot_delayed_ac[1] != rot_delayed_ac[0].rotation.aircraft:
                print('Rotation ' + rotation.id + ' Assign orig: ' + rotation.aircraft.id +\
                      ' Assign final: ' + rot_delayed_ac[1].id +\
                      ' Delay [min]: ' + str(rot_delayed_ac[0].delay.total_seconds()/60))
            elif rot_cancelled != None:
                print('Rotation ' + rotation.id + ' Assign orig: ' +rotation.aircraft.id+ ' CANCELLED ')
            else:
                raise Exception('Something is wrong')

        for slot in self.model.set_slots_full:
            slot_orig = True if slot in self.model.set_slots and round(self.model.dv_slot_orig[slot]())==1 else None
            slot_cancelled = True if round(self.model.dv_slot_cancelled[slot]())==1 else None
            slot_delayed = next((sd for sd in self.model.dv_slot_delayed
                                if sd.slot == slot
                                and round(self.model.dv_slot_delayed[sd]())==1), None)
            slot_swap = next((ss for ss in self.model.dv_slot_swap
                              if ss[0]==slot
                              and round(self.model.dv_slot_swap[ss]())==1), None)
            slot_ffs = next((ffs for ffs in self.model.dv_slot_free_fleet_space
                             if ffs.slot==slot
                             and round(self.model.dv_slot_free_fleet_space[ffs]())==1), None)
            # check that rotation assigned ones
            found_none = [slot_orig, slot_cancelled, slot_delayed, slot_swap, slot_ffs].count(None)
            if found_none != 4:
                raise Exception('Slot assigned multiple times or unassigned')

            if slot_orig != None:
                print('Slot '+slot.id+ ' Aircraft '+slot.aircraft.id+ ' As scheduled')
            elif slot_cancelled != None:
                print('Slot ' + slot.id + ' Aircraft '+slot.aircraft.id+ ' Cancelled')
            elif slot_delayed != None:
                slot_delay = str(round(slot_delayed.delay.total_seconds()/60))
                print('Slot ' + slot.id + ' Aircraft '+slot.aircraft.id+' Delayed [min] '+slot_delay)
            elif slot_swap != None:
                print('Slot ' + slot.id + ' Aircraft '+slot.aircraft.id+
                      ' Swapped with slot '+slot_swap[1].id+' Aircraft '+slot_swap[1].aircraft.id)
            elif slot_ffs != None:
                print('Slot ' + slot.id + ' Aircraft '+slot.aircraft.id+ ' Moved to free fleet space between '+\
                      slot_ffs.node_start.time.strftime('%Y%m%d_%H%M') + ' and ' +\
                      slot_ffs.node_end.time.strftime('%Y%m%d_%H%M'))
            else:
                raise Exception('Something is wrong')


def disruption_recovery_pyomo(self):
    # Update current time
    self._update_now()
    disruption_recovery = DisruptionRecoveryOptimizer(self)
    disruption_recovery.solve()
    # disruption_recovery.results() # Uncomment for printing results
    return disruption_recovery


# ================================================================================= #
# DICTIONARIES OF MODULE OPTIONS
# ================================================================================= #
module_disruptions_recovery = {0: disruption_recovery_pyomo}
