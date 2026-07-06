from datetime import datetime as dt, timedelta
from geographiclib.geodesic import Geodesic
from config import G
import math
import copy


def find_id(parent, date):
    '''Generate an id for the rotation instance'''
    if isinstance(date, str)==0:
        date_string = dt.strftime(date, '%Y-%m-%d')
    else:
        date_string = date
    rot_id = parent.id_general
    id = date_string + '|' + rot_id
    return id

def find_localized_datetime(date, time):
    ''' Return a UTC localized departure time from the combination of a date and a time'''
    found_datetime = dt.combine(date, time)
    found_datetime = G.TIMEZONE_UTC.localize(found_datetime)
    return found_datetime

def find_delta_days(day1, day2):
    '''Find the number of days in between two weekdays'''
    delta_days = day2 - day1
    if delta_days < 0:
        delta_days += 7
    return delta_days




class Flight_norm:
    def __init__(self,
                 id,
                 id_general,
                 time_dep,
                 time_arr,
                 weekday_dep,
                 weekday_arr,
                 rotation,
                 leg_number,
                 airport_dep,
                 airport_arr,
                 block_time,
                 subtypes=None,
                 ):
        self.id = id
        self.id_general = id_general
        self.time_dep = time_dep
        self.time_arr = time_arr
        self.weekday_dep = weekday_dep
        self.weekday_arr = weekday_arr
        self.rotation = rotation
        self.leg_number = leg_number
        self.airport_dep = airport_dep
        self.airport_arr = airport_arr
        self.block_time = int(block_time)
        self.subtypes = subtypes

        self.cancelled = False
        self.waypoints = []

    def find_waypoints(self):
        ''' Find waypoints between origin and destination airport'''
        # Find route
        geod = Geodesic.WGS84
        gd = geod.Inverse(self.airport_dep.coordinates['latitude'],
                                   self.airport_dep.coordinates['longitude'],
                                   self.airport_arr.coordinates['latitude'],
                                   self.airport_arr.coordinates['longitude'])
        line = geod.Line(gd['lat1'], gd['lon1'], gd['azi1'])
        # Number of points to draw in line
        number_points = math.floor(self.block_time/G.LOG_DISCRETE_TIME_STEP) + 2

        for i in range(1, number_points):
            point = line.Position(gd['s12'] / number_points * i)
            coordinates = {'latitude': point['lat2'],
                           'longitude': point['lon2']}
            self.waypoints.append(coordinates)



class Rotation_norm:
    def __init__(self,
                 id,
                 id_general,
                 n_legs,
                 time_dep,
                 weekday_dep,
                 subtypes=None
                 ):
        self.id = id
        self.id_general = id_general
        self.n_legs = n_legs
        self.time_dep = time_dep
        self.weekday_dep = weekday_dep
        self.subtypes = subtypes

        self.flights = []



class Flight():
    def __init__(self,
                 flight_norm,
                 rotation,
                 dep_sched,
                 arr_sched,
                 aircraft = None
                 ):

        self.id = find_id(flight_norm, dep_sched)
        self.flight_norm = flight_norm
        self.rotation = rotation
        self.dep_sched = dep_sched
        self.arr_sched = arr_sched
        self.aircraft = aircraft

        self.airport_dep = flight_norm.airport_dep
        self.airport_arr = flight_norm.airport_arr
        if G.LOG_DISCRETE_TIME == 1: # Only copy flight waypoints if log should be generated
            self.waypoints = copy.deepcopy(flight_norm.waypoints)
        else:
            self.waypoints = []

        # Initialize actual departure and arrival time
        self.dep_act = dep_sched
        self.arr_act = arr_sched

        # Initialize dictionary of flight delays
        self.delay_primary = 0
        self.delay_reactionary = 0
        self.delay_technical = 0


    def next_waypoint(self):
        wp = self.waypoints.pop(0)
        return wp

class Rotation():
    def __init__(self,
                 rotation_norm,
                 date,
                 aircraft = None
                 ):
        self.id = find_id(rotation_norm,date)
        self.rotation_norm = rotation_norm
        self.dep_sched = find_localized_datetime(date, rotation_norm.time_dep)
        self.aircraft = aircraft

        self.dep_act = self.dep_sched
        self.flights = self.generate_flights()
        # Generate flights instances included in rotation
        if self.flights:
            self.arr_sched = self.flights[-1].arr_sched
            self.arr_act = self.arr_sched
        else:
            self.arr_sched = self.dep_sched  # Fallback value
            self.arr_act = self.dep_sched
        self._val_recovery_prec_assignment = None


    def generate_flights(self):
        ''' Generate flight instances to include in the rotation'''
        flights = []
        try:
            for flight_norm in self.rotation_norm.flights:
                # Find delta days between start of rotation, and departure and arrival of flight
                delta_days_rot_dep = find_delta_days(self.rotation_norm.weekday_dep, flight_norm.weekday_dep)
                delta_days_dep_arr = find_delta_days(flight_norm.weekday_dep, flight_norm.weekday_arr)
                # Find departure and arrival date for flight
                date_dep = (self.dep_sched+timedelta(delta_days_rot_dep)).date()
                date_arr = (date_dep+timedelta(delta_days_dep_arr))
                # Find scheduled departure and arrival for flight
                dep_sched = find_localized_datetime(date_dep, flight_norm.time_dep)
                arr_sched = find_localized_datetime(date_arr, flight_norm.time_arr)
                # Generate flight instance
                flight = Flight(flight_norm=flight_norm,
                                rotation=self,
                                dep_sched=dep_sched,
                                arr_sched=arr_sched)
                # Add flight to rotation
                flights.append(flight)
        except Exception as e:
                # This will print the actual error (e.g., TypeError, ValueError)
                print(f"Error generating flights for rotation {self.id}: {e}")
                # Only keep the breakpoint if you want to inspect this specific failure
                breakpoint()
        # Order flights
        flights = sorted(flights, key=lambda x: x.dep_sched)
        return flights


class ReserveSlot:
    def __init__(self,
                 date,
                 day_count
                 ):
        self.id = 'RS_'+ date.strftime('%Y:%m:%d')+'_'+str(day_count)
        self.dep_sched = self.__replace_date_in_datetime(G.RESERVE_SLOT_START, date)
        self.arr_sched = self.__replace_date_in_datetime(G.RESERVE_SLOT_END, date+timedelta(days=1))

    @staticmethod
    def __replace_date_in_datetime(datetime_old, date_new):
        ''' Given a datetime and a date, modify the date of the datetime to the new date'''
        datetime_new = datetime_old.replace(year = date_new.year,
                                            month = date_new.month,
                                            day = date_new.day)
        return datetime_new


class Airport:
    def __init__(self,
                 id,
                 name,
                 country_code,
                 latitude,
                 longitude
                 ):
        self.id = id
        self.name = name
        self.country_code = country_code
        self.coordinates = {'latitude': latitude,
                            'longitude': longitude}
        self.TAT_sampled = []



