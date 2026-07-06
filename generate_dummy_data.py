import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, time
# Import both classes from the shared file
from data_import.mock_classes import MockDistFit, MockDisruptions

# Define directories
DIRS = [
    "Data/input",
    "Data/input/schedules",
    "Data/input/Engineering_non_recurring_test",
    "Data/pickle",
    "Data/pickle/AOG",
    "Data/output"
]

for d in DIRS:
    os.makedirs(d, exist_ok=True)

# ==========================================
# 1. Generate CSV Inputs
# ==========================================

# 1.1 AircraftRegistrations.csv
pd.DataFrame({
    'AircraftTypeCodeIATA': ['789', '772'],
    'AircraftRegistrationFull': ['SYN-789A', 'SYN-772A'],
    'RegistrationStartDate': ['2015-01-01', '2010-06-20'],
    'RegistrationEndDate': [np.nan, np.nan],
    'SubfleetDetailCode': ['789', '772'],
    'AsiaTail': [0, 0]
}).to_csv('Data/input/AircraftRegistrations.csv', index=False)

# 1.3 Airports.csv
pd.DataFrame({
    'IataAirportCode': ['AMS', 'JFK', 'DXB'],
    'IcaoAirportCode': ['EHAM', 'KJFK', 'OMDB'],
    'AirportName': ['Amsterdam Airport', 'Kennedy Airport', 'Dubai Airport'],
    'CountryCode': ['NL', 'US', 'AE'],
    'DateUntil': [np.nan, np.nan, np.nan]
}).to_csv('Data/input/Airports.csv', index=False)

# 1.4 TimeZones.csv
pd.DataFrame({
    'AirportCode': ['AMS', 'JFK', 'DXB'],
    'OffsetMinutes': [60, -300, 240]
}).to_csv('Data/input/TimeZones.csv', index=False)

# 1.5 AirportsCoordinates.csv
pd.DataFrame({
    'AirportICAO': ['EHAM', 'KJFK', 'OMDB'],
    'Latitude': [52.3, 40.6, 25.2],
    'Longitude': [4.76, -73.7, 55.3]
}).to_csv('Data/input/AirportsCoordinates.csv', index=False)

# 1.7 TurnAround.csv
pd.DataFrame({
    'AircraftTypeCodeIATA': ['789', '772'],
    'TurnAroundTime': [90, 120],
    'DeparturePrepTime': [30, 30],
    'ArrivalPrepTime': [30, 30]
}).to_csv('Data/input/TurnAround.csv', index=False)

# ==========================================
# 2. Generate Scenario Settings (CSV)
# ==========================================

pd.DataFrame({
    'Id': ['default_run'],
    'Rotations_start': ['2023-01-01'],
    'Slotsnorm_scenario': ['standard'],
    'Aircraft_types': ['789, 772'],
    'Reserves_per_day': [1],
    'AOG_distr': [np.nan],
    'maint_sched_constr_clean': [1],
    'maint_sched_constr_wp_anticipation': [1],
    # Force these to standard Python integers
    'clean_target': [int(100)],
    'wp_anticipation_target': [int(5)]
}).to_csv('Data/input/Scenarios_simulation.csv', index=False)

# ==========================================
# 2b. Slots_norm_scenarios: turnaround (TO) + weekly hangar A-check slots
# ==========================================
# NR maintenance only fires on executed *hangar A-check* slots, so the mock must define at
# least one (Slot_type='A', Location='H'). The two TO rows mirror the prior mock; the two A
# rows give each subtype a weekly hangar A-check the maintenance scheduler can populate.
pd.DataFrame([
    {'Variant': 'standard', 'Slotnr': 1, 'Subtypes': '789, 772', 'Time_start': '08:00:00',
     'Time_end': '20:00:00', 'Day_start': 0, 'Day_end': 0, 'Cycle_duration': 7,
     'Slot_type': 'TO', 'Location': 'H', 'Regis_allowed': 'SYN-789A,SYN-772A', 'Slot_remarks': 'TO'},
    {'Variant': 'standard', 'Slotnr': 2, 'Subtypes': '789', 'Time_start': '22:00:00',
     'Time_end': '06:00:00', 'Day_start': 1, 'Day_end': 2, 'Cycle_duration': 7,
     'Slot_type': 'TO', 'Location': 'P', 'Regis_allowed': 'SYN-789A', 'Slot_remarks': 'TO'},
    {'Variant': 'standard', 'Slotnr': 3, 'Subtypes': '789', 'Time_start': '06:00:00',
     'Time_end': '18:00:00', 'Day_start': 3, 'Day_end': 3, 'Cycle_duration': 7,
     'Slot_type': 'A', 'Location': 'H', 'Regis_allowed': 'SYN-789A', 'Slot_remarks': 'A'},
    {'Variant': 'standard', 'Slotnr': 4, 'Subtypes': '772', 'Time_start': '06:00:00',
     'Time_end': '18:00:00', 'Day_start': 5, 'Day_end': 5, 'Cycle_duration': 7,
     'Slot_type': 'A', 'Location': 'H', 'Regis_allowed': 'SYN-772A', 'Slot_remarks': 'A'},
]).to_excel('Data/input/Slots_norm_scenarios.xlsx', index=False)

# ==========================================
# 3. Generate Schedule Pickle
# ==========================================

dates = pd.date_range(start='2023-01-01', periods=7, freq='D')
schedule_data = []

for i, date in enumerate(dates):
    rotation_start_dt = date + pd.Timedelta(hours=10)
    rotation_weekday = rotation_start_dt.weekday()
    hov_time_obj = rotation_start_dt.time()

    common = {
        'RotationId': f'ROT{i}',
        'RotationHeadStdUtc': hov_time_obj,
        'RotationHeadStdUtcWeekday': rotation_weekday,
        'NumberOfLegs': 2,
        'AircraftType': '789',
        'AircraftOwner': 'KL',
        'FlightGroup': 'ICA',
        'RotationCancelled': 0,
        'FlightCancelled': 0
    }

    # Leg 1: AMS -> JFK
    schedule_data.append({
        **common,
        'FlightLegId': f'FL{i}A',
        'LegNumber': 1,
        'DepartureAirport': 'AMS',
        'ArrivalAirport': 'JFK',
        'ScheduledDepartureTimeAtHovUtc': hov_time_obj,
        'ScheduledDepartureTimeAtHovUtcWeekday': rotation_weekday,
        'ScheduledArrivalTimeAtHovUtc': (date + pd.Timedelta(hours=18)).time(),
        'ScheduledArrivalTimeAtHovUtcWeekday': rotation_weekday,
        'ActualBlockTimeDuration': 480
    })

    # Leg 2: JFK -> AMS
    schedule_data.append({
        **common,
        'FlightLegId': f'FL{i}B',
        'LegNumber': 2,
        'DepartureAirport': 'JFK',
        'ArrivalAirport': 'AMS',
        'ScheduledDepartureTimeAtHovUtc': (date + pd.Timedelta(hours=20)).time(),
        'ScheduledDepartureTimeAtHovUtcWeekday': rotation_weekday,
        'ScheduledArrivalTimeAtHovUtc': (date + pd.Timedelta(hours=27)).time(),
        'ScheduledArrivalTimeAtHovUtcWeekday': (rotation_weekday + 1) % 7,
        'ActualBlockTimeDuration': 420
    })

df_schedule = pd.DataFrame(schedule_data)
for col in ['LegCancelled', 'FlightCancellationTimeUtc', 'ScheduledDepartureTimeLocal', 'ScheduledArrivalTimeLocal']:
    df_schedule[col] = np.nan

with open('Data/input/schedules/schedule_2023-01-01_1weeks', 'wb') as f:
    pickle.dump(df_schedule, f)

# ==========================================
# 4. Generate Required Mock Pickles
# ==========================================

# 4.1 Tasks Deferred Defects (Required to avoid 'Task type not supported' error)
# Use 'CORR' for MEL/NSRE tasks and ensure class names contain 'MEL' or 'NSRE'
df_tasks_dd = pd.DataFrame({
    'ac_fleet': ['787', '787', '777'],
    'task_type': ['CORR', 'CORR', 'ADHOC'],
    'deferral_class': ['MEL-C', 'NSRE-10', 'A'],
    'duration': [2.0, 2.0, 2.0],
    'labor_sched': [4.0, 4.0, 4.0],
    'labor_act': [4.0, 4.0, 4.0],
    'deferral_days': [10, 10, 10],
    'task_work_type': ['P', 'H', 'P']
})
with open('Data/pickle/tasks_DD', 'wb') as f:
    pickle.dump(df_tasks_dd, f)

# 4.2 Maintenance Distributions
dist_fleet_config = [
    {
        'fleet': '787',
        'time_between_AOG_fitted': MockDistFit(),
        'AOG_duration_fitted': MockDistFit(),
        'labor_fitted': MockDistFit(),
        'probability_NR': 0.3,
        'inter_arrival_time_probabilities': {5: 1.0},
        'dd_count_probabilities': {1: 1.0}
    },
    {
        'fleet': '777',
        'time_between_AOG_fitted': MockDistFit(),
        'AOG_duration_fitted': MockDistFit(),
        'labor_fitted': MockDistFit(),
        'probability_NR': 0.3,
        'inter_arrival_time_probabilities': {5: 1.0},
        'dd_count_probabilities': {1: 1.0}
    }
]

with open('Data/pickle/AOG/distributions_AOG', 'wb') as f:
    pickle.dump(dist_fleet_config, f)
with open('Data/pickle/distributions_NR', 'wb') as f:
    pickle.dump(dist_fleet_config, f)
with open('Data/pickle/distributions_DD', 'wb') as f:
    pickle.dump(dist_fleet_config, f)

# 4.3 TAT & Flights
dist_tat = [
    {'type': 'HubStation', 'airports': ['AMS'], 'TAT_fitted_dist': MockDistFit(), 'TAT_historical': [90, 100, 110, 120, 80] * 200, 'TAT_min': 45},
    {'type': 'OutStation', 'airports': ['JFK', 'DXB'], 'TAT_fitted_dist': MockDistFit(), 'TAT_historical': [], 'TAT_min': 45},
    {'type': 'OutStationLongTAT', 'airports': [], 'TAT_fitted_dist': MockDistFit(), 'TAT_historical': [], 'TAT_min': 45}
]
with open('Data/pickle/TAT', 'wb') as f:
    pickle.dump(dist_tat, f)

df_flight_dur = pd.DataFrame({
    'OrigDestAirports': [('AMS', 'JFK'), ('JFK', 'AMS')],
    'ActualBlockTimeDuration': [480, 420]
})
with open('Data/pickle/flights_duration', 'wb') as f:
    pickle.dump(df_flight_dur, f)

with open('Data/pickle/delays_outstations', 'wb') as f:
    pickle.dump({'delays_fitted_dist': MockDistFit(), 'probability_no_delay': 0.8, 'delays_historical': []}, f)

with open('Data/pickle/distributions_disruptions_fitted', 'wb') as f:
    pickle.dump(MockDisruptions(), f)

print("Dummy data generated successfully.")
