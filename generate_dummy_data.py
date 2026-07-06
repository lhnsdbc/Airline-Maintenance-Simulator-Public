import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, time
# Import both classes from the shared file
from data_import.mock_classes import MockDistFit, MockDisruptions

RNG = np.random.default_rng(20260706)
BASE_AIRPORT = 'AMS'
FLEET_SUBTYPE = '772'
FLEET_TYPE = '777'
FLEET_SIZE = 31
AIRPORT_COUNT = 38
ROTATION_COUNT = 127
LEG_COUNT_TARGET = 311
MAINTENANCE_TASK_COUNT = 403

# Define directories
DIRS = [
    "Data/input",
    "Data/input/schedules",
    "Data/input/Engineering_non_recurring_test",
    "Data/input/sac_gnn",
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
registrations = [f'SYN-{i:03d}' for i in range(1, FLEET_SIZE + 1)]
pd.DataFrame({
    'AircraftTypeCodeIATA': [FLEET_SUBTYPE] * FLEET_SIZE,
    'AircraftRegistrationFull': registrations,
    'RegistrationStartDate': ['2015-01-01'] * FLEET_SIZE,
    'RegistrationEndDate': [np.nan] * FLEET_SIZE,
    'SubfleetDetailCode': [FLEET_SUBTYPE] * FLEET_SIZE,
    'AsiaTail': [0] * FLEET_SIZE
}).to_csv('Data/input/AircraftRegistrations.csv', index=False)

# 1.3 Airports.csv
airport_codes = [BASE_AIRPORT] + [f'A{i:02d}' for i in range(1, AIRPORT_COUNT)]
airport_icao = ['EHAM'] + [f'ZZ{i:02d}' for i in range(1, AIRPORT_COUNT)]
country_codes = ['NL'] + [f'X{i % 29:02d}' for i in range(1, AIRPORT_COUNT)]
pd.DataFrame({
    'IataAirportCode': airport_codes,
    'IcaoAirportCode': airport_icao,
    'AirportName': [f'Synthetic Airport {code}' for code in airport_codes],
    'CountryCode': country_codes,
    'DateUntil': [np.nan] * AIRPORT_COUNT
}).to_csv('Data/input/Airports.csv', index=False)

# 1.4 TimeZones.csv
offsets = [60] + RNG.choice([-480, -300, -180, 0, 120, 180, 240, 330, 480, 540], size=AIRPORT_COUNT - 1).tolist()
pd.DataFrame({
    'AirportCode': airport_codes,
    'OffsetMinutes': offsets
}).to_csv('Data/input/TimeZones.csv', index=False)

# 1.5 AirportsCoordinates.csv
pd.DataFrame({
    'AirportICAO': airport_icao,
    'Latitude': [52.3] + np.round(RNG.uniform(-35, 55, AIRPORT_COUNT - 1), 4).tolist(),
    'Longitude': [4.76] + np.round(RNG.uniform(-120, 140, AIRPORT_COUNT - 1), 4).tolist()
}).to_csv('Data/input/AirportsCoordinates.csv', index=False)

# 1.7 TurnAround.csv
pd.DataFrame({
    'AircraftTypeCodeIATA': [FLEET_SUBTYPE],
    'TurnAroundTime': [120],
    'DeparturePrepTime': [30],
    'ArrivalPrepTime': [30]
}).to_csv('Data/input/TurnAround.csv', index=False)

pd.DataFrame({
    'AircraftTypeCodeIATA': [FLEET_SUBTYPE],
    'AircraftTypeName': ['Synthetic Widebody']
}).to_csv('Data/input/AircraftTypeDetails.csv', index=False)

pd.DataFrame({
    'AircraftTypeCodeIATA': [FLEET_SUBTYPE],
    'AircraftRegistrationFull': ['SYN-SPARE'],
    'RegistrationStartDate': ['2015-01-01'],
    'SubfleetDetailCode': [FLEET_SUBTYPE],
    'AsiaTail': [0]
}).to_csv('Data/input/AircraftRegistrations_additional.csv', index=False)

pd.DataFrame({'A_check_id': ['SYN-A'], 'Duration_hours': [12]}).to_csv('Data/input/A-checks.csv', index=False)
pd.DataFrame({'DelayCode': ['SYN'], 'Description': ['Synthetic delay']}).to_csv('Data/input/DelayCodes.csv', index=False)
pd.DataFrame({'ReasonCode': ['SYN'], 'Description': ['Synthetic change reason']}).to_csv('Data/input/FlightChangeReasons.csv', index=False)
pd.DataFrame({'Column': ['synthetic_nr_log']}).to_csv('Data/input/Engineering_non_recurring_col_names.csv', index=False)
pd.DataFrame({'task_id': ['SYN-NR-001'], 'labor_hours': [1.0]}).to_csv(
    'Data/input/Engineering_non_recurring_test/nr_log_001.csv', index=False)

# ==========================================
# 2. Generate Scenario Settings (CSV)
# ==========================================

pd.DataFrame({
    'Id': ['default_run'],
    'Rotations_start': ['2023-01-01'],
    'Slotsnorm_scenario': ['standard'],
    'Aircraft_types': [FLEET_SUBTYPE],
    'Aircraft_remove': [np.nan],
    'Aircraft_additional': [np.nan],
    'Reserves_per_day': [1],
    'AOG_distr': ['distributions_AOG'],
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
# least one (Slot_type='A', Location='H'). These synthetic rows match the scale of the
# larger private scenario without copying its slot definitions.
slot_rows = []
for day in range(7):
    slot_rows.append({
        'Variant': 'standard',
        'Slotnr': day + 1,
        'Subtypes': FLEET_SUBTYPE,
        'Time_start': '08:00:00',
        'Time_end': '20:00:00',
        'Day_start': day,
        'Day_end': day,
        'Cycle_duration': 7,
        'Slot_type': 'TO',
        'Location': 'P' if day % 2 else 'H',
        'Regis_allowed': ','.join(registrations[day::7]),
        'Slot_remarks': 'TO',
    })
for i, day in enumerate([1, 3, 5], start=8):
    slot_rows.append({
        'Variant': 'standard',
        'Slotnr': i,
        'Subtypes': FLEET_SUBTYPE,
        'Time_start': '06:00:00',
        'Time_end': '18:00:00',
        'Day_start': day,
        'Day_end': day,
        'Cycle_duration': 7,
        'Slot_type': 'A',
        'Location': 'H',
        'Regis_allowed': ','.join(registrations[(day - 1)::3]),
        'Slot_remarks': 'A',
    })
pd.DataFrame(slot_rows).to_excel('Data/input/Slots_norm_scenarios.xlsx', index=False)

# ==========================================
# 3. Generate Schedule Pickle
# ==========================================

dates = pd.date_range(start='2023-01-01', periods=7, freq='D')
schedule_data = []

base_legs = np.full(ROTATION_COUNT, 2)
while int(base_legs.sum()) < LEG_COUNT_TARGET:
    idx = int(RNG.integers(0, ROTATION_COUNT))
    if base_legs[idx] < 4:
        base_legs[idx] += 1

for i in range(ROTATION_COUNT):
    date = dates[i % len(dates)]
    rotation_start_dt = date + pd.Timedelta(hours=int(RNG.integers(0, 24)),
                                            minutes=int(RNG.choice([0, 15, 30, 45])))
    rotation_weekday = rotation_start_dt.weekday()
    hov_time_obj = rotation_start_dt.time()
    n_legs = int(base_legs[i])
    route = [BASE_AIRPORT]
    for _ in range(n_legs - 1):
        route.append(str(RNG.choice(airport_codes[1:])))
    route.append(BASE_AIRPORT)
    common = {
        'RotationId': f'SYNROT{i:04d}',
        'RotationHeadStdUtc': hov_time_obj,
        'RotationHeadStdUtcWeekday': rotation_weekday,
        'NumberOfLegs': n_legs,
        'AircraftType': FLEET_SUBTYPE,
        'AircraftOwner': 'SYN',
        'FlightGroup': 'ICA',
        'RotationCancelled': 0,
        'FlightCancelled': 0
    }
    leg_departure = rotation_start_dt
    for leg_idx in range(n_legs):
        duration = int(RNG.triangular(45, 610, 900))
        arrival = leg_departure + pd.Timedelta(minutes=duration)
        schedule_data.append({
            **common,
            'FlightLegId': f'SYNFL{i:04d}{leg_idx + 1}',
            'LegNumber': leg_idx + 1,
            'DepartureAirport': route[leg_idx],
            'ArrivalAirport': route[leg_idx + 1],
            'ScheduledDepartureTimeAtHovUtc': leg_departure.time(),
            'ScheduledDepartureTimeAtHovUtcWeekday': leg_departure.weekday(),
            'ScheduledArrivalTimeAtHovUtc': arrival.time(),
            'ScheduledArrivalTimeAtHovUtcWeekday': arrival.weekday(),
            'ActualBlockTimeDuration': duration
        })
        leg_departure = arrival + pd.Timedelta(minutes=int(RNG.choice([90, 120, 150, 180])))

df_schedule = pd.DataFrame(schedule_data)
for col in ['LegCancelled', 'FlightCancellationTimeUtc', 'ScheduledDepartureTimeLocal', 'ScheduledArrivalTimeLocal']:
    df_schedule[col] = np.nan

with open('Data/input/schedules/schedule_2023-01-01_1weeks', 'wb') as f:
    pickle.dump(df_schedule, f)

# ==========================================
# 3b. Synthetic maintenance policy data
# ==========================================

task_codes = [f'ANON-TASK-{i:04d}' for i in range(1, MAINTENANCE_TASK_COUNT + 1)]
intervals = RNG.choice([94, 125, 188, 250, 300, 375, 438, 563, 750, 1125, 16667],
                       size=MAINTENANCE_TASK_COUNT,
                       p=[0.12, 0.08, 0.14, 0.10, 0.16, 0.12, 0.08, 0.08, 0.05, 0.04, 0.03])
labor = np.round(RNG.lognormal(mean=0.0, sigma=0.75, size=MAINTENANCE_TASK_COUNT), 2)
labor = np.clip(labor, 0.08, 12.84)
skills = RNG.choice(['GROUP_A', 'GROUP_B', 'GROUP_C', 'GROUP_D', 'GROUP_E'], size=MAINTENANCE_TASK_COUNT)
panels = RNG.choice(['NONE'] + [f'AREA{i:03d}' for i in range(1, 106)], size=MAINTENANCE_TASK_COUNT)
policy_df = pd.DataFrame({
    'Task_code': task_codes,
    'Interval': intervals,
    'Labour': labor,
    'Skill': skills,
    'Panel': panels,
})
policy_df.to_csv('Data/input/sac_gnn/Maintenance policy data.csv', index=False)
imputed_df = policy_df.copy()
for day in range(1, 32):
    imputed_df[str(day)] = np.round(RNG.uniform(0, imputed_df['Interval'], size=MAINTENANCE_TASK_COUNT), 0)
imputed_df.to_csv('Data/input/sac_gnn/final_imputed_maintenance_policy.csv', index=False)

# ==========================================
# 4. Generate Required Mock Pickles
# ==========================================

# 4.1 Tasks Deferred Defects (Required to avoid 'Task type not supported' error)
# Use 'CORR' for MEL/NSRE tasks and ensure class names contain 'MEL' or 'NSRE'
df_tasks_dd = pd.DataFrame({
    'ac_fleet': [FLEET_TYPE] * 20,
    'task_type': RNG.choice(['CORR', 'ADHOC'], size=20),
    'deferral_class': RNG.choice(['MEL-C', 'NSRE-10', 'A'], size=20),
    'duration': np.round(RNG.uniform(0.5, 6.0, size=20), 1),
    'labor_sched': np.round(RNG.uniform(1.0, 12.0, size=20), 1),
    'labor_act': np.round(RNG.uniform(1.0, 12.0, size=20), 1),
    'deferral_days': RNG.choice([3, 7, 10, 14, 30], size=20),
    'task_work_type': RNG.choice(['P', 'H'], size=20)
})
with open('Data/pickle/tasks_DD', 'wb') as f:
    pickle.dump(df_tasks_dd, f)

# 4.2 Maintenance Distributions
dist_fleet_config = [
    {
        'fleet': FLEET_TYPE,
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
    {'type': 'HubStation', 'airports': [BASE_AIRPORT], 'TAT_fitted_dist': MockDistFit(), 'TAT_historical': [90, 100, 110, 120, 80] * 200, 'TAT_min': 45},
    {'type': 'OutStation', 'airports': airport_codes[1:], 'TAT_fitted_dist': MockDistFit(), 'TAT_historical': [], 'TAT_min': 45},
    {'type': 'OutStationLongTAT', 'airports': [], 'TAT_fitted_dist': MockDistFit(), 'TAT_historical': [], 'TAT_min': 45}
]
with open('Data/pickle/TAT', 'wb') as f:
    pickle.dump(dist_tat, f)

df_flight_dur = pd.DataFrame({
    'OrigDestAirports': [(row['DepartureAirport'], row['ArrivalAirport']) for row in schedule_data],
    'ActualBlockTimeDuration': [row['ActualBlockTimeDuration'] for row in schedule_data]
})
df_flight_dur['OrigDestAirports'] = [tuple(sorted(x)) for x in df_flight_dur['OrigDestAirports']]
df_flight_dur = df_flight_dur.groupby('OrigDestAirports', as_index=False)['ActualBlockTimeDuration'].mean().round()
df_flight_dur['ActualBlockTimeDuration'] = df_flight_dur['ActualBlockTimeDuration'].astype(int)
with open('Data/pickle/flights_duration', 'wb') as f:
    pickle.dump(df_flight_dur, f)

with open('Data/pickle/delays_outstations', 'wb') as f:
    pickle.dump({'delays_fitted_dist': MockDistFit(), 'probability_no_delay': 0.8, 'delays_historical': []}, f)

with open('Data/pickle/distributions_disruptions_fitted', 'wb') as f:
    pickle.dump(MockDisruptions(), f)

print("Dummy data generated successfully.")
