import pandas as pd
from data_import.input_output import load_csv, write_pickle, read_pickle
from config import RUN_CONFIG, P
from datetime import datetime as dt
from datetime import timedelta
from itertools import groupby, product
import numpy as np
from matplotlib import pyplot as plt
import time
from math import floor, ceil
from distfit import distfit
from tqdm import tqdm
import sys
from scipy import stats


class DistributionsDisruptions:
    def __init__(self, disruption_levels, transition_probability_matrix):
        self.disruption_levels = disruption_levels.copy()
        self.transition_probability_matrix = transition_probability_matrix


def find_distributions_network():
    '''
    This function calls all relevant functions to find distributions and data to be used during the simulation.
    Results are saved into pickles.
    '''

    ######### IMPORT DATA #########
    # Import dataframe
    df_flights_ICA = load_csv('historical_flights_reduced', localize_time=True)
    # Only keep ICA flights
    df_flights_ICA = df_flights_ICA[(df_flights_ICA['FlightGroup']=='ICA')
                             & (df_flights_ICA['AircraftType']!='73J')]

    ######### DISRUPTIONS AT OUTSTATIONS #########
    find_outstations_delays(df_flights_ICA)

    ######### TAT #########
    find_TAT_distributions(df_flights_ICA)

    ######## BLOCK TIME #########
    find_flights_duration(df_flights_ICA)

    ######### DISRUPTION AT AMS #########
    find_hub_disruptions()


    plt.show()


def find_outstations_delays(df_flights_ICA):
    '''
    This function finds the distribution o flight delays at outstations.
    The found distributions are saved into a pickle file.
    '''
    ####### FILTER DATA #######
    # Make copy of df
    df_flights = df_flights_ICA.copy(deep=True)
    # Only keep flights until 2019, to remove data from COVID time
    df_flights = df_flights[df_flights['ScheduledDepartureTimeUtc'] <= '31-12-2019']
    # Remove cancelled flights
    df_flights = df_flights[df_flights['FlightCancelled'] == 0]
    # Remove flights operated by B747
    df_flights = df_flights[(df_flights['AircraftType']!='744') & (df_flights['AircraftType']!='74E')]
    # Remove flights departing from AMS
    df_flights = df_flights[df_flights['DepartureAirport']!='AMS']

    # Remove flights with delay code 41-52 (technical) or 93 (reactionary aircraft rotation)
    condition_1 = pd.isnull(df_flights['DelayCode1']) | (
                ((df_flights['DelayCode1'] < 41) | (df_flights['DelayCode1'] > 52)) & (df_flights['DelayCode1'] != 93))
    condition_2 = pd.isnull(df_flights['DelayCode2']) | (
                ((df_flights['DelayCode2'] < 41) | (df_flights['DelayCode2'] > 52)) & (df_flights['DelayCode2'] != 93))
    df_flights = df_flights[condition_1 & condition_2]

    # Make all nan values of delay duration into zeros
    df_flights['DelayCode1Duration'] = df_flights['DelayCode1Duration'].fillna(0)
    df_flights['DelayCode2Duration'] = df_flights['DelayCode2Duration'].fillna(0)
    df_flights['DepartureDelayDuration'] = df_flights['DepartureDelayDuration'].fillna(df_flights['DelayCode1Duration'] + df_flights['DelayCode2Duration'])

    # Exclude flights that leave with an earlier anticipation that a specified value
    df_flights = df_flights[df_flights['DepartureDelayDuration'] >= P.MAX_ANTICIPATION_ALLOWED]

    ####### FLIGHT DELAYS AT OUTSTATIONS DISTRIBUTION #######
    # List of historical delays
    delays_historical = df_flights['DepartureDelayDuration'].to_list()

    # Compute empirical probability of not having delay
    delays_duration_less_equal_zero = [dl for dl in delays_historical if dl<=0]
    probability_no_delay = len(delays_duration_less_equal_zero)/len(delays_historical)

    # Initialize distfit
    dist = distfit(todf=True)

    # Find bins to group empirically observed delays
    min_delay = min(delays_historical)
    max_delay = max(delays_historical)
    n_bins = round((max_delay - min_delay + 1)/ P.DELAYS_DISTRIBUTIONS_BIN_SIZE)
    dist.bins = n_bins

    # Determine best-fitting probability distribution for data
    dist.fit_transform(np.array(delays_historical))

    # Save results in dictionary
    delays_outstations = {'delays_historical': delays_historical,
                          'probability_no_delay': probability_no_delay,
                          'delays_fitted_dist': dist
                          }

    # Print summary and plot obtained fitting
    disr_name = 'Delays at Outstations'
    print('\n\n################# DISRUPTION LEVEL: ', disr_name, ' #################')
    print(dist.summary)
    dist.plot()
    dist.plot_summary()


    ####### SAVE FITTED DISTRIBUTIONS #######
    write_pickle(delays_outstations, 'delays_outstations')

    ####### PRINT HISTOGRAMS #######
    fig, axs = plt.subplots(1)
    fitted_dist_name = dist.model['name']
    plt.suptitle('Delays at outstations - ' + fitted_dist_name)
    max_to_show = 150

    axs.hist(delays_historical, bins=range(floor(min(delays_historical)), max_to_show),
                density=True, label='Historical')
    # Fitted distribution
    x = np.array(range(round(min(delays_historical)), max_to_show))
    y = dist.model['distr'].pdf(x, *dist.model['arg'],
                                      loc=dist.model['loc'],
                                      scale=dist.model['scale'])
    axs.plot(x, y, label='Fitted')
    axs.grid()
    axs.set(xlabel='[min]')

    axs.legend()
    # plt.show()
    print('Delays at outstations fitted ')




def find_TAT_distributions(df_flights_ICA):
    '''
    This function finds distributions for the turn around time at different categories of airport:
    - Hub (AMS)
    - Outstations whit long TAT (average is greater than arbitrary value 80 minutes)
    - Outstations whit short TAT (average is shorter than arbitrary value 80 minutes)

    The found distributions are saved into a pickle file.
    '''
    ####### FILTER DATA #######
    # Make copy of df
    df_flights = df_flights_ICA.copy(deep=True)
    # Only keep flights until 2019, to remove data from COVID time
    df_flights = df_flights[df_flights['ScheduledDepartureTimeUtc'] <= '31-12-2019']
    # Remove cancelled flights
    df_flights = df_flights[df_flights['FlightCancelled'] == 0]
    # Remove flights operated by B747
    df_flights = df_flights[(df_flights['AircraftType']!='744') & (df_flights['AircraftType']!='74E')]
    # In order to isolate turn around time, only keep flights for which delay code 1 is 93 (reactionary),
    # and there is no delay code 2 assigned
    df_flights = df_flights[(df_flights['DelayCode1'] == 93) & (pd.isnull(df_flights['DelayCode2']))]
    # Remove flights for which data on previous leg is not available
    df_flights = df_flights[pd.isnull(df_flights['PreviousLegActualArrivalTimeUtc'])==0]

    ####### PROCESS DATA #######
    # Find turn around time
    df_flights['TAT'] = (df_flights['ActualBlockDepartureTimeUtc'] -
                         df_flights['PreviousLegActualArrivalTimeUtc']).dt.total_seconds()/60
    df_flights['TAT'] = df_flights['TAT'].round().astype(int)
    # Add column with mean delay per airport
    df_flights['TATAirportMean'] = df_flights['TAT'].groupby(df_flights['DepartureAirport']).transform('mean')
    # Divide airports per TAT category
    df_flights['TATType'] = ''
    df_flights.loc[df_flights['DepartureAirport']=='AMS','TATType'] = 'HubStation'
    df_flights.loc[(df_flights['DepartureAirport']!='AMS')&
                   (df_flights['TATAirportMean'] <= P.SEPARATION_SHORT_LONG_TAT),'TATType'] = 'OutStationShortTAT'
    df_flights.loc[(df_flights['DepartureAirport']!='AMS')&
                   (df_flights['TATAirportMean'] > P.SEPARATION_SHORT_LONG_TAT),'TATType'] = 'OutStationLongTAT'

    # Find list of TAT types
    TAT_types_names = list(set(df_flights['TATType'].to_list()))
    TAT_types_names.sort()

    # Check if different aircraft types can be modelled in same distribution
    # Find aircraft type from aircraft subtype
    df_flights['AircraftType_general'] = ''
    df_flights['AircraftType_general'] = df_flights['AircraftType_general'].mask(((df_flights['AircraftType']=='772')|
                                                                                  (df_flights['AircraftType']=='77W')),
                                                                                 '77')
    df_flights['AircraftType_general'] = df_flights['AircraftType_general'].mask(((df_flights['AircraftType']=='789')|
                                                                                  (df_flights['AircraftType']=='781')),
                                                                                 '78')
    df_flights['AircraftType_general'] = df_flights['AircraftType_general'].mask(((df_flights['AircraftType']=='333')|
                                                                                  (df_flights['AircraftType']=='332')),
                                                                                 '33')
    ac_types = list(set(df_flights['AircraftType_general']))
    ac_type_ref = ac_types[0]
    ac_types_other = ac_types[1:]

    KS_results = {}
    for TAT_type in TAT_types_names:
        df_flights_TAT_type = df_flights[df_flights['TATType']==TAT_type]
        TAT_ref = df_flights_TAT_type[df_flights_TAT_type['AircraftType_general']==ac_type_ref]['TAT'].to_list()
        for ac_type_compare in ac_types_other:
            TAT_compare = df_flights_TAT_type[df_flights_TAT_type['AircraftType_general']==ac_type_compare]['TAT'].to_list()
            test_id = TAT_type+'-'+ac_type_ref+'-'+ac_type_compare
            KS = stats.kstest(TAT_ref,TAT_compare)
            KS_results[test_id] = KS

    # Generate list of dictionaries for each TAT type
    TAT = []
    for tt in TAT_types_names:
        TAT_historical = df_flights[df_flights['TATType']==tt]['TAT'].to_list()
        # Find min TAT in historical data
        TAT_min = min(TAT_historical)
        # Airports in this TAT type
        airports = sorted(list(set(df_flights[df_flights['TATType']==tt]['DepartureAirport'].to_list())))

        type_dic = {
            'type': tt,
            'TAT_historical': TAT_historical,
            'TAT_fitted_dist': None,
            'airports':airports,
            'TAT_min': TAT_min
                    }
        TAT.append(type_dic)

    ####### FIT EMPIRICAL DATA DATA #######
    for tt in TAT:
        # NOTE: These distributions are the ones selected when testing the full set of distributions allowed by
        #  distfit. However, pickling the results when fitting the full set of distributions throws exeptions if
        #  scipy is updated to its latest version. If scipy updated, only fit these distributions to allow
        #  for pickling.
        distributions_to_test = ['fisk', 'vonmises_line', 'mielke']

        # Initialize distfit
        dist = distfit(todf=True, distr='full')#distributions_to_test)#'full')
        TAT_historical = tt['TAT_historical']
        # Find bins to group empirically observed delays
        min_TAT = min(TAT_historical)
        max_TAT = max(TAT_historical)
        n_bins = round((max_TAT - min_TAT + 1)/P.TAT_DISTRIBUTIONS_BIN_SIZE)
        dist.bins = n_bins

        # Determine best-fitting probability distribution for data
        dist.fit_transform(np.array(TAT_historical))
        # Save results in disruption levels dictionary
        tt['TAT_fitted_dist'] = dist

        # Print summary and plot obtained fitting
        TAT_type = tt['type']
        print('\n\n################# TURN AROUND TYPE: ', TAT_type, ' #################')
        print(dist.summary)
        dist.plot()
        dist.plot_summary()

    ####### SAVE FITTED DISTRIBUTIONS #######
    write_pickle(TAT, 'TAT')
    breakpoint()
    ####### PRINT HISTOGRAMS #######
    fig, axs = plt.subplots(len(TAT), sharex='col')
    plt.suptitle('Turn Around Time at different airport types')

    for i in range(len(TAT)):
        tt = TAT[i]
        # Fitted distribution
        tat_fitted = tt['TAT_fitted_dist']

        tat_historical = tt['TAT_historical']
        tat_type = tt['type']
        tat_fitted_dist_name = tat_fitted.model['name']

        axs[i].set_title(tat_type+' - '+tat_fitted_dist_name)
        axs[i].hist(tat_historical, bins=range(floor(min(tat_historical)), ceil(max(tat_historical))),
                       density=True, label='Historical')
        x = np.array(range(round(min(tat_historical)), round(max(tat_historical))))
        y = tat_fitted.model['distr'].pdf(x, *tat_fitted.model['arg'],
                                          loc=tat_fitted.model['loc'],
                                          scale=tat_fitted.model['scale'])

        axs[i].plot(x, y, label='Fitted')
        axs[i].grid()

    axs[len(TAT)-1].set(xlabel='[min]')
    axs[0].legend()
    # plt.show()

    print('Turn Around Time fitted')

def find_flights_duration(df_flights_ICA):
    ''' This function finds the average block time duration for a certain (origin,destination),
    where origin and destinations are reported in alphabetical order. Saves to pickle a dataframe containing
    (origin,destination) and the corresponding block time in minutes. Not that block time duration is defined as the
    time between the off block time and the arrival time (on block time) of a flight. '''

    ###### Make a copy of the original df ######
    df_flights = df_flights_ICA.copy(deep=True)
    # Only keep flights for which the block time is known
    df_flights = df_flights[pd.isnull(df_flights['ActualBlockTimeDuration'])==0]

    # Remove cancelled flights and flights with same arrival and departure airport in data
    df_flights = df_flights[df_flights['FlightCancelled'] == 0]
    df_flights = df_flights[df_flights['DepartureAirport']!=df_flights['ArrivalAirport']]

    ###### Generate dataframe with origin-destination tuples and their flight duration ######
    # Generate origin-destination column and list
    df_flights['OrigDestAirports'] = list(zip(df_flights['DepartureAirport'],df_flights['ArrivalAirport']))
    df_flights['OrigDestAirports'] = [tuple(sorted(x)) for x in df_flights['OrigDestAirports']]
    flights_duration = df_flights.groupby('OrigDestAirports', as_index=False)['ActualBlockTimeDuration'].mean().round()
    flights_duration['ActualBlockTimeDuration'] = flights_duration['ActualBlockTimeDuration'].astype(int)

    write_pickle(flights_duration, 'flights_duration')
    print('Flights duration data generated correctly')


def find_hub_disruptions():
    '''
        Given historical data on disruption events, this function:
        - Categorizes disruption events based on their severity level
        - Finds empirical distributions for the duration of each disruption level event
        - Finds analytical distributions for the delays happening in each disruption level
        - Finds the transition matrix between disruption events

        Running this function in RUN_CONFIG mode:
        10: will take raw data as input and compute empirical distributions for disruption events duration,
            delays durations for different levels of disruption severity, and transition matrix between disruption events.
            The results are saved as a pickle file.
            The function will then compute analytical distributions for delay durations
            This mode takes about 1 hour to run, when three years of input is used.
        11: Will take the pickle file generated during mode 10 as input, and only compute fit analytical distributions
        '''
    # If config requires extraction of empirical distributions, import data on flights delays from AMS and extract
    # empirical distribuitons and disruption events transition matrix.
    if RUN_CONFIG.MODE == 10:
        find_AMS_disruption_events_and_delays_duration_and_transition_matrix()

    # Find analytical distributions of delays duration per disruption event severity level
    find_AMS_delay_analytical_distributions()


def find_AMS_disruption_events_and_delays_duration_and_transition_matrix():
    disruption_levels = P.DISRUPTION_LEVELS
    departure_time_ref = P.DEPARTURE_TIME_REF
    separator_level_id = 10000  # Disruption LevelId to separate days between each other
    separator_bracket_id = 10000  # Separator sequenceBracketId

    # Start time count
    start_time = time.time()
    print('Start time script')

    ##### Import flights data #####
    df_flights = load_csv('FlightLegsWithPrideLegs_reduced_allFlights', localize_time=False)
    # Filter out flights with no relevant time information
    df_flights = df_flights[~pd.isnull(df_flights[departure_time_ref])]
    # Remove flights with delay code 41-52 (technical) or 93 (reactionary aircraft rotation)
    condition_1 = pd.isnull(df_flights['DelayCode1']) | (((df_flights['DelayCode1']<41) | (df_flights['DelayCode1']>52)) & (df_flights['DelayCode1']!=93))
    condition_2 = pd.isnull(df_flights['DelayCode2']) | (((df_flights['DelayCode2']<41) | (df_flights['DelayCode2']>52)) & (df_flights['DelayCode2']!=93))
    df_flights = df_flights[condition_1 & condition_2]

    # df_flights = df_flights[0:10000] #TODO can be used for fast computing during development

    ##### Generate brackets dataframe #####
    brackets_n = P.BRACKETS_N
    brackets_duration = P.BRACKETS_DURATION
    brackets_time_start = P.BRACKETS_TIME_START
    list_brackets_start = []
    list_brackets_end = []
    for br in range (brackets_n):
        list_brackets_start.append((brackets_time_start + timedelta(minutes=br*brackets_duration)).time())
        list_brackets_end.append((brackets_time_start + timedelta(minutes=(br+1) * brackets_duration-1)).time())

    brackets_dict = {'BracketSeqNumber': range(brackets_n),
                     'TimeStart': list_brackets_start,
                     'TimeEnd': list_brackets_end}
    df_brackets = pd.DataFrame(data=brackets_dict)

    print((time.time() - start_time), 'Data was imported')

    ##### Generate dataframes #####
    date_start = dt.strptime(min(df_flights[departure_time_ref]), '%Y-%m-%d %H:%M:%S').date()
    date_end = dt.strptime(max(df_flights[departure_time_ref]), '%Y-%m-%d %H:%M:%S').date()
    date_range = pd.date_range(date_start, date_end, freq='d')
    df_dates = pd.DataFrame(data={'Date':date_range})
    df_dates['Date'] = df_dates['Date'].dt.date
    # Generate temporary columns for merging df
    df_dates['tmp'] = 1
    df_brackets['tmp'] = 1
    # Merge date and bracket dataframes
    df_dates = pd.merge(df_dates, df_brackets, on=['tmp'])
    df_dates = df_dates.drop('tmp', axis=1)
    df_dates = df_dates.merge(df_brackets['BracketSeqNumber'], how='outer')
    # Use index as id of each bracket
    df_dates = df_dates.sort_values(['Date', 'BracketSeqNumber'])
    df_dates.reset_index(inplace=True, drop=True)
    df_dates['BracketId'] = df_dates.index

    print((time.time() - start_time), 'Dates dataframe was generated')

    ##### Find departure date and time for each flight #####
    df_flights[departure_time_ref] = pd.to_datetime(df_flights[departure_time_ref])
    df_flights['BlockDepLocal_date'] = df_flights[departure_time_ref].dt.date
    df_flights['BlockDepLocal_time'] = df_flights[departure_time_ref].dt.time
    # Make all nan values of delay duration into zeros
    df_flights['DelayCode1Duration'] = df_flights['DelayCode1Duration'].fillna(0)
    df_flights['DelayCode2Duration'] = df_flights['DelayCode2Duration'].fillna(0)
    df_flights['DepartureDelayDuration'] = df_flights['DepartureDelayDuration'].fillna(0)
    # Exclude flights that leave with an earlier anticipation that a specified value
    df_flights = df_flights[df_flights['DepartureDelayDuration'] >= P.MAX_ANTICIPATION_ALLOWED]

    # Assign a bracket to each flight, and find average delay of bracket
    df_flights['Bracket'] = ''
    df_dates['DelayAvg'] = ''

    # Bar progress
    progress = tqdm(df_dates.iterrows(), total=df_dates.shape[0], file=sys.stdout)
    progress.set_description('Computing average delay per bracket')

    for index, br in progress:
        # Assign bracket to flights
        df_flights.loc[(df_flights['BlockDepLocal_date'] == br['Date']) & \
                       (df_flights['BlockDepLocal_time'] >= br['TimeStart']) & \
                       (df_flights['BlockDepLocal_time'] <= br['TimeEnd']) \
            , 'Bracket'] = br['BracketId']

        delay_avg = df_flights.loc[df_flights['Bracket']==br['BracketId']]['DepartureDelayDuration'].mean()
        # Round delay to the minute
        if pd.isnull(delay_avg)==0:
            delay_avg = round(delay_avg)
        df_dates.loc[index, 'DelayAvg'] = delay_avg

    print((time.time() - start_time), 'Flights were assigned to brackets, bracket average delay found')

    # Assign a disruption level to each bracket
    df_dates['DisrLevel'] = None
    df_dates['DisrLevelId'] = None
    for disr in disruption_levels:
        level = disr['level']
        level_id = disr['levelId']
        delay_min = disr['min']
        delay_max = disr['max']
        df_dates.loc[(df_dates['DelayAvg']>=delay_min) & (df_dates['DelayAvg'] <= delay_max), 'DisrLevel'] = level
        df_dates.loc[(df_dates['DelayAvg'] >= delay_min) & (df_dates['DelayAvg'] <= delay_max), 'DisrLevelId'] = level_id
    # Brackets for which data is not available are considered as disruption delay 0
    df_dates['DisrLevel'] = df_dates['DisrLevel'].fillna('norm')
    df_dates['DisrLevelId'] = df_dates['DisrLevelId'].fillna(0)

    print((time.time() - start_time), 'brackets were assigned a disruption level')


    ##### Find sequence of disruptions events #####
    # Add row to separate days between each other
    df_separators = df_dates.iloc[:0, :].copy()
    df_separators['Date'] = date_range
    df_separators['Date'] = df_separators['Date'].dt.date
    df_separators['BracketSeqNumber'] = separator_bracket_id   # Large number greater than bracket seq number
    df_separators['DisrLevelId'] = separator_level_id
    df_dates = pd.concat([df_dates, df_separators])
    df_dates = df_dates.sort_values(['Date', 'BracketSeqNumber'])

    # Find sequences of disruption events
    disruptions = df_dates['DisrLevelId'].to_list()
    disruptions_events = []
    for key, group in groupby(disruptions):
        disruptions_events.append([key, len(list(group))])

    print((time.time() - start_time), 'Disruptions events were found')

    ##### Find transition probability matrix #####
    # Generate probability matrix
    transition_probability_matrix = np.zeros([len(disruption_levels), len(disruption_levels)])
    # Fill matrix with all occurances of a transition
    for transition_from, transition_to in product( range(len(disruption_levels)), range(len(disruption_levels))):
        transition_occurrances = [disr for ind, disr in enumerate(disruptions_events[:-1])
                                  if disruptions_events[ind][0] == transition_from
                                  and disruptions_events[ind + 1][0] == transition_to]
        transition_count = len(transition_occurrances)
        transition_probability_matrix[transition_from,transition_to] =  transition_probability_matrix[transition_from,transition_to] + transition_count
    # Divide each cell by the row sum to obtain a transition probability
    for i in range(len(disruption_levels)):
        transition_probability_matrix[i,:] = transition_probability_matrix[i,:]/sum(transition_probability_matrix[i,:])

    print((time.time() - start_time), 'Transition probability matrix was generated')

    ##### Find delays duration disribution for each state #####
    for disr_level in disruption_levels:
        disr_occurrances = [disr for disr in disruptions_events if disr[0] == disr_level['levelId']]
        disr_level['events_duration'] = [disr[1] for disr in disr_occurrances]


    ##### Find list of delays for each disruption category #####
    for disr_level in disruption_levels:
        level_id = disr_level['levelId']
        # Find brackets with certain disruption level
        brackets_with_disruption_level = df_dates[df_dates['DisrLevelId']==level_id]['BracketId'].to_list()
        # Find corresponding flights, but only from ICA fleet
        flights_with_disruption_level = df_flights[(df_flights['Bracket'].isin(brackets_with_disruption_level))
                                                    & (df_flights['FlightGroup']=='ICA')
                                                    & (df_flights['AircraftType']!='73J')  ]
        disr_level['delays_duration'] = flights_with_disruption_level['DepartureDelayDuration'].to_list()

    print((time.time() - start_time), 'Lists of disruption events durations and delays durations were found')

    ##### Save obtained results to pickle #####
    distributions_disruptions = DistributionsDisruptions(disruption_levels, transition_probability_matrix)
    write_pickle(distributions_disruptions, 'distributions_disruptions')

    print((time.time() - start_time), 'Results saved to file')

def find_AMS_delay_analytical_distributions():
    '''
    Given pre-computed historical data on disruption events durations and transitions, as well as aircraft delays,
    this function fits analytical distributions to be used for sampling.
    returns a DistributionsDisruptions object contiaining:
    - Transition probability matrix, for which each element (i,j) states the probability of transitioning from
        disruption state i to state j.
    - Data regarding each disruption state level:
        -- disruption level event duration: historical data and fitted distribution
        -- aircraft delays: historical data, probability of having a delay <=0 min, and fitted distribution
    '''
    # Read pickle containing empirical data
    distributions_disruptions = read_pickle('distributions_disruptions')

    ################## FLIGHT DELAYS DURATIONS ######################
    # It is assumed that aircraft cannot have a negative delay (anticipation).
    # Compute the probability that a flight has a delay <= 0 from historical observations
    for disr_level in distributions_disruptions.disruption_levels:
        delays_duration = disr_level['delays_duration']
        # List of observed delays less or equal to zero
        delays_duration_less_equal_zero = [dl for dl in delays_duration if dl<=0]
        # Probability to have a delay <=0
        p_delay_less_equal_zero = len(delays_duration_less_equal_zero)/len(delays_duration)
        # save in dictionary
        disr_level['probability_no_delay'] = p_delay_less_equal_zero

    # Find distributions for the delays duration per disruption event severity level
    for disr_level in distributions_disruptions.disruption_levels:
        # Initialize distfit
        dist = distfit(todf=True)

        # Find bins to group empirically observed delays
        min_delay = min(disr_level['delays_duration'])
        max_delay = max(disr_level['delays_duration'])
        n_bins = round((max_delay - min_delay + 1) / P.DELAYS_DISTRIBUTIONS_BIN_SIZE)
        dist.bins = n_bins

        # Empirical data to be fitted
        delays_durations = disr_level['delays_duration']
        # Determine best-fitting probability distribution for data
        dist.fit_transform(np.array(delays_durations))
        # Save results in disruption levels dictionary
        disr_level['delays_fitted_dist'] = dist

        # Print summary and plot obtained fitting
        disr_name = disr_level['level']
        print('\n\n################# DISRUPTION LEVEL: ', disr_name, ' #################')
        print(dist.summary)
        dist.plot()
        dist.plot_summary()



    ################## DISRUPTION EVENTS DURATIONS ###################### 
    # Find distributions for the disruption events duration per disruption event severity level
    for disr_level in distributions_disruptions.disruption_levels:
        # Initialize distfit
        dist = distfit(todf=True)

        # Since minimum disruption event duration is 1 and not 0, reduce by one unit the observed data
        # to find a fitting distribution
        events_durations_to_fit = [ed-1 for ed in disr_level['events_duration']]
        # Find bins to group empirically observed delays
        min_delay = min(events_durations_to_fit)
        max_delay = max(events_durations_to_fit)
        n_bins = round((max_delay - min_delay + 1))
        dist.bins = n_bins

        # Determine best-fitting probability distribution for data
        dist.fit_transform(np.array(events_durations_to_fit))
        # Save results in disruption levels dictionary
        disr_level['events_fitted_dist'] = dist

        # Print summary and plot obtained fitting
        disr_name = disr_level['level']
        print('\n\n################# DISRUPTION LEVEL: ', disr_name, ' #################')
        print(dist.summary)
        dist.plot()
        dist.plot_summary()


    ################## SAVE FITTED DISTRIBUTIONS ######################
    # Save updated disruptions distributions dictionary to a pickle
    write_pickle(distributions_disruptions, 'distributions_disruptions_fitted')


    ##### Print histograms of delays #####
    fig, axs = plt.subplots(len(distributions_disruptions.disruption_levels),2, sharex='col')
    plt.suptitle('Aircraft delays distributions and disruption events duration distributions for different disruption severity level')

    for i in range(len(distributions_disruptions.disruption_levels)):
        disr = distributions_disruptions.disruption_levels[i]

        max_shown_delays = 400
        max_shown_events = 30

        # Fitted distribution
        delays_fitted_dist = disr['delays_fitted_dist']

        # Delays durations
        delays_durations = disr['delays_duration']
        axs[i, 0].set_title(disr['level'] + ' - ' + delays_fitted_dist.model['name'])
        axs[i, 0].hist(delays_durations, bins=range(floor(min(delays_durations)), max_shown_delays),
                       density=True, label='Historical')
        x = np.array(range(round(min(delays_durations)), max_shown_delays))
        y = delays_fitted_dist.model['distr'].pdf(x, *delays_fitted_dist.model['arg'], loc=delays_fitted_dist.model['loc'],
                                           scale=delays_fitted_dist.model['scale'])

        axs[i, 0].plot(x, y,label='Fitted')
        axs[i, 0].grid()


        # Disruption events durations NOTE: duration reduced by one unit
        events_durations = [ed-1 for ed in disr['events_duration']]
        axs[i, 1].set_title(disr['level'])
        axs[i, 1].hist(events_durations, bins=range(floor(min(events_durations)), max_shown_events),
                       density=True, label='Historical')
        events_fitted_dist = disr['events_fitted_dist']
        x = np.linspace(0, max_shown_events ,1000)
        y = events_fitted_dist.model['distr'].pdf(x, *events_fitted_dist.model['arg'], loc=events_fitted_dist.model['loc'],
                                           scale=events_fitted_dist.model['scale'])

        axs[i, 1].plot(x, y,label='Fitted')
        axs[i, 1].grid()

    axs[len(distributions_disruptions.disruption_levels) - 1, 0].set(xlabel='[min]')
    axs[len(distributions_disruptions.disruption_levels) - 1, 1].set(xlabel='[20 min units]')
    axs[0,1].legend()


    print('Disruptions distributions found')




