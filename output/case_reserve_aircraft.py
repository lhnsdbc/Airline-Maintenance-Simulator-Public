import os.path

from data_import.input_output import read_pickle, write_pickle, load_csv, load_excel
from validation.validation_network import print_hist
from matplotlib import pyplot as plt
from scipy.stats import lognorm
from config import directories, RESULTS, G
import copy
import numpy as np
import pandas as pd
from operator import add
from output.results_manage_files import compute_flights_cost
from scipy import stats
import timeit
import math
import glob

scale_weight_name = 'AOG duration factor'

SCALE_WEIGHTS = [1, 1.2, 1.4, 1.6, 1.8, 2]

def case_reserve_aircraft():
    reserve_aircraft_cost_new()
    breakpoint()
    reserve_zero_one_two()
    breakpoint()
    validation_costs_model()
    breakpoint()
    plot_AOG_scenarios()
    breakpoint()
    generate_AOG_scenarios()

def validation_costs_model():
    directory_runway = os.path.join(directories.input, 'runway')
    directories_files = glob.glob(os.path.join(directory_runway, '*'))
    dfs_concat = []
    for directory_file in directories_files:
        file_runway = pd.read_csv(directory_file, delimiter=';', decimal=',')
        dfs_concat.append(file_runway)
    df_runway = pd.concat(dfs_concat)
    df_runway = df_runway[df_runway['KoF']=='ICA']
    df_runway['cost_total'] = df_runway['eu_claim'] + df_runway['future_value_loss'] + df_runway['refunds'] + df_runway['hotel_overnights']
    cost_claim_avg = sum(df_runway['eu_claim'])/len(df_runway)
    cost_future_value_avg = sum(df_runway['future_value_loss'])/len(df_runway)
    cost_refunds_avg = sum(df_runway['refunds'])/len(df_runway)
    cost_hotel_avg = sum(df_runway['hotel_overnights'])/len(df_runway)
    cost_total_avg = cost_claim_avg + cost_future_value_avg + cost_refunds_avg + cost_hotel_avg

    df_op_costs = pd.read_excel(os.path.join(directories.input, 'operational_costs.xlsx'))
    df_op_costs.columns = df_op_costs.columns.astype(str) #[str(col) for col in df_op_costs.columns]
    fleets = ['772', '77W', '789', '781']
    costs = []
    for fleet in fleets:
        rows = [c for c in df_op_costs[fleet].to_list() if c != 'N.A.']
        costs.extend(rows)
    costs_op_avg = np.mean(costs)


    df_runway = pd.merge(df_runway, )

    print('hello')



def add_ci_norm(df, column):
    column_ci_min = column + '_ci_min'
    column_ci_max = column + '_ci_max'
    df[column_ci_min] = -1
    df[column_ci_max] = -1
    sim_runs = list(set(df['simulation_run_id'].to_list()))
    print(column)
    for sim_run in sim_runs:
        data = df[df['simulation_run_id'] == sim_run][column].to_list()
        shapiro_res = stats.shapiro(data)
        print(shapiro_res.pvalue)
        ci = stats.norm.interval(alpha=0.95, loc=np.mean(data), scale=stats.sem(data))
        df[column_ci_min] = df[column_ci_min].mask(df['simulation_run_id'] == sim_run, ci[0])
        df[column_ci_max] = df[column_ci_max].mask(df['simulation_run_id'] == sim_run, ci[1])
    return df

def add_ci_bootstrap(df, column):
    column_ci_min = column + '_ci_min'
    column_ci_max = column + '_ci_max'
    df[column_ci_min] = -1
    df[column_ci_max] = -1
    sim_runs = list(set(df['simulation_run_id'].to_list()))
    for sim_run in sim_runs:
        data = df[df['simulation_run_id'] == sim_run][column].to_list()
        data = (data,)
        res = stats.bootstrap(data, np.mean, confidence_level=0.95)
        ci = res.confidence_interval
        df[column_ci_min] = df[column_ci_min].mask(df['simulation_run_id'] == sim_run, ci[0])
        df[column_ci_max] = df[column_ci_max].mask(df['simulation_run_id'] == sim_run, ci[1])
    return df

def differece_of_samples(reserve1, reserve2, axis):
    reserve2 = np.mean(reserve2, axis=-1)
    reserve1 = np.mean(reserve1, axis=-1)
    # print('reserve', len(reserve1))
    difference = reserve1-reserve2
    # print('difference len', len(difference))
    # difference = np.mean(difference)
    # print('difference', difference)
    return difference

def add_ci_bootstrap_diff(df, column):
    column_ci_min = column + '_diff_ci_min'
    column_ci_max = column + '_diff_ci_max'
    df[column_ci_min] = -1
    df[column_ci_max] = -1
    sws = list(set(df['AOG_scale_weight'].to_list()))
    for sw in sws:
        data_reserve1 = df[(df['AOG_scale_weight'] == sw)
                           & (df['reserve_scenario'] == False)][column].to_list()
        data_reserve2 = df[(df['AOG_scale_weight'] == sw)
                           & (df['reserve_scenario'] == True)][column].to_list()
        res = stats.bootstrap((data_reserve1, data_reserve2), differece_of_samples, confidence_level=0.95,
                              method='percentile')
        ci = res.confidence_interval
        df[column_ci_min] = df[column_ci_min].mask(df['AOG_scale_weight'] == sw, ci[0])
        df[column_ci_max] = df[column_ci_max].mask(df['AOG_scale_weight'] == sw, ci[1])
    return df


def reserve_aircraft_cost_new():
    # df_fv = load_excel('Future_value') # TODO uncomment to print future value plot
    # fig, ax = plt.subplots()
    # plt.suptitle('Future value cost for an aircraft with 350 pax')
    # ax.plot(df_fv['Delay'], df_fv['ICA_non_elite']*350, c='C6')
    # ax.set(ylabel='Future value per pax')
    # ax.set(xlabel='Delay [min]')
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0)
    # reduce_dashboard_results()
    # breakpoint()

    MONETARY_UNITS_ON = True
    MONETARY_UNIT_VALUE = 18

    filename = 'dashboard_' + RESULTS.FILE_NAMES['flights']

    # Comment and uncomment to choose if full data or test data should be used
    # directory_input = directories.dashboard
    directory_input = os.path.join(directories.dashboard, 'Case_reserve')
    # directory_input = os.path.join(directories.dashboard, 'Case_reserve_reduced')

    # Validation cf
    # directory_input = os.path.join(directories.output, RESULTS.VALIDATION_SIMULATIONS[0])
    # filename = 'complete_' + RESULTS.FILE_NAMES['flights'] + '_'+RESULTS.VALIDATION_SIMULATIONS[0]

    time_now = timeit.default_timer()
    df_flights = load_csv(filename=filename, directory_input=directory_input)#, nrows=2 * 10**6)
    print('Time import data: ', (timeit.default_timer() - time_now)/60)
    time_now = timeit.default_timer()
    # Remove not executed flights
    df_flights = df_flights[df_flights['execution_state'] != 'future_rotation']
    df_flights = compute_flights_cost(df_flights)
    # Only keep two legs per flight
    df_flights = df_flights[(df_flights['airport_dep'] == G.AIRPORT_BASE) | (df_flights['airport_arr'] == G.AIRPORT_BASE)]
    # Find if data from scenario including extra reserve
    df_flights['reserve_scenario'] = (df_flights['simulation_run_id'].str.contains('_rs')) \
                                     | (df_flights['simulation_run_id'] == '20230222_134230')
    df_flights['reserve_scenario_string'] = '1 reserve'
    df_flights['reserve_scenario_string'] = df_flights['reserve_scenario_string'].mask(df_flights['reserve_scenario'], '2 reserves')
    df_flights['reserve_scenario_string'] = df_flights['reserve_scenario_string'].mask(
        (df_flights['simulation_run_id'].str.contains('_rs0')), '0 reserves')

    # Find AOG scale weight from simulation name
    df_flights['AOG_scale_weight'] = 1
    df_flights['AOG_scale_weight'] = df_flights['AOG_scale_weight'].mask(df_flights['simulation_run_id'].str.contains('sw12'), 1.2)
    df_flights['AOG_scale_weight'] = df_flights['AOG_scale_weight'].mask(df_flights['simulation_run_id'].str.contains('sw14'), 1.4)
    df_flights['AOG_scale_weight'] = df_flights['AOG_scale_weight'].mask(df_flights['simulation_run_id'].str.contains('sw16'), 1.6)
    df_flights['AOG_scale_weight'] = df_flights['AOG_scale_weight'].mask(df_flights['simulation_run_id'].str.contains('sw18'), 1.8)
    df_flights['AOG_scale_weight'] = df_flights['AOG_scale_weight'].mask(df_flights['simulation_run_id'].str.contains('sw2'), 2)


    df_flights['cost_delay'] = df_flights['cost_future_value'] + df_flights['cost_compensation']
    # Find average cost per simulation iteration
    columns_costs = [ 'cost_total', 'cost_future_value', 'cost_cancellation', 'cost_compensation']#, 'cost_food']
    for column in columns_costs+['cost_delay']:
        # Change to monetary units if requested
        if MONETARY_UNITS_ON:
            df_flights[column] = df_flights[column]*MONETARY_UNIT_VALUE
        df_flights[column+'_iter'] = df_flights.groupby(df_flights['simulation_id'])[column].transform('sum')
        # In millions, for a year instead of six months
        df_flights[column+'_iter'] = df_flights[column+'_iter']*2/10**6
    # df_flights['cost_iter_total'] = df_flights.groupby(df_flights['simulation_id'])['cost_total'].transform('sum')

    df_costs = df_flights[['simulation_id', 'simulation_run_id', 'reserve_scenario', 'AOG_scale_weight',
                           'reserve_scenario_string']+[cl+'_iter' for cl in columns_costs+['cost_delay']]]
    df_costs = df_costs.drop_duplicates()
    suffix_scenario_avg = '_scenario_avg'
    for column in columns_costs+['cost_delay']:
        df_costs[column+suffix_scenario_avg] = df_costs[column+'_iter'].groupby(df_costs['simulation_run_id']).transform('mean')
        df_costs = add_ci_norm(df_costs, column+'_iter')
        df_costs = add_ci_bootstrap_diff(df_costs, column+'_iter')
    # df_costs['cost_scenario_avg'] = df_costs['cost_iter_total'].groupby(df_costs['simulation_run_id']).transform('mean')/10**6
    df_costs = df_costs.drop(columns=['simulation_id']+[cl+'_iter' for cl in columns_costs+['cost_delay']])
    df_costs = df_costs.drop_duplicates()

    # RESERVE VALUE
    suffix_reserve = '_rs'
    df_reserves = df_costs[df_costs['reserve_scenario']]
    df_reserves = df_reserves.drop(columns=['simulation_run_id', 'reserve_scenario', 'reserve_scenario_string'])
    df_value = pd.merge(df_costs[df_costs['reserve_scenario']==0], df_reserves,
                        how='left', on='AOG_scale_weight', suffixes=('', suffix_reserve))
    suffix_value = '_value'
    for column in columns_costs+['cost_delay']:
        column_name = column + suffix_scenario_avg
        column_rs_name = column_name + suffix_reserve
        column_value_name = column + suffix_value
        df_value[column_value_name] = df_value[column_name] - df_value[column_rs_name]

    # COMPLETION FACTOR
    # Remove future rotations
    df_cf = df_flights[df_flights['execution_state']!='future_rotation']
    df_cf['fl_executed'] = df_cf['execution_state'] == 'executed'
    df_cf['fl_cancelled'] = df_cf['execution_state'].str.contains('cancelled')
    df_cf['fl_executed_count'] = df_cf.groupby('simulation_id')['fl_executed'].transform('sum')
    df_cf['fl_cancelled_count'] = df_cf.groupby('simulation_id')['fl_cancelled'].transform('sum')
    df_cf['compf_iteration'] = df_cf['fl_executed_count']/(df_cf['fl_executed_count'] + df_cf['fl_cancelled_count'])*100
    df_cf['cancf_iteration'] = df_cf['fl_cancelled_count']/(df_cf['fl_executed_count'] + df_cf['fl_cancelled_count'])*100
    df_cf = df_cf.drop_duplicates('simulation_id')
    # df_cf_iters = df_cf.copy()
    df_cf['fl_executed_avg'] = df_cf.groupby('simulation_run_id')['fl_executed_count'].transform('mean')
    df_cf['fl_cancelled_avg'] = df_cf.groupby('simulation_run_id')['fl_cancelled_count'].transform('mean')
    df_cf['compf_avg'] = df_cf.groupby('simulation_run_id')['compf_iteration'].transform('mean')
    df_cf['cancf_avg'] = df_cf.groupby('simulation_run_id')['cancf_iteration'].transform('mean')
    # Confidence interval 95%
    df_cf = add_ci_bootstrap(df_cf, 'cancf_iteration')
    df_cf = df_cf.drop_duplicates('simulation_run_id')
    df_cf = df_cf[['simulation_run_id', 'reserve_scenario', 'AOG_scale_weight', 'reserve_scenario_string',
                   'fl_executed_avg', 'fl_cancelled_avg', 'compf_avg', 'cancf_avg', 'cancf_iteration_ci_min',
                   'cancf_iteration_ci_max']]

    # REDUCED CF
    df_reserves = df_cf[df_cf['reserve_scenario']]
    df_reserves = df_reserves.drop(columns=['simulation_run_id', 'reserve_scenario', 'reserve_scenario_string'])
    df_reduced_cf = pd.merge(df_cf[df_cf['reserve_scenario'] == 0], df_reserves,
                             how='left', on='AOG_scale_weight', suffixes=('', suffix_reserve))
    df_reduced_cf['cancf_reduced'] = df_reduced_cf['cancf_avg'] - df_reduced_cf['cancf_avg' +suffix_reserve]

    # DELAYS
    df_delays = df_flights[df_flights['execution_state']=='executed']
    columns_delays = ['delay_arrival', 'delay_departure']
    for column in columns_delays:
        df_delays[column + '_iter'] = df_delays.groupby(df_delays['simulation_id'])[column].transform('mean')

    # AVERAGE DELAY
    df_delays_run = df_delays.drop_duplicates('simulation_id')
    df_delays_run['delay_departure_avg'] = df_delays_run.groupby('simulation_run_id')['delay_departure_iter'].transform('mean')
    df_delays_run['delay_arrival_avg'] = df_delays_run.groupby('simulation_run_id')['delay_arrival_iter'].transform('mean')
    # Confidence interval 95%
    columns_ci = ['delay_departure_iter', 'delay_arrival_iter']
    for col_ci in columns_ci:
        df_delays_run = add_ci_norm(df_delays_run, col_ci)

    df_delays_run = df_delays_run[['simulation_run_id', 'reserve_scenario', 'AOG_scale_weight',
                                   'reserve_scenario_string', 'delay_departure_avg', 'delay_arrival_avg']
                                  +[cl+'_ci_min' for cl in columns_ci] + [cl+'_ci_max' for cl in columns_ci]]
    df_delays_run = df_delays_run.drop_duplicates('simulation_run_id')


    # LONG DELAY
    delay_durations = [60, 120, 180, 240, 1000]
    for i in range(len(delay_durations)-1):
        # for delay in delay_durations:
        delay = delay_durations[i]
        delay_max = delay_durations[i+1]
        df_delays['delay'+str(delay)] = (df_delays['delay_arrival'] >= delay) & (df_delays['delay_arrival'] < delay_max)
    delay_durations.remove(1000)
    # df_delays['delay3'] = df_delays['delay_arrival']>3*60
    # df_delays['delay4'] = df_delays['delay_arrival']>4*60
    columns_delays_long = ['delay'+str(dl) for dl in delay_durations]
    # columns_delays_long = ['delay3', 'delay4']
    df_delays['group_count'] = df_delays.groupby(df_delays['simulation_id'])['simulation_id'].transform(len)
    for column in columns_delays_long:
        df_delays[column + '_iter'] = df_delays.groupby(df_delays['simulation_id'])[column].transform('sum')
        df_delays[column + '_cf_iter'] = df_delays[column + '_iter']/df_delays['group_count']*100
    df_delays_long = df_delays.drop_duplicates('simulation_id')

    suffix_delay_avg = '_avg'
    for delay in delay_durations:
        df_delays_long['delay'+str(delay)+suffix_delay_avg] = df_delays_long.groupby('simulation_run_id') \
            ['delay'+str(delay)+'_iter'].transform('mean')
        df_delays_long['delay'+str(delay)+'_cf'+suffix_delay_avg] = df_delays_long.groupby('simulation_run_id') \
            ['delay'+str(delay)+'_cf_iter'].transform('mean')
    # df_delays_long['delay3_avg'] = df_delays_long.groupby('simulation_run_id')['delay3_iter'].transform('mean')
    # df_delays_long['delay4_avg'] = df_delays_long.groupby('simulation_run_id')['delay4_iter'].transform('mean')
    # df_delays_long['delay3_cf_avg'] = df_delays_long.groupby('simulation_run_id')['delay3_cf_iter'].transform('mean')
    # df_delays_long['delay4_cf_avg'] = df_delays_long.groupby('simulation_run_id')['delay4_cf_iter'].transform('mean')
    df_delays_long = df_delays_long[['simulation_run_id', 'reserve_scenario', 'AOG_scale_weight',
                                     'reserve_scenario_string']+['delay'+str(dl)+'_cf'+suffix_delay_avg for dl in delay_durations]]
    df_delays_long = df_delays_long.drop_duplicates('simulation_run_id')

    print('Time elaboration: ',(timeit.default_timer()-time_now)/60)
    # breakpoint()
    ############################### PLOTS COMPLETION FACTOR ###############################
    colors = {'1 reserve': 'C0', '2 reserves': 'C1'}
    colors_sw = {1: 'C0', 1.2: 'C1', 1.4: 'C2', 1.6: 'C3', 1.8: 'C4', 2: 'C5'}
    if MONETARY_UNITS_ON:
        costs_unit = '[MU]'
    else:
        costs_unit = '[Million €]'

    # # CANCELLATION FACTOR #TODO old scatter
    # fig, ax = plt.subplots()
    # plt.suptitle('Cancellation factor for different AOG duration scenarios')
    # for reserve in ['1 reserve', '2 reserves']:
    #     rows_rs = df_cf[df_cf['reserve_scenario_string'] == reserve]
    #     ax.scatter(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], c=colors[reserve], label=reserve)
    # ax.set(ylabel='Cancellation factor [%]')
    # ax.set(xlabel=scale_weight_name)
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0)
    # ax.legend()

    x_ticks_cf = np.arange(1,2.1,0.2)
    y_ticks_cf = np.arange(0,0.45,0.05)

    # CANCELLATION FACTOR WITH CI
    df_cf = df_cf.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Cancellation factor for different AOG duration scenarios')
    for reserve in ['1 reserve', '2 reserves']:
        rows_rs = df_cf[df_cf['reserve_scenario_string'] == reserve]
        error_bars = [(rows_rs['cancf_avg'] - rows_rs['cancf_iteration_ci_min'].to_list()),
                      (rows_rs['cancf_iteration_ci_max'] - rows_rs['cancf_avg']).to_list()]
        ax.errorbar(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], yerr=error_bars,
                    c=colors[reserve], fmt='-o', label=reserve)
        # ax.plot(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], c=colors[reserve], label=reserve)
        ax.fill_between(rows_rs['AOG_scale_weight'], rows_rs['cancf_iteration_ci_min'],
                        rows_rs['cancf_iteration_ci_max'],
                        color=colors[reserve], alpha=0.1)
    ax.set(ylabel='Cancellation factor [%]')
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    ax.legend(loc='upper left')
    ax.set_xticks(x_ticks_cf)
    ax.set_yticks(y_ticks_cf)

    # CANCELLATION FACTOR WITH CI ONLY BASELINE
    df_cf = df_cf.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Cancellation factor for different AOG duration scenarios')
    reserve = '1 reserve'
    rows_rs = df_cf[(df_cf['reserve_scenario_string'] == reserve) & (df_cf['AOG_scale_weight']==1)]
    error_bars = [(rows_rs['cancf_avg'] - rows_rs['cancf_iteration_ci_min'].to_list()),
                  (rows_rs['cancf_iteration_ci_max'] - rows_rs['cancf_avg']).to_list()]
    ax.errorbar(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], yerr=error_bars,
                c=colors[reserve], fmt='-o', label=reserve)
    ax.set(ylabel='Cancellation factor [%]')
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    ax.legend(loc='upper left')
    ax.set_xticks(x_ticks_cf)
    ax.set_yticks(y_ticks_cf)

    # CANCELLATION FACTOR WITH CI BASELINE + RESERVE
    df_cf = df_cf.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Cancellation factor for different AOG duration scenarios')
    for reserve in ['1 reserve', '2 reserves']:
        rows_rs = df_cf[(df_cf['reserve_scenario_string'] == reserve) & (df_cf['AOG_scale_weight']==1)]
        error_bars = [(rows_rs['cancf_avg'] - rows_rs['cancf_iteration_ci_min'].to_list()),
                      (rows_rs['cancf_iteration_ci_max'] - rows_rs['cancf_avg']).to_list()]
        ax.errorbar(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], yerr=error_bars,
                    c=colors[reserve], fmt='-o', label=reserve)
    ax.set(ylabel='Cancellation factor [%]')
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    ax.legend(loc='upper left')
    ax.set_xticks(x_ticks_cf)
    ax.set_yticks(y_ticks_cf)

    # CANCELLATION FACTOR WITH CI BASELINE + RESERVE + SCALE WEIGHTS
    df_cf = df_cf.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Cancellation factor for different AOG duration scenarios')
    for reserve in ['1 reserve']:
        rows_rs = df_cf[df_cf['reserve_scenario_string'] == reserve]
        error_bars = [(rows_rs['cancf_avg'] - rows_rs['cancf_iteration_ci_min'].to_list()),
                      (rows_rs['cancf_iteration_ci_max'] - rows_rs['cancf_avg']).to_list()]
        ax.errorbar(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], yerr=error_bars,
                    c=colors[reserve], fmt='-o', label=reserve)
        # ax.plot(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], c=colors[reserve], label=reserve)
        ax.fill_between(rows_rs['AOG_scale_weight'], rows_rs['cancf_iteration_ci_min'],
                        rows_rs['cancf_iteration_ci_max'],
                        color=colors[reserve], alpha=0.1)
    reserve = '2 reserves'
    rows_rs = df_cf[(df_cf['reserve_scenario_string'] == reserve) & (df_cf['AOG_scale_weight'] == 1)]
    error_bars = [(rows_rs['cancf_avg'] - rows_rs['cancf_iteration_ci_min'].to_list()),
                  (rows_rs['cancf_iteration_ci_max'] - rows_rs['cancf_avg']).to_list()]
    ax.errorbar(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], yerr=error_bars,
                c=colors[reserve], fmt='-o', label=reserve)
    ax.set(ylabel='Cancellation factor [%]')
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    ax.legend(loc='upper left')
    ax.set_xticks(x_ticks_cf)
    ax.set_yticks(y_ticks_cf)

    # REDUCED CANCELLATION FACTOR
    # fig, ax = plt.subplots()
    # plt.suptitle('Cancellation factor reduction through the use of an extra reserve')
    # ax.scatter(df_reduced_cf['AOG_scale_weight'], df_reduced_cf['cancf_reduced'], c='C6')
    # ax.set(ylabel='Cancellation factor reduction [%]')
    # ax.set(xlabel=scale_weight_name)
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0)

    # # AVERAGE DELAY # TODO old scatter
    # fig, ax = plt.subplots()
    # plt.suptitle('Flights arrival delay for different AOG duration scenarios')
    # for reserve in ['1 reserve', '2 reserves']:
    #     rows_rs = df_delays_run[df_delays_run['reserve_scenario_string'] == reserve]
    #     ax.scatter(rows_rs['AOG_scale_weight'], rows_rs['delay_arrival_avg'], c=colors[reserve], label=reserve)
    # ax.set(ylabel='Average arrival delay')
    # ax.set(xlabel=scale_weight_name)
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0, ymax=14)
    # ax.legend()

    # # AVERAGE DEPARTURE DELAY CI
    # df_delays_run = df_delays_run.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    # fig, ax = plt.subplots()
    # plt.suptitle('Average departure delay for different AOG duration scenarios')
    # for reserve in ['1 reserve', '2 reserves']:
    #     rows_rs = df_delays_run[df_delays_run['reserve_scenario_string'] == reserve]
    #     ax.plot(rows_rs['AOG_scale_weight'], rows_rs['delay_departure_avg'], c=colors[reserve], label=reserve)
    #     ax.fill_between(rows_rs['AOG_scale_weight'], rows_rs['delay_departure_iter_ci_min'],
    #                     rows_rs['delay_departure_iter_ci_max'],
    #                     color=colors[reserve], alpha=0.1)
    # ax.set(ylabel='Departure delay')
    # ax.set(xlabel=scale_weight_name)
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0)
    # ax.legend()
    #
    # # AVERAGE ARRIVAL DELAY CI
    # df_delays_run = df_delays_run.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    # fig, ax = plt.subplots()
    # plt.suptitle('Average arrival delay for different AOG duration scenarios')
    # for reserve in ['1 reserve', '2 reserves']:
    #     rows_rs = df_delays_run[df_delays_run['reserve_scenario_string'] == reserve]
    #     ax.plot(rows_rs['AOG_scale_weight'], rows_rs['delay_arrival_avg'], c=colors[reserve], label=reserve)
    #     ax.fill_between(rows_rs['AOG_scale_weight'], rows_rs['delay_arrival_iter_ci_min'],
    #                     rows_rs['delay_arrival_iter_ci_max'],
    #                     color=colors[reserve], alpha=0.1)
    # ax.set(ylabel='Arrival delay')
    # ax.set(xlabel=scale_weight_name)
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0)
    # ax.legend()

    # DELAYS EXCEEDANCE PROBABILITY FOR SCALE WEIGHT=1, DIFFERENT RESERVES
    df_delays = df_delays.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Exceedance probability of delays for an '+scale_weight_name+' of 1')
    for reserve in ['1 reserve', '2 reserves']:
        sw = 1
        rows_rs = df_delays[(df_delays['reserve_scenario_string'] == reserve)
                            & (df_delays['AOG_scale_weight'] == sw)]
        delays_to_plot = [dl for dl in rows_rs['delay_departure'].to_list() if dl<=60*8]
        bins = range(0,math.ceil(max(delays_to_plot)+10), 10)
        ax.hist(delays_to_plot, color=colors[reserve], label=reserve, #colors_sw[sw]
                histtype='step', bins=bins, linestyle='solid', density=True, cumulative=-1) #'step'
    # ax.set(ylabel='Long delays [%]')
    ax.set(xlabel='Departure delay [min]', ylabel='Exceedance Probability')
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)#, ymax=1)
    ax.legend()
    ax.set_xticks(range(0,math.ceil(max(delays_to_plot)+10), 20))

    # DELAYS EXCEEDANCE PROBABILITY FOR DIFFERENT SCALE WEIGHTS, RESERVE=1
    df_delays = df_delays.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('CDF of delays for 1 reserve')
    reserve = '1 reserve'
    for sw in SCALE_WEIGHTS:
        rows_rs = df_delays[(df_delays['reserve_scenario_string'] == reserve)
                            & (df_delays['AOG_scale_weight'] == sw)]
        delays_to_plot = [dl for dl in rows_rs['delay_departure'].to_list() if dl<=60*8]
        bins = range(0,math.ceil(max(delays_to_plot)+10), 10)
        ax.hist(delays_to_plot, label=reserve, #colors_sw[sw]
                histtype='step', bins=bins, linestyle='solid', density=True, cumulative=-1) #'step'
    # ax.set(ylabel='Long delays [%]')
    ax.set(xlabel='Departure delay [min]', ylabel='Exceedance Probability')
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)#, ymax=1)
    ax.legend()
    ax.set_xticks(range(0,math.ceil(max(delays_to_plot)+10), 20))

    # DELAYS CDF SCALE WEIGHT 1
    df_delays = df_delays.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('CDF of delays for an ' + scale_weight_name + ' of 1')
    for reserve in ['1 reserve', '2 reserves']:
        sw = 1
        rows_rs = df_delays[(df_delays['reserve_scenario_string'] == reserve)
                            & (df_delays['AOG_scale_weight'] == sw)]
        delays_to_plot = [dl for dl in rows_rs['delay_departure'].to_list() if dl <= 60 * 8]
        bins = range(0, math.ceil(max(delays_to_plot) + 10), 10)
        ax.hist(delays_to_plot, color=colors[reserve], label=reserve,  # colors_sw[sw]
                histtype='step', bins=bins, linestyle='solid', density=True, cumulative=1)  # 'step'
    # ax.set(ylabel='Long delays [%]')
    ax.set(xlabel='Departure delay [min]', ylabel='Cumulative distribution')
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)  # , ymax=1)
    ax.legend()
    ax.set_xticks(range(0, math.ceil(max(delays_to_plot) + 10), 20))

    #
    # # DELAYS CDF NEW
    # df_delays = df_delays.sort_values(['AOG_scale_weight', 'reserve_scenario_string'])
    # fig, ax = plt.subplots()
    # plt.suptitle('CDF of delays per scenario')
    # data_delay = []
    # labels_reserve = []
    # colors_delay = []
    # for sw in list(set(df_delays['AOG_scale_weight'])):
    #     for reserve in ['1 reserve', '2 reserves']:
    #         if reserve == '1 reserve':
    #             line_style = 'solid'
    #         else:
    #             line_style = '--'
    #         rows_rs = df_delays[(df_delays['reserve_scenario_string'] == reserve)
    #                             & (df_delays['AOG_scale_weight'] == sw)]
    #         delays_to_plot = [dl / 60 for dl in rows_rs['delay_departure'].to_list() if dl > 0 and dl <= 60 * 8]
    #         data_delay.append(delays_to_plot)
    #         labels_reserve.append(reserve)
    #         colors_delay.append(colors[reserve])
    # bins = range(0, math.ceil(max([el for ls in data_delay for el in ls]) + 1))
    # ax.hist(data_delay, bins=bins, label=labels_reserve, color=colors_delay,# label=reserve + ' sw ' + str(sw), # color=colors[# reserve], # colors_sw[sw]
    #         histtype='bar', linestyle=line_style, density=True, cumulative=-1)  # 'step'
    # # ax.set(ylabel='Long delays [%]')
    # ax.set(xlabel='Departure delay [hours]')
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0)  # , ymax=1)
    # ax.legend()

    # DELAYS CDF NEW 2
    df_delays = df_delays.sort_values(['AOG_scale_weight', 'reserve_scenario_string'])
    fig, ax = plt.subplots()
    plt.suptitle('CDF of delays per scenario')

    for reserve in ['1 reserve', '2 reserves']:
        data_delay = []
        labels_reserve = []
        labels_sw = []
        colors_delay = []
        if reserve == '1 reserve':
            line_style = 'solid'
            fill = True
            hatch = None #str('\\\\')
            edgecolor = colors[reserve]
            linewidth = 1
            alpha = 1
        else:
            line_style = 'solid'
            fill = False
            hatch = None #'//'
            edgecolor = colors[reserve]
            linewidth = 1
            alpha = .5
        sw_values = list(set(df_delays['AOG_scale_weight']))
        sw_values = sorted(sw_values)
        for sw in sw_values:

            rows_rs = df_delays[(df_delays['reserve_scenario_string'] == reserve)
                                & (df_delays['AOG_scale_weight'] == sw)]
            delays_to_plot = [dl / 60 for dl in rows_rs['delay_departure'].to_list() if dl <= 60 * 8]# dl > 0 and ]
            data_delay.append(delays_to_plot)
            if len(labels_reserve) < 1:
                labels_reserve.append(reserve)
            colors_delay.append(colors[reserve])
        # bins = range(0, math.ceil(max([el for ls in data_delay for el in ls]) + 1))
        bins = np.arange(0, math.ceil(max([el for ls in data_delay for el in ls]) + 0.5), 0.5)
        _,b,_ = ax.hist(data_delay, bins=bins, label=labels_reserve, color=colors_delay,
                        # label=reserve + ' sw ' + str(sw), # color=colors[# reserve], # colors_sw[sw]
                        histtype='bar', linestyle=line_style, density=True, cumulative=-1, fill=fill, hatch=hatch,
                        edgecolor=edgecolor, linewidth=linewidth)  # 'step'
        rects = ax.patches
        sw_labels = [sw for sw in sw_values for i in range(len(b)-1)]
        for rect, label in zip(rects, sw_labels):
            # height = rect.get_height()
            # ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label, ha='center', va='bottom')
            ax.text(rect.get_x() + rect.get_width() / 2, - 0.008, label, ha='center', va='bottom', size=8,
                    rotation='vertical')
        # ax.bar_label(rects, labels=labels_sw)
        if reserve == '2 reserves':
            ax.hist(data_delay, bins=bins, color=colors_delay,
                    # label=reserve + ' sw ' + str(sw), # color=colors[# reserve], # colors_sw[sw]
                    histtype='bar', linestyle=line_style, density=True, cumulative=-1, fill=True, hatch=None,
                    edgecolor=None, alpha=.4, linewidth=0)  # 'step'
    # ax.set(ylabel='Long delays [%]')
    ax.set(xlabel='Departure delay [hours]')
    ax.set_xticks(bins)
    ax.tick_params(axis='x', length=100, size=10)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)  # , ymax=1)
    ax.legend()

    # # LONG DELAY
    # fig, ax = plt.subplots()
    # plt.suptitle('Occurrences of flights with arrival delay over three hours')
    # for reserve in ['1 reserve', '2 reserves']:
    #     rows_rs = df_delays_long[df_delays_long['reserve_scenario_string'] == reserve]
    #     ax.scatter(rows_rs['AOG_scale_weight'], rows_rs['delay180_cf_avg'], c=colors[reserve], label=reserve)
    # ax.set(ylabel='Long delays [%]')
    # ax.set(xlabel=scale_weight_name)
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0, ymax=0.4)
    # ax.legend()


    # # DELAY BRACKETS
    # fig, axs = plt.subplots(ncols=len(delay_durations), sharex=True, sharey=True)
    # plt.suptitle('Percentage of flights within delay bracket')
    # for columns_count in range(len(delay_durations)):
    #     column = delay_durations[columns_count]
    #     column_name = 'delay' + str(column) + '_cf' + suffix_delay_avg
    #     for reserve in ['1 reserve', '2 reserves']:
    #         rows_rs = df_delays_long[df_delays_long['reserve_scenario_string'] == reserve]
    #         axs[columns_count].scatter(rows_rs['AOG_scale_weight'], rows_rs[column_name], c=colors[reserve], label = reserve)
    #     axs[columns_count].grid(axis='y')
    #     axs[columns_count].set_title(str(column) + ' minutes')
    #     axs[columns_count].set(xlabel=scale_weight_name)
    #
    # axs[0].set(ylabel='[%]')
    # axs[columns_count].set_ylim(ymin=0)
    # axs[columns_count].legend()
    # axs[columns_count].set_xticks(np.linspace(1,2,6))
    # axs[columns_count].set_yticks(np.linspace(0, 2.5, 11))


    ############################### PLOTS COSTS ###############################

    colors = {'1 reserve': 'C0', '2 reserves': 'C1'}

    # # ALL COSTS # TODO old scatter
    # fig, axs = plt.subplots(ncols=len(columns_costs), sharex=True, sharey=True)
    # plt.suptitle('Cost of disruptions for different AOG duration scenarios')
    # for columns_count in range(len(columns_costs)):
    #     column = columns_costs[columns_count]
    #     column_name = column + suffix_scenario_avg
    #     for reserve in ['1 reserve', '2 reserves']:
    #         rows_rs = df_costs[df_costs['reserve_scenario_string'] == reserve]
    #         axs[columns_count].scatter(rows_rs['AOG_scale_weight'], rows_rs[column_name], c=colors[reserve], label = reserve)
    #     axs[columns_count].grid(axis='y')
    #     axs[columns_count].set_title(column.replace('_', ' ').replace('cost', ''))
    #     axs[columns_count].set(xlabel=scale_weight_name)
    #
    # axs[0].set(ylabel='[Million €]')
    # axs[columns_count].set_ylim(ymin=0)
    # axs[columns_count].legend()
    # axs[columns_count].set_xticks(np.linspace(1,2,6))
    # axs[columns_count].set_yticks(np.linspace(0, 40, 21))

    # ALL COSTS WITH CI
    df_costs = df_costs.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, axs = plt.subplots(ncols=len(columns_costs), sharex=True, sharey=True)
    plt.suptitle('Cost of disruptions for different AOG duration scenarios')
    for columns_count in range(len(columns_costs)):
        column = columns_costs[columns_count]
        column_name = column + suffix_scenario_avg
        for reserve in ['1 reserve', '2 reserves']:
            rows_rs = df_costs[df_costs['reserve_scenario_string'] == reserve]
            error_bars = [(rows_rs[column_name] - rows_rs[column + '_iter_ci_min'].to_list()),
                          (rows_rs[column + '_iter_ci_max'] - rows_rs[column_name]).to_list()]
            axs[columns_count].errorbar(rows_rs['AOG_scale_weight'], rows_rs[column_name], yerr=error_bars,
                                        c=colors[reserve], fmt='-o', label=reserve, markersize=4)
            axs[columns_count].fill_between(rows_rs['AOG_scale_weight'], rows_rs[column + '_iter_ci_min'],
                                            rows_rs[column + '_iter_ci_max'], color=colors[reserve], alpha=0.1)
        axs[columns_count].grid(axis='y')
        axs[columns_count].set_title(column.replace('_', ' ').replace('cost', ''))
        axs[columns_count].set(xlabel=scale_weight_name)

    axs[0].set(ylabel=costs_unit)
    axs[columns_count].set_ylim(ymin=0)
    axs[columns_count].legend()
    axs[columns_count].set_xticks(np.linspace(1,2,6))
    if MONETARY_UNITS_ON:
        axs[columns_count].set_yticks(np.arange(0, 1000, 100))
    else:
        axs[columns_count].set_yticks(np.arange(0, 61, 5))

    # ALL COSTS WITH CI, SUMMED DELAY COSTS
    df_costs = df_costs.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    column_costs_aggreg_delay = [ 'cost_total', 'cost_delay', 'cost_cancellation']
    fig, axs = plt.subplots(ncols=len(column_costs_aggreg_delay), sharex=True, sharey=True)
    plt.suptitle('Cost of disruptions for different AOG duration scenarios')
    for columns_count in range(len(column_costs_aggreg_delay)):
        column = column_costs_aggreg_delay[columns_count]
        column_name = column + suffix_scenario_avg
        for reserve in ['1 reserve', '2 reserves']:
            rows_rs = df_costs[df_costs['reserve_scenario_string'] == reserve]
            error_bars = [(rows_rs[column_name] - rows_rs[column + '_iter_ci_min'].to_list()),
                          (rows_rs[column + '_iter_ci_max'] - rows_rs[column_name]).to_list()]
            axs[columns_count].errorbar(rows_rs['AOG_scale_weight'], rows_rs[column_name], yerr=error_bars,
                                        c=colors[reserve], fmt='-o', label=reserve, markersize=4)
            axs[columns_count].fill_between(rows_rs['AOG_scale_weight'], rows_rs[column + '_iter_ci_min'],
                                            rows_rs[column + '_iter_ci_max'], color=colors[reserve], alpha=0.1)
        axs[columns_count].grid(axis='y')
        axs[columns_count].set_title(column.replace('_', ' ').replace('cost', ''))
        axs[columns_count].set(xlabel=scale_weight_name)

    axs[0].set(ylabel=costs_unit)
    axs[columns_count].set_ylim(ymin=0)
    axs[columns_count].legend()
    axs[columns_count].set_xticks(np.linspace(1,2,6))
    if MONETARY_UNITS_ON:
        axs[columns_count].set_yticks(np.arange(0, 1000, 100))
    else:
        axs[columns_count].set_yticks(np.arange(0, 61, 5))

    # # TOTAL AND CANCELLATIONS
    # columns_to_plot = ['cost_total', 'cost_cancellation']
    # fig, axs = plt.subplots(nrows=len(columns_to_plot), sharex=True)
    # plt.suptitle('Cost of disruptions for different AOG duration scenarios')
    # for columns_count in range(len(columns_to_plot)):
    #     column = columns_to_plot[columns_count]
    #     column_name = column + suffix_scenario_avg
    #     for reserve in ['1 reserve', '2 reserves']:
    #         rows_rs = df_costs[df_costs['reserve_scenario_string'] == reserve]
    #         axs[columns_count].scatter(rows_rs['AOG_scale_weight'], rows_rs[column_name], c=colors[reserve], label = reserve)
    #     axs[columns_count].set(ylabel=column+' [Million €]')
    #     axs[columns_count].grid(axis='y')
    #     axs[columns_count].set_title(column.replace('_', ' ').replace('cost', ''))
    #     axs[columns_count].set_ylim(ymin=0)
    # axs[columns_count].set(xlabel=scale_weight_name)
    # axs[columns_count].set_xticks(np.linspace(1, 2, 6))
    # axs[0].legend()


    # # RESERVE VALUE SCATTER # TODO old scatter
    # fig, ax = plt.subplots()
    # plt.suptitle('Avoided costs of disruptions through an extra reserve')
    # ax.scatter(df_value['AOG_scale_weight'], df_value['cost_total'+suffix_value], c='C6')
    # ax.set(ylabel='[Million €]')
    # ax.set(xlabel=scale_weight_name)
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0)
    # ax.set_xticks(np.linspace(1, 2, 6))

    # RESERVE VALUE PLOT WITH CI
    df_value = df_value.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Avoided costs of disruptions through an extra reserve')
    error_bars = [(df_value['cost_total'+suffix_value] - df_value['cost_total_iter_diff_ci_min'].to_list()),
                  (df_value['cost_total_iter_diff_ci_max'] - df_value['cost_total'+suffix_value]).to_list()]
    ax.errorbar(df_value['AOG_scale_weight'], df_value['cost_total'+suffix_value], yerr=error_bars,
                c='C6', fmt='-o')
    # ax.plot(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], c=colors[reserve], label=reserve)
    ax.fill_between(df_value['AOG_scale_weight'], df_value['cost_total_iter_diff_ci_min'],
                    df_value['cost_total_iter_diff_ci_max'],
                    color='C6', alpha=0.1)
    ax.set(ylabel=costs_unit)
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    # ax.legend()

    # RESERVE VALUE COMPONENTS WITH CI
    df_value = df_value.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Components of avoided costs of disruptions through an extra reserve')
    columns_to_plot = [cl for cl in columns_costs if cl!='cost_total']
    for column in columns_to_plot:
        error_bars = [(df_value[column+suffix_value] - df_value[column + '_iter_diff_ci_min'].to_list()),
                      (df_value[column + '_iter_diff_ci_max'] - df_value[column+suffix_value]).to_list()]
        ax.errorbar(df_value['AOG_scale_weight'], df_value[column+suffix_value], yerr=error_bars, fmt='-o',
                    label=column)
        # ax.plot(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], c=colors[reserve], label=reserve)
        ax.fill_between(df_value['AOG_scale_weight'], df_value[column+'_iter_diff_ci_min'],
                        df_value[column+'_iter_diff_ci_max'], alpha=0.1)
    ax.set(ylabel=costs_unit)
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    ax.legend()

    # RESERVE VALUE BAR
    fig, ax = plt.subplots()
    plt.suptitle('Avoided costs of disruptions through an extra reserve (per component)')
    bottom = [0 for el in list(set(df_value['AOG_scale_weight'].to_list()))]
    for cost_col in [cl for cl in columns_costs if cl!='cost_total']:
        print(cost_col+suffix_value)
        ax.bar(df_value['AOG_scale_weight'], df_value[cost_col+suffix_value],
               label= 'avoided '+ cost_col.replace('_',' '), width=0.15, bottom=bottom)
        bottom = list(map(add, bottom, df_value[cost_col+suffix_value].to_list()))
    ax.set(ylabel=costs_unit)
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    ax.set_xticks(np.linspace(1, 2, 6))
    ax.legend()

    # RESERVE VALUE STACKPLOT
    df_value = df_value.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Avoided costs of disruptions through an extra reserve (per component)')
    values_all = []
    label_costs = []
    columns_to_plot = ['cost_compensation', 'cost_future_value', 'cost_cancellation']
    for cost_col in columns_to_plot: #[cl for cl in columns_costs if cl!='cost_total']:
        values = df_value[cost_col+suffix_value]
        values_all.append(values)
        label_costs.append(cost_col)
    values_plot = np.vstack(values_all)
    ax.stackplot(df_value['AOG_scale_weight'], values_plot, labels=label_costs)
    # colors=['C'+str(i+2) for i in range(len(label_costs))])
    ax.set(ylabel=costs_unit)
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    ax.set_xticks(np.linspace(1, 2, 6))
    ax.legend()

    # RESERVE VALUE STACKPLOT AGGREGATED DELAYS
    df_value = df_value.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Avoided costs of disruptions through an extra reserve (per component)')
    values_all = []
    label_costs = []
    columns_to_plot = ['cost_delay', 'cost_cancellation']
    print('cancellation costs [MU]')
    for cost_col in columns_to_plot: #[cl for cl in columns_costs if cl!='cost_total']:
        values = df_value[cost_col+suffix_value]
        values_all.append(values)
        label_costs.append(cost_col)
        print(cost_col, values)
    values_plot = np.vstack(values_all)
    ax.stackplot(df_value['AOG_scale_weight'], values_plot, labels=label_costs)
    # colors=['C'+str(i+2) for i in range(len(label_costs))])
    ax.set(ylabel=costs_unit)
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0, ymax=140)
    ax.set_xticks(np.linspace(1, 2, 6))
    ax.legend()


    # Box plot total costs
    # weights = list(set(df_costs['AOG_scale_weight'].to_list()))
    # fig, axs = plt.subplots(nrows=1, ncols=len(weights), sharey=True, sharex=True)
    # weights_count = 0
    # for weight in weights:
    #     rows_weight = df_costs[df_costs['AOG_scale_weight'] == weight]
    #     axs[weights_count].scatter(rows_weight['reserve_scenario_string'], rows_weight['cost_scenario_avg'],)
    #
    #     weights_count += 1

    plt.show()
    print('hello')


def add_ci_bootstrap_diff_reserve0(df, column):
    column_ci_min = column + '_diff_ci_min'
    column_ci_max = column + '_diff_ci_max'
    df[column_ci_min] = np.nan
    df[column_ci_max] = np.nan
    sws = list(set(df['AOG_scale_weight'].to_list()))
    for sw in sws:
        for rs in ['2 reserves', '0 reserves']:
            if rs == '2 reserves':
                data_reserve1 = df[(df['AOG_scale_weight'] == sw)
                                   & (df['reserve_scenario_string'] == rs)][column].to_list()
                data_reserve2 = df[(df['AOG_scale_weight'] == sw)
                                   & (df['reserve_scenario'] == True)][column].to_list()
                res = stats.bootstrap((data_reserve1, data_reserve2), differece_of_samples, confidence_level=0.95,
                                      method='percentile')
            elif rs == '0 reserves':
                data_reserve2 = df[(df['AOG_scale_weight'] == sw)
                                   & (df['reserve_scenario_string'] == rs)][column].to_list()
                data_reserve1 = df[(df['AOG_scale_weight'] == sw)
                                   & (df['reserve_scenario'] == True)][column].to_list()
                res = stats.bootstrap((data_reserve1, data_reserve2), differece_of_samples, confidence_level=0.95,
                                      method='percentile')
            else:
                raise Exception('Not supported')

            ci = res.confidence_interval
            df[column_ci_min] = df[column_ci_min].mask(
                (df['AOG_scale_weight'] == sw) & (df['reserve_scenario_string'] == rs), ci[0])
            df[column_ci_max] = df[column_ci_max].mask(
                (df['AOG_scale_weight'] == sw) & (df['reserve_scenario_string'] == rs), ci[1])
    return df


def reserve_zero_one_two():
    # df_fv = load_excel('Future_value') # TODO uncomment to print future value plot
    # fig, ax = plt.subplots()
    # plt.suptitle('Cancellation factor reduction through the use of an extra reserve')
    # ax.scatter(df_fv['Delay'], df_fv['ICA_non_elite'], c='C6')
    # ax.set(ylabel='Future value per pax')
    # ax.set(xlabel='Delay [min]')
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0)
    # reduce_dashboard_results()
    # breakpoint()
    filename = 'dashboard_' + RESULTS.FILE_NAMES['flights']

    # Comment and uncomment to choose if full data or test data should be used
    directory_input = directories.dashboard
    # directory_input = os.path.join(directories.dashboard, 'Case_reserve')
    # directory_input = os.path.join(directories.dashboard, 'Case_reserve_reduced')

    time_now = timeit.default_timer()
    df_flights = load_csv(filename=filename, directory_input=directory_input)  # , nrows=2 * 10**6)
    print('Time import data: ', (timeit.default_timer() - time_now) / 60)
    time_now = timeit.default_timer()
    df_flights = compute_flights_cost(df_flights)
    # Only keep two legs per flight
    df_flights = df_flights[
        (df_flights['airport_dep'] == G.AIRPORT_BASE) | (df_flights['airport_arr'] == G.AIRPORT_BASE)]
    # Find if data from scenario including extra reserve
    df_flights['reserve_scenario'] = (df_flights['simulation_run_id'].str.contains('_rs')) \
                                     | (df_flights['simulation_run_id'] == '20230222_134230')
    df_flights['reserve_scenario_string'] = '1 reserve'
    df_flights['reserve_scenario_string'] = df_flights['reserve_scenario_string'].mask(df_flights['reserve_scenario'],
                                                                                       '2 reserves')
    df_flights['reserve_scenario_string'] = df_flights['reserve_scenario_string'].mask(
        (df_flights['simulation_run_id'].str.contains('_rs0')), '0 reserves')

    # Find AOG scale weight from simulation name
    df_flights['AOG_scale_weight'] = 1
    df_flights['AOG_scale_weight'] = df_flights['AOG_scale_weight'].mask(
        df_flights['simulation_run_id'].str.contains('sw12'), 1.2)
    df_flights['AOG_scale_weight'] = df_flights['AOG_scale_weight'].mask(
        df_flights['simulation_run_id'].str.contains('sw14'), 1.4)
    df_flights['AOG_scale_weight'] = df_flights['AOG_scale_weight'].mask(
        df_flights['simulation_run_id'].str.contains('sw16'), 1.6)
    df_flights['AOG_scale_weight'] = df_flights['AOG_scale_weight'].mask(
        df_flights['simulation_run_id'].str.contains('sw18'), 1.8)
    df_flights['AOG_scale_weight'] = df_flights['AOG_scale_weight'].mask(
        df_flights['simulation_run_id'].str.contains('sw2'), 2)

    # Find average cost per simulation iteration
    columns_costs = ['cost_total', 'cost_future_value', 'cost_cancellation', 'cost_compensation']  # , 'cost_food']
    for column in columns_costs:
        df_flights[column + '_iter'] = df_flights.groupby(df_flights['simulation_id'])[column].transform('sum')
        # In millions, for a year instead of six months
        df_flights[column + '_iter'] = df_flights[column + '_iter'] * 2 / 10 ** 6
    # df_flights['cost_iter_total'] = df_flights.groupby(df_flights['simulation_id'])['cost_total'].transform('sum')

    df_costs = df_flights[['simulation_id', 'simulation_run_id', 'reserve_scenario', 'AOG_scale_weight',
                           'reserve_scenario_string'] + [cl + '_iter' for cl in columns_costs]]
    df_costs = df_costs.drop_duplicates()
    suffix_scenario_avg = '_scenario_avg'
    for column in columns_costs:
        df_costs[column + suffix_scenario_avg] = df_costs[column + '_iter'].groupby(
            df_costs['simulation_run_id']).transform('mean')
        df_costs = add_ci_norm(df_costs, column + '_iter')
        df_costs = add_ci_bootstrap_diff_reserve0(df_costs, column + '_iter')
    # df_costs['cost_scenario_avg'] = df_costs['cost_iter_total'].groupby(df_costs['simulation_run_id']).transform('mean')/10**6
    df_costs = df_costs.drop(columns=['simulation_id'] + [cl + '_iter' for cl in columns_costs])
    df_costs = df_costs.drop_duplicates()

    # RESERVE VALUE
    suffix_reserve = '_rs'
    df_reserves = df_costs[df_costs['reserve_scenario']]
    df_reserves = df_reserves.drop(columns=['simulation_run_id', 'reserve_scenario', 'reserve_scenario_string'])
    df_value = pd.merge(df_costs[df_costs['reserve_scenario'] == 1], df_costs[df_costs['reserve_scenario'] == 0],
                        how='left', on='AOG_scale_weight', suffixes=(suffix_reserve, '' ))
    suffix_value = '_value'
    for column in columns_costs:
        column_name = column + suffix_scenario_avg
        column_rs_name = column_name + suffix_reserve
        column_value_name = column + suffix_value
        df_value[column_value_name] =  df_value[column_rs_name] - df_value[column_name]

    # COMPLETION FACTOR
    # Remove future rotations
    df_cf = df_flights[df_flights['execution_state'] != 'future_rotation']
    df_cf['fl_executed'] = df_cf['execution_state'] == 'executed'
    df_cf['fl_cancelled'] = df_cf['execution_state'].str.contains('cancelled')
    df_cf['fl_executed_count'] = df_cf.groupby('simulation_id')['fl_executed'].transform('sum')
    df_cf['fl_cancelled_count'] = df_cf.groupby('simulation_id')['fl_cancelled'].transform('sum')
    df_cf['compf_iteration'] = df_cf['fl_executed_count'] / (
                df_cf['fl_executed_count'] + df_cf['fl_cancelled_count']) * 100
    df_cf['cancf_iteration'] = df_cf['fl_cancelled_count'] / (
                df_cf['fl_executed_count'] + df_cf['fl_cancelled_count']) * 100
    df_cf = df_cf.drop_duplicates('simulation_id')
    # df_cf_iters = df_cf.copy()
    df_cf['fl_executed_avg'] = df_cf.groupby('simulation_run_id')['fl_executed_count'].transform('mean')
    df_cf['fl_cancelled_avg'] = df_cf.groupby('simulation_run_id')['fl_cancelled_count'].transform('mean')
    df_cf['compf_avg'] = df_cf.groupby('simulation_run_id')['compf_iteration'].transform('mean')
    df_cf['cancf_avg'] = df_cf.groupby('simulation_run_id')['cancf_iteration'].transform('mean')
    # Confidence interval 95%
    df_cf = add_ci_bootstrap(df_cf, 'cancf_iteration')
    df_cf = df_cf.drop_duplicates('simulation_run_id')
    df_cf = df_cf[['simulation_run_id', 'reserve_scenario', 'AOG_scale_weight', 'reserve_scenario_string',
                   'fl_executed_avg', 'fl_cancelled_avg', 'compf_avg', 'cancf_avg', 'cancf_iteration_ci_min',
                   'cancf_iteration_ci_max']]

    # REDUCED CF
    df_reserves = df_cf[df_cf['reserve_scenario']]
    df_reserves = df_reserves.drop(columns=['simulation_run_id', 'reserve_scenario', 'reserve_scenario_string'])
    df_reduced_cf = pd.merge(df_cf[df_cf['reserve_scenario'] == 0], df_reserves,
                             how='left', on='AOG_scale_weight', suffixes=('', suffix_reserve))
    df_reduced_cf['cancf_reduced'] = df_reduced_cf['cancf_avg'] - df_reduced_cf['cancf_avg' + suffix_reserve]

    # DELAYS
    df_delays = df_flights[df_flights['execution_state'] == 'executed']
    columns_delays = ['delay_arrival', 'delay_departure']
    for column in columns_delays:
        df_delays[column + '_iter'] = df_delays.groupby(df_delays['simulation_id'])[column].transform('mean')

    # AVERAGE DELAY
    df_delays_run = df_delays.drop_duplicates('simulation_id')
    df_delays_run['delay_departure_avg'] = df_delays_run.groupby('simulation_run_id')['delay_departure_iter'].transform(
        'mean')
    df_delays_run['delay_arrival_avg'] = df_delays_run.groupby('simulation_run_id')['delay_arrival_iter'].transform(
        'mean')
    # Confidence interval 95%
    columns_ci = ['delay_departure_iter', 'delay_arrival_iter']
    for col_ci in columns_ci:
        df_delays_run = add_ci_norm(df_delays_run, col_ci)

    df_delays_run = df_delays_run[['simulation_run_id', 'reserve_scenario', 'AOG_scale_weight',
                                   'reserve_scenario_string', 'delay_departure_avg', 'delay_arrival_avg']
                                  + [cl + '_ci_min' for cl in columns_ci] + [cl + '_ci_max' for cl in columns_ci]]
    df_delays_run = df_delays_run.drop_duplicates('simulation_run_id')

    # LONG DELAY
    delay_durations = [60, 120, 180, 240, 1000]
    for i in range(len(delay_durations) - 1):
        # for delay in delay_durations:
        delay = delay_durations[i]
        delay_max = delay_durations[i + 1]
        df_delays['delay' + str(delay)] = (df_delays['delay_arrival'] >= delay) & (
                    df_delays['delay_arrival'] < delay_max)
    delay_durations.remove(1000)
    # df_delays['delay3'] = df_delays['delay_arrival']>3*60
    # df_delays['delay4'] = df_delays['delay_arrival']>4*60
    columns_delays_long = ['delay' + str(dl) for dl in delay_durations]
    # columns_delays_long = ['delay3', 'delay4']
    df_delays['group_count'] = df_delays.groupby(df_delays['simulation_id'])['simulation_id'].transform(len)
    for column in columns_delays_long:
        df_delays[column + '_iter'] = df_delays.groupby(df_delays['simulation_id'])[column].transform('sum')
        df_delays[column + '_cf_iter'] = df_delays[column + '_iter'] / df_delays['group_count'] * 100
    df_delays_long = df_delays.drop_duplicates('simulation_id')

    suffix_delay_avg = '_avg'
    for delay in delay_durations:
        df_delays_long['delay' + str(delay) + suffix_delay_avg] = df_delays_long.groupby('simulation_run_id') \
            ['delay' + str(delay) + '_iter'].transform('mean')
        df_delays_long['delay' + str(delay) + '_cf' + suffix_delay_avg] = df_delays_long.groupby('simulation_run_id') \
            ['delay' + str(delay) + '_cf_iter'].transform('mean')
    # df_delays_long['delay3_avg'] = df_delays_long.groupby('simulation_run_id')['delay3_iter'].transform('mean')
    # df_delays_long['delay4_avg'] = df_delays_long.groupby('simulation_run_id')['delay4_iter'].transform('mean')
    # df_delays_long['delay3_cf_avg'] = df_delays_long.groupby('simulation_run_id')['delay3_cf_iter'].transform('mean')
    # df_delays_long['delay4_cf_avg'] = df_delays_long.groupby('simulation_run_id')['delay4_cf_iter'].transform('mean')
    df_delays_long = df_delays_long[['simulation_run_id', 'reserve_scenario', 'AOG_scale_weight',
                                     'reserve_scenario_string'] + ['delay' + str(dl) + '_cf' + suffix_delay_avg for dl
                                                                   in delay_durations]]
    df_delays_long = df_delays_long.drop_duplicates('simulation_run_id')

    print('Time elaboration: ', (timeit.default_timer() - time_now) / 60)
    # breakpoint()
    ############################### PLOTS COMPLETION FACTOR ###############################
    colors = {'1 reserve': 'C0', '2 reserves': 'C1', '0 reserves': 'C3'}
    colors_sw = {1: 'C0', 1.2: 'C1', 1.4: 'C2', 1.6: 'C3', 1.8: 'C4', 2: 'C5'}

    # # CANCELLATION FACTOR #TODO old scatter
    # fig, ax = plt.subplots()
    # plt.suptitle('Cancellation factor for different AOG duration scenarios')
    # for reserve in ['1 reserve', '2 reserves']:
    #     rows_rs = df_cf[df_cf['reserve_scenario_string'] == reserve]
    #     ax.scatter(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], c=colors[reserve], label=reserve)
    # ax.set(ylabel='Cancellation factor [%]')
    # ax.set(xlabel=scale_weight_name)
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0)
    # ax.legend()

    x_ticks_cf = np.arange(1, 2.2, 0.2)
    y_ticks_cf = np.arange(0, 0.45, 0.05)

    # CANCELLATION FACTOR WITH CI
    df_cf = df_cf.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Cancellation factor for different AOG duration scenarios')
    for reserve in ['1 reserve', '2 reserves', '0 reserves']:
        rows_rs = df_cf[df_cf['reserve_scenario_string'] == reserve]
        error_bars = [(rows_rs['cancf_avg'] - rows_rs['cancf_iteration_ci_min'].to_list()),
                      (rows_rs['cancf_iteration_ci_max'] - rows_rs['cancf_avg']).to_list()]
        ax.errorbar(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], yerr=error_bars,
                    c=colors[reserve], fmt='-o', label=reserve)
        # ax.plot(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], c=colors[reserve], label=reserve)
        ax.fill_between(rows_rs['AOG_scale_weight'], rows_rs['cancf_iteration_ci_min'],
                        rows_rs['cancf_iteration_ci_max'],
                        color=colors[reserve], alpha=0.1)
    ax.set(ylabel='Cancellation factor [%]')
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    ax.legend(loc='upper left')
    ax.set_xticks(x_ticks_cf)
    ax.set_yticks(y_ticks_cf)

    # CANCELLATION FACTOR WITH CI ONLY BASELINE
    df_cf = df_cf.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Cancellation factor for different AOG duration scenarios')
    reserve = '1 reserve'
    rows_rs = df_cf[(df_cf['reserve_scenario_string'] == reserve) & (df_cf['AOG_scale_weight'] == 1)]
    error_bars = [(rows_rs['cancf_avg'] - rows_rs['cancf_iteration_ci_min'].to_list()),
                  (rows_rs['cancf_iteration_ci_max'] - rows_rs['cancf_avg']).to_list()]
    ax.errorbar(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], yerr=error_bars,
                c=colors[reserve], fmt='-o', label=reserve)
    ax.set(ylabel='Cancellation factor [%]')
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    ax.legend(loc='upper left')
    ax.set_xticks(x_ticks_cf)
    ax.set_yticks(y_ticks_cf)

    # CANCELLATION FACTOR WITH CI BASELINE + RESERVE
    df_cf = df_cf.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Cancellation factor for different AOG duration scenarios')
    for reserve in ['1 reserve', '2 reserves', '0 reserves']:
        rows_rs = df_cf[(df_cf['reserve_scenario_string'] == reserve) & (df_cf['AOG_scale_weight'] == 1)]
        error_bars = [(rows_rs['cancf_avg'] - rows_rs['cancf_iteration_ci_min'].to_list()),
                      (rows_rs['cancf_iteration_ci_max'] - rows_rs['cancf_avg']).to_list()]
        ax.errorbar(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], yerr=error_bars,
                    c=colors[reserve], fmt='-o', label=reserve)
    ax.set(ylabel='Cancellation factor [%]')
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    ax.legend(loc='upper left')
    ax.set_xticks(x_ticks_cf)
    ax.set_yticks(y_ticks_cf)

    # CANCELLATION FACTOR WITH CI BASELINE + RESERVE + SCALE WEIGHTS
    df_cf = df_cf.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Cancellation factor for different AOG duration scenarios')
    for reserve in ['1 reserve']:
        rows_rs = df_cf[df_cf['reserve_scenario_string'] == reserve]
        error_bars = [(rows_rs['cancf_avg'] - rows_rs['cancf_iteration_ci_min'].to_list()),
                      (rows_rs['cancf_iteration_ci_max'] - rows_rs['cancf_avg']).to_list()]
        ax.errorbar(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], yerr=error_bars,
                    c=colors[reserve], fmt='-o', label=reserve)
        # ax.plot(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], c=colors[reserve], label=reserve)
        ax.fill_between(rows_rs['AOG_scale_weight'], rows_rs['cancf_iteration_ci_min'],
                        rows_rs['cancf_iteration_ci_max'],
                        color=colors[reserve], alpha=0.1)
    reserve = '2 reserves'
    rows_rs = df_cf[(df_cf['reserve_scenario_string'] == reserve) & (df_cf['AOG_scale_weight'] == 1)]
    error_bars = [(rows_rs['cancf_avg'] - rows_rs['cancf_iteration_ci_min'].to_list()),
                  (rows_rs['cancf_iteration_ci_max'] - rows_rs['cancf_avg']).to_list()]
    ax.errorbar(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], yerr=error_bars,
                c=colors[reserve], fmt='-o', label=reserve)
    ax.set(ylabel='Cancellation factor [%]')
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    ax.legend(loc='upper left')
    ax.set_xticks(x_ticks_cf)
    ax.set_yticks(y_ticks_cf)

    # REDUCED CANCELLATION FACTOR
    # fig, ax = plt.subplots()
    # plt.suptitle('Cancellation factor reduction through the use of an extra reserve')
    # ax.scatter(df_reduced_cf['AOG_scale_weight'], df_reduced_cf['cancf_reduced'], c='C6')
    # ax.set(ylabel='Cancellation factor reduction [%]')
    # ax.set(xlabel=scale_weight_name)
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0)

    # # AVERAGE DELAY # TODO old scatter
    # fig, ax = plt.subplots()
    # plt.suptitle('Flights arrival delay for different AOG duration scenarios')
    # for reserve in ['1 reserve', '2 reserves']:
    #     rows_rs = df_delays_run[df_delays_run['reserve_scenario_string'] == reserve]
    #     ax.scatter(rows_rs['AOG_scale_weight'], rows_rs['delay_arrival_avg'], c=colors[reserve], label=reserve)
    # ax.set(ylabel='Average arrival delay')
    # ax.set(xlabel=scale_weight_name)
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0, ymax=14)
    # ax.legend()

    # # AVERAGE DEPARTURE DELAY CI
    # df_delays_run = df_delays_run.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    # fig, ax = plt.subplots()
    # plt.suptitle('Average departure delay for different AOG duration scenarios')
    # for reserve in ['1 reserve', '2 reserves']:
    #     rows_rs = df_delays_run[df_delays_run['reserve_scenario_string'] == reserve]
    #     ax.plot(rows_rs['AOG_scale_weight'], rows_rs['delay_departure_avg'], c=colors[reserve], label=reserve)
    #     ax.fill_between(rows_rs['AOG_scale_weight'], rows_rs['delay_departure_iter_ci_min'],
    #                     rows_rs['delay_departure_iter_ci_max'],
    #                     color=colors[reserve], alpha=0.1)
    # ax.set(ylabel='Departure delay')
    # ax.set(xlabel=scale_weight_name)
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0)
    # ax.legend()
    #
    # # AVERAGE ARRIVAL DELAY CI
    # df_delays_run = df_delays_run.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    # fig, ax = plt.subplots()
    # plt.suptitle('Average arrival delay for different AOG duration scenarios')
    # for reserve in ['1 reserve', '2 reserves']:
    #     rows_rs = df_delays_run[df_delays_run['reserve_scenario_string'] == reserve]
    #     ax.plot(rows_rs['AOG_scale_weight'], rows_rs['delay_arrival_avg'], c=colors[reserve], label=reserve)
    #     ax.fill_between(rows_rs['AOG_scale_weight'], rows_rs['delay_arrival_iter_ci_min'],
    #                     rows_rs['delay_arrival_iter_ci_max'],
    #                     color=colors[reserve], alpha=0.1)
    # ax.set(ylabel='Arrival delay')
    # ax.set(xlabel=scale_weight_name)
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0)
    # ax.legend()

    # DELAYS CDF SCALE WEIGHT 1
    df_delays = df_delays.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('CDF of delays for an ' + scale_weight_name + ' of 1')
    for reserve in ['1 reserve', '2 reserves', '0 reserves']:
        sw = 1
        rows_rs = df_delays[(df_delays['reserve_scenario_string'] == reserve)
                            & (df_delays['AOG_scale_weight'] == sw)]
        delays_to_plot = [dl for dl in rows_rs['delay_departure'].to_list() if dl <= 60 * 8]
        bins = range(0, math.ceil(max(delays_to_plot) + 10), 10)
        ax.hist(delays_to_plot, color=colors[reserve], label=reserve,  # colors_sw[sw]
                histtype='step', bins=bins, linestyle='solid', density=True, cumulative=-1)  # 'step'
    # ax.set(ylabel='Long delays [%]')
    ax.set(xlabel='Departure delay [min]', ylabel='Cumulative distribution')
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)  # , ymax=1)
    ax.legend()
    ax.set_xticks(range(0, math.ceil(max(delays_to_plot) + 10), 20))

    #
    # # DELAYS CDF NEW
    # df_delays = df_delays.sort_values(['AOG_scale_weight', 'reserve_scenario_string'])
    # fig, ax = plt.subplots()
    # plt.suptitle('CDF of delays per scenario')
    # data_delay = []
    # labels_reserve = []
    # colors_delay = []
    # for sw in list(set(df_delays['AOG_scale_weight'])):
    #     for reserve in ['1 reserve', '2 reserves']:
    #         if reserve == '1 reserve':
    #             line_style = 'solid'
    #         else:
    #             line_style = '--'
    #         rows_rs = df_delays[(df_delays['reserve_scenario_string'] == reserve)
    #                             & (df_delays['AOG_scale_weight'] == sw)]
    #         delays_to_plot = [dl / 60 for dl in rows_rs['delay_departure'].to_list() if dl > 0 and dl <= 60 * 8]
    #         data_delay.append(delays_to_plot)
    #         labels_reserve.append(reserve)
    #         colors_delay.append(colors[reserve])
    # bins = range(0, math.ceil(max([el for ls in data_delay for el in ls]) + 1))
    # ax.hist(data_delay, bins=bins, label=labels_reserve, color=colors_delay,# label=reserve + ' sw ' + str(sw), # color=colors[# reserve], # colors_sw[sw]
    #         histtype='bar', linestyle=line_style, density=True, cumulative=-1)  # 'step'
    # # ax.set(ylabel='Long delays [%]')
    # ax.set(xlabel='Departure delay [hours]')
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0)  # , ymax=1)
    # ax.legend()

    # DELAYS CDF NEW 2
    df_delays = df_delays.sort_values(['AOG_scale_weight', 'reserve_scenario_string'])
    fig, ax = plt.subplots()
    plt.suptitle('CDF of delays per scenario')

    for reserve in ['1 reserve', '2 reserves', '0 reserves']:
        data_delay = []
        labels_reserve = []
        labels_sw = []
        colors_delay = []
        if reserve == '1 reserve':
            line_style = 'solid'
            fill = True
            hatch = None  # str('\\\\')
            edgecolor = colors[reserve]
            linewidth = 1
            alpha = 1
        else:
            line_style = 'solid'
            fill = False
            hatch = None  # '//'
            edgecolor = colors[reserve]
            linewidth = 1
            alpha = .5
        sw_values = list(set(df_delays['AOG_scale_weight']))
        sw_values = sorted(sw_values)
        for sw in sw_values:

            rows_rs = df_delays[(df_delays['reserve_scenario_string'] == reserve)
                                & (df_delays['AOG_scale_weight'] == sw)]
            delays_to_plot = [dl / 60 for dl in rows_rs['delay_departure'].to_list() if dl <= 60 * 8]  # dl > 0 and ]
            data_delay.append(delays_to_plot)
            if len(labels_reserve) < 1:
                labels_reserve.append(reserve)
            colors_delay.append(colors[reserve])
        # bins = range(0, math.ceil(max([el for ls in data_delay for el in ls]) + 1))
        bins = np.arange(0, math.ceil(max([el for ls in data_delay for el in ls]) + 0.5), 0.5)
        _, b, _ = ax.hist(data_delay, bins=bins, label=labels_reserve, color=colors_delay,
                          # label=reserve + ' sw ' + str(sw), # color=colors[# reserve], # colors_sw[sw]
                          histtype='bar', linestyle=line_style, density=True, cumulative=-1, fill=fill, hatch=hatch,
                          edgecolor=edgecolor, linewidth=linewidth)  # 'step'
        rects = ax.patches
        sw_labels = [sw for sw in sw_values for i in range(len(b) - 1)]
        for rect, label in zip(rects, sw_labels):
            # height = rect.get_height()
            # ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label, ha='center', va='bottom')
            ax.text(rect.get_x() + rect.get_width() / 2, - 0.008, label, ha='center', va='bottom', size=8,
                    rotation='vertical')
        # ax.bar_label(rects, labels=labels_sw)
        if reserve == '2 reserves':
            ax.hist(data_delay, bins=bins, color=colors_delay,
                    # label=reserve + ' sw ' + str(sw), # color=colors[# reserve], # colors_sw[sw]
                    histtype='bar', linestyle=line_style, density=True, cumulative=-1, fill=True, hatch=None,
                    edgecolor=None, alpha=.4, linewidth=0)  # 'step'
    # ax.set(ylabel='Long delays [%]')
    ax.set(xlabel='Departure delay [hours]')
    ax.set_xticks(bins)
    ax.tick_params(axis='x', length=100, size=10)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)  # , ymax=1)
    ax.legend()

    # # LONG DELAY
    # fig, ax = plt.subplots()
    # plt.suptitle('Occurrences of flights with arrival delay over three hours')
    # for reserve in ['1 reserve', '2 reserves']:
    #     rows_rs = df_delays_long[df_delays_long['reserve_scenario_string'] == reserve]
    #     ax.scatter(rows_rs['AOG_scale_weight'], rows_rs['delay180_cf_avg'], c=colors[reserve], label=reserve)
    # ax.set(ylabel='Long delays [%]')
    # ax.set(xlabel=scale_weight_name)
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0, ymax=0.4)
    # ax.legend()

    # # DELAY BRACKETS
    # fig, axs = plt.subplots(ncols=len(delay_durations), sharex=True, sharey=True)
    # plt.suptitle('Percentage of flights within delay bracket')
    # for columns_count in range(len(delay_durations)):
    #     column = delay_durations[columns_count]
    #     column_name = 'delay' + str(column) + '_cf' + suffix_delay_avg
    #     for reserve in ['1 reserve', '2 reserves']:
    #         rows_rs = df_delays_long[df_delays_long['reserve_scenario_string'] == reserve]
    #         axs[columns_count].scatter(rows_rs['AOG_scale_weight'], rows_rs[column_name], c=colors[reserve], label = reserve)
    #     axs[columns_count].grid(axis='y')
    #     axs[columns_count].set_title(str(column) + ' minutes')
    #     axs[columns_count].set(xlabel=scale_weight_name)
    #
    # axs[0].set(ylabel='[%]')
    # axs[columns_count].set_ylim(ymin=0)
    # axs[columns_count].legend()
    # axs[columns_count].set_xticks(np.linspace(1,2,6))
    # axs[columns_count].set_yticks(np.linspace(0, 2.5, 11))

    ############################### PLOTS COSTS ###############################

    # colors = {'1 reserve': 'C0', '2 reserves': 'C1'}

    # # ALL COSTS # TODO old scatter
    # fig, axs = plt.subplots(ncols=len(columns_costs), sharex=True, sharey=True)
    # plt.suptitle('Cost of disruptions for different AOG duration scenarios')
    # for columns_count in range(len(columns_costs)):
    #     column = columns_costs[columns_count]
    #     column_name = column + suffix_scenario_avg
    #     for reserve in ['1 reserve', '2 reserves']:
    #         rows_rs = df_costs[df_costs['reserve_scenario_string'] == reserve]
    #         axs[columns_count].scatter(rows_rs['AOG_scale_weight'], rows_rs[column_name], c=colors[reserve], label = reserve)
    #     axs[columns_count].grid(axis='y')
    #     axs[columns_count].set_title(column.replace('_', ' ').replace('cost', ''))
    #     axs[columns_count].set(xlabel=scale_weight_name)
    #
    # axs[0].set(ylabel='[Million €]')
    # axs[columns_count].set_ylim(ymin=0)
    # axs[columns_count].legend()
    # axs[columns_count].set_xticks(np.linspace(1,2,6))
    # axs[columns_count].set_yticks(np.linspace(0, 40, 21))

    # ALL COSTS WITH CI
    df_costs = df_costs.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, axs = plt.subplots(ncols=len(columns_costs), sharex=True, sharey=True)
    plt.suptitle('Cost of disruptions for different AOG duration scenarios')
    for columns_count in range(len(columns_costs)):
        column = columns_costs[columns_count]
        column_name = column + suffix_scenario_avg
        for reserve in ['1 reserve', '2 reserves', '0 reserves']:
            rows_rs = df_costs[df_costs['reserve_scenario_string'] == reserve]
            error_bars = [(rows_rs[column_name] - rows_rs[column + '_iter_ci_min'].to_list()),
                          (rows_rs[column + '_iter_ci_max'] - rows_rs[column_name]).to_list()]
            axs[columns_count].errorbar(rows_rs['AOG_scale_weight'], rows_rs[column_name], yerr=error_bars,
                                        c=colors[reserve], fmt='-o', label=reserve, markersize=4)
            axs[columns_count].fill_between(rows_rs['AOG_scale_weight'], rows_rs[column + '_iter_ci_min'],
                                            rows_rs[column + '_iter_ci_max'], color=colors[reserve], alpha=0.1)
        axs[columns_count].grid(axis='y')
        axs[columns_count].set_title(column.replace('_', ' ').replace('cost', ''))
        axs[columns_count].set(xlabel=scale_weight_name)

    axs[0].set(ylabel='[Million €]')
    axs[columns_count].set_ylim(ymin=0)
    axs[columns_count].legend()
    axs[columns_count].set_xticks(np.linspace(1, 2, 6))
    axs[columns_count].set_yticks(np.arange(0, 61, 5))

    # # TOTAL AND CANCELLATIONS
    # columns_to_plot = ['cost_total', 'cost_cancellation']
    # fig, axs = plt.subplots(nrows=len(columns_to_plot), sharex=True)
    # plt.suptitle('Cost of disruptions for different AOG duration scenarios')
    # for columns_count in range(len(columns_to_plot)):
    #     column = columns_to_plot[columns_count]
    #     column_name = column + suffix_scenario_avg
    #     for reserve in ['1 reserve', '2 reserves']:
    #         rows_rs = df_costs[df_costs['reserve_scenario_string'] == reserve]
    #         axs[columns_count].scatter(rows_rs['AOG_scale_weight'], rows_rs[column_name], c=colors[reserve], label = reserve)
    #     axs[columns_count].set(ylabel=column+' [Million €]')
    #     axs[columns_count].grid(axis='y')
    #     axs[columns_count].set_title(column.replace('_', ' ').replace('cost', ''))
    #     axs[columns_count].set_ylim(ymin=0)
    # axs[columns_count].set(xlabel=scale_weight_name)
    # axs[columns_count].set_xticks(np.linspace(1, 2, 6))
    # axs[0].legend()

    # # RESERVE VALUE SCATTER # TODO old scatter
    # fig, ax = plt.subplots()
    # plt.suptitle('Avoided costs of disruptions through an extra reserve')
    # ax.scatter(df_value['AOG_scale_weight'], df_value['cost_total'+suffix_value], c='C6')
    # ax.set(ylabel='[Million €]')
    # ax.set(xlabel=scale_weight_name)
    # ax.grid(axis='y')
    # ax.set_ylim(ymin=0)
    # ax.set_xticks(np.linspace(1, 2, 6))

    plt.show()
    df_value = df_value[df_value['reserve_scenario_string_rs']!='2 reserves']
    # RESERVE VALUE PLOT WITH CI
    df_value = df_value.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Avoided costs of disruptions through an extra reserve')
    error_bars = [(df_value['cost_total' + suffix_value] - df_value['cost_total_iter_diff_ci_min_rs'].to_list()),
                  (df_value['cost_total_iter_diff_ci_max_rs'] - df_value['cost_total' + suffix_value]).to_list()]
    ax.errorbar(df_value['AOG_scale_weight'], df_value['cost_total' + suffix_value],yerr=error_bars,
                c='C6', fmt='-o')
    # ax.plot(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], c=colors[reserve], label=reserve)
    ax.fill_between(df_value['AOG_scale_weight'], df_value['cost_total_iter_diff_ci_min'],
                    df_value['cost_total_iter_diff_ci_max'],
                    color='C6', alpha=0.1)
    ax.set(ylabel='[Million €]')
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    # ax.legend()

    # RESERVE VALUE COMPONENTS WITH CI
    df_value = df_value.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Components of avoided costs of disruptions through an extra reserve')
    columns_to_plot = [cl for cl in columns_costs if cl != 'cost_total']
    for column in columns_to_plot:
        error_bars = [(df_value[column + suffix_value] - df_value[column + '_iter_diff_ci_min_rs'].to_list()),
                      (df_value[column + '_iter_diff_ci_max_rs'] - df_value[column + suffix_value]).to_list()]
        ax.errorbar(df_value['AOG_scale_weight'], df_value[column + suffix_value], yerr=error_bars,
                    fmt='-o', label=column)
        # ax.plot(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], c=colors[reserve], label=reserve)
        ax.fill_between(df_value['AOG_scale_weight'], df_value[column + '_iter_diff_ci_min'],
                        df_value[column + '_iter_diff_ci_max'], alpha=0.1)
    ax.set(ylabel='[Million €]')
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    ax.legend()

    # RESERVE VALUE BAR
    fig, ax = plt.subplots()
    plt.suptitle('Avoided costs of disruptions through an extra reserve (per component)')
    bottom = [0 for el in list(set(df_value['AOG_scale_weight'].to_list()))]
    for cost_col in [cl for cl in columns_costs if cl != 'cost_total']:
        print(cost_col + suffix_value)
        ax.bar(df_value['AOG_scale_weight'], df_value[cost_col + suffix_value],
               label='avoided ' + cost_col.replace('_', ' '), width=0.15, bottom=bottom)
        bottom = list(map(add, bottom, df_value[cost_col + suffix_value].to_list()))
    ax.set(ylabel='[Million €]')
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    ax.set_xticks(np.linspace(1, 2, 6))
    ax.legend()

    # RESERVE VALUE STACKPLOT
    df_value = df_value.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Avoided costs of disruptions through an extra reserve (per component)')
    values_all = []
    label_costs = []
    columns_to_plot = ['cost_compensation', 'cost_future_value', 'cost_cancellation']
    for cost_col in columns_to_plot:  # [cl for cl in columns_costs if cl!='cost_total']:
        values = df_value[cost_col + suffix_value]
        values_all.append(values)
        label_costs.append(cost_col)
    values_plot = np.vstack(values_all)
    ax.stackplot(df_value['AOG_scale_weight'], values_plot, labels=label_costs)
    # colors=['C'+str(i+2) for i in range(len(label_costs))])
    ax.set(ylabel='[Million €]')
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    ax.set_xticks(np.linspace(1, 2, 6))
    ax.legend()

    # RESERVE VALUE STACKPLOT AGGREGATED DELAY PLOT
    df_value = df_value.sort_values(['reserve_scenario_string', 'AOG_scale_weight'])
    fig, ax = plt.subplots()
    plt.suptitle('Avoided costs of disruptions through an extra reserve (per component)')
    values_all = []
    label_costs = []
    columns_to_plot = ['cost_delay', 'cost_cancellation']
    for cost_col in columns_to_plot:  # [cl for cl in columns_costs if cl!='cost_total']:
        values = df_value[cost_col + suffix_value]
        values_all.append(values)
        label_costs.append(cost_col)
    values_plot = np.vstack(values_all)
    ax.stackplot(df_value['AOG_scale_weight'], values_plot, labels=label_costs)
    # colors=['C'+str(i+2) for i in range(len(label_costs))])
    ax.set(ylabel='[Million €]')
    ax.set(xlabel=scale_weight_name)
    ax.grid(axis='y')
    ax.set_ylim(ymin=0)
    ax.set_xticks(np.linspace(1, 2, 6))
    ax.legend()

    # Box plot total costs
    # weights = list(set(df_costs['AOG_scale_weight'].to_list()))
    # fig, axs = plt.subplots(nrows=1, ncols=len(weights), sharey=True, sharex=True)
    # weights_count = 0
    # for weight in weights:
    #     rows_weight = df_costs[df_costs['AOG_scale_weight'] == weight]
    #     axs[weights_count].scatter(rows_weight['reserve_scenario_string'], rows_weight['cost_scenario_avg'],)
    #
    #     weights_count += 1

    plt.show()
    print('hello')

# UNUSED: only referenced by commented-out call sites (legacy from old repo).
def reduce_dashboard_results():
    # Directory original full file
    filename = 'dashboard_' + RESULTS.FILE_NAMES['flights']
    directory_input = os.path.join(directories.dashboard, 'Case_reserve_reduced')
    # Directory where the reduced file should be saved
    directory_output = os.path.join(directories.dashboard, 'Case_reserve_reduced')
    directory_output = os.path.join(directory_output, filename+'.csv')
    # Open file and reduce it
    df_flights = load_csv(filename=filename, directory_input=directory_input)#, nrows=2 * 10 ** 6)
    df_flights = df_flights[df_flights['iteration']<=10]
    df_flights = df_flights[df_flights['simulation_run_id'].str.contains('sw1_|sw16|sw2')]
    # Save file
    df_flights.to_csv(directory_output, index=False)


# def reserve_aircraft_cost():
#     df_fv = load_excel('Future_value')
#     fig, ax = plt.subplots()
#     plt.suptitle('Cancellation factor reduction through the use of an extra reserve')
#     ax.scatter(df_fv['Delay'], df_fv['ICA_non_elite'], c='C6')
#     ax.set(ylabel='Future value per pax')
#     ax.set(xlabel='Delay [min]')
#     ax.grid(axis='y')
#     ax.set_ylim(ymin=0)
#
#     filename = 'dashboard_' + RESULTS.FILE_NAMES['flights']
#     df_flights = load_csv(filename=filename, directory_input=directories.dashboard)#, nrows=2 * 10**6)
#     # Find if data from scenario including extra reserve
#     df_flights['reserve_scenario'] = (df_flights['simulation_run_id'].str.contains('_rs')) \
#                                      | (df_flights['simulation_run_id'] == '20230222_134230')
#     df_flights['reserve_scenario_string'] = '1 reserve'
#     df_flights['reserve_scenario_string'] = df_flights['reserve_scenario_string'].mask(df_flights['reserve_scenario'], '2 reserves')
#
#     # Find AOG scale weight from simulation name
#     df_flights['AOG_scale_weight'] = 1
#     df_flights['AOG_scale_weight'] = df_flights['AOG_scale_weight'].mask(df_flights['simulation_run_id'].str.contains('sw12'), 1.2)
#     df_flights['AOG_scale_weight'] = df_flights['AOG_scale_weight'].mask(df_flights['simulation_run_id'].str.contains('sw14'), 1.4)
#     df_flights['AOG_scale_weight'] = df_flights['AOG_scale_weight'].mask(df_flights['simulation_run_id'].str.contains('sw16'), 1.6)
#     df_flights['AOG_scale_weight'] = df_flights['AOG_scale_weight'].mask(df_flights['simulation_run_id'].str.contains('sw18'), 1.8)
#     df_flights['AOG_scale_weight'] = df_flights['AOG_scale_weight'].mask(df_flights['simulation_run_id'].str.contains('sw2'), 2)
#
#     # Find average cost per simulation iteration
#     columns_costs = [ 'cost_total', 'cost_future_value', 'cost_cancellation', 'cost_compensation', 'cost_food']
#     for column in columns_costs:
#         df_flights[column+'_iter'] = df_flights.groupby(df_flights['simulation_id'])[column].transform('sum')
#     # df_flights['cost_iter_total'] = df_flights.groupby(df_flights['simulation_id'])['cost_total'].transform('sum')
#
#     df_costs = df_flights[['simulation_id', 'simulation_run_id', 'reserve_scenario', 'AOG_scale_weight',
#                            'reserve_scenario_string']+[cl+'_iter' for cl in columns_costs]]
#     df_costs = df_costs.drop_duplicates()
#     suffix_scenario_avg = '_scenario_avg'
#     for column in columns_costs:
#         df_costs[column+suffix_scenario_avg] = df_costs[column+'_iter'].groupby(df_costs['simulation_run_id']).transform('mean')/10**6
#     # df_costs['cost_scenario_avg'] = df_costs['cost_iter_total'].groupby(df_costs['simulation_run_id']).transform('mean')/10**6
#     df_costs = df_costs.drop(columns=['simulation_id']+[cl+'_iter' for cl in columns_costs])
#     df_costs = df_costs.drop_duplicates()
#
#     # RESERVE VALUE
#     suffix_reserve = '_rs'
#     df_reserves = df_costs[df_costs['reserve_scenario']]
#     df_reserves = df_reserves.drop(columns=['simulation_run_id', 'reserve_scenario', 'reserve_scenario_string'])
#     df_value = pd.merge(df_costs[df_costs['reserve_scenario']==0], df_reserves,
#                         how='left', on='AOG_scale_weight', suffixes=('', suffix_reserve))
#     suffix_value = '_value'
#     for column in columns_costs:
#         column_name = column + suffix_scenario_avg
#         column_rs_name = column_name + suffix_reserve
#         column_value_name = column + suffix_value
#         df_value[column_value_name] = df_value[column_name] - df_value[column_rs_name]
#
#     # COMPLETION FACTOR
#     # Remove future rotations
#     df_cf = df_flights[df_flights['execution_state']!='future_rotation']
#     df_cf['fl_executed'] = df_cf['execution_state'] == 'executed'
#     df_cf['fl_cancelled'] = df_cf['execution_state'].str.contains('cancelled')
#     df_cf['fl_executed_count'] = df_cf.groupby('simulation_id')['fl_executed'].transform('sum')
#     df_cf['fl_cancelled_count'] = df_cf.groupby('simulation_id')['fl_cancelled'].transform('sum')
#     df_cf['compf_iteration'] = df_cf['fl_executed_count']/(df_cf['fl_executed_count'] + df_cf['fl_cancelled_count'])*100
#     df_cf['cancf_iteration'] = df_cf['fl_cancelled_count']/(df_cf['fl_executed_count'] + df_cf['fl_cancelled_count'])*100
#     df_cf = df_cf.drop_duplicates('simulation_id')
#     df_cf['fl_executed_avg'] = df_cf.groupby('simulation_run_id')['fl_executed_count'].transform('mean')
#     df_cf['fl_cancelled_avg'] = df_cf.groupby('simulation_run_id')['fl_cancelled_count'].transform('mean')
#     df_cf['compf_avg'] = df_cf.groupby('simulation_run_id')['compf_iteration'].transform('mean')
#     df_cf['cancf_avg'] = df_cf.groupby('simulation_run_id')['cancf_iteration'].transform('mean')
#     df_cf = df_cf.drop_duplicates('simulation_run_id')
#     df_cf = df_cf[['simulation_run_id', 'reserve_scenario', 'AOG_scale_weight', 'reserve_scenario_string',
#                    'fl_executed_avg', 'fl_cancelled_avg', 'compf_avg', 'cancf_avg']]
#
#     # REDUCED CF
#     df_reserves = df_cf[df_cf['reserve_scenario']]
#     df_reserves = df_reserves.drop(columns=['simulation_run_id', 'reserve_scenario', 'reserve_scenario_string'])
#     df_reduced_cf = pd.merge(df_cf[df_cf['reserve_scenario'] == 0], df_reserves,
#                              how='left', on='AOG_scale_weight', suffixes=('', suffix_reserve))
#     df_reduced_cf['cancf_reduced'] = df_reduced_cf['cancf_avg'] - df_reduced_cf['cancf_avg' +suffix_reserve]
#
#     # DELAYS
#     df_delays = df_flights[df_flights['execution_state']=='executed']
#     columns_delays = ['delay_arrival', 'delay_departure']
#     for column in columns_delays:
#         df_delays[column + '_iter'] = df_delays.groupby(df_delays['simulation_id'])[column].transform('mean')
#
#     # AVERAGE DELAY
#     df_delays_run = df_delays.drop_duplicates('simulation_id')
#     df_delays_run['delay_departure_avg'] = df_delays_run.groupby('simulation_run_id')['delay_departure_iter'].transform('mean')
#     df_delays_run['delay_arrival_avg'] = df_delays_run.groupby('simulation_run_id')['delay_arrival_iter'].transform('mean')
#     df_delays_run = df_delays_run[['simulation_run_id', 'reserve_scenario', 'AOG_scale_weight',
#                                    'reserve_scenario_string', 'delay_departure_avg', 'delay_arrival_avg']]
#     df_delays_run = df_delays_run.drop_duplicates('simulation_run_id')
#
#     # LONG DELAY
#     delay_durations = [60, 120, 180, 240, 1000]
#     for i in range(len(delay_durations)-1):
#     # for delay in delay_durations:
#         delay = delay_durations[i]
#         delay_max = delay_durations[i+1]
#         df_delays['delay'+str(delay)] = (df_delays['delay_arrival'] >= delay) & (df_delays['delay_arrival'] < delay_max)
#     delay_durations.remove(1000)
#     # df_delays['delay3'] = df_delays['delay_arrival']>3*60
#     # df_delays['delay4'] = df_delays['delay_arrival']>4*60
#     columns_delays_long = ['delay'+str(dl) for dl in delay_durations]
#     # columns_delays_long = ['delay3', 'delay4']
#     df_delays['group_count'] = df_delays.groupby(df_delays['simulation_id'])['simulation_id'].transform(len)
#     for column in columns_delays_long:
#         df_delays[column + '_iter'] = df_delays.groupby(df_delays['simulation_id'])[column].transform('sum')
#         df_delays[column + '_cf_iter'] = df_delays[column + '_iter']/df_delays['group_count']*100
#     df_delays_long = df_delays.drop_duplicates('simulation_id')
#
#     suffix_delay_avg = '_avg'
#     for delay in delay_durations:
#         df_delays_long['delay'+str(delay)+suffix_delay_avg] = df_delays_long.groupby('simulation_run_id')\
#             ['delay'+str(delay)+'_iter'].transform('mean')
#         df_delays_long['delay'+str(delay)+'_cf'+suffix_delay_avg] = df_delays_long.groupby('simulation_run_id')\
#             ['delay'+str(delay)+'_cf_iter'].transform('mean')
#     # df_delays_long['delay3_avg'] = df_delays_long.groupby('simulation_run_id')['delay3_iter'].transform('mean')
#     # df_delays_long['delay4_avg'] = df_delays_long.groupby('simulation_run_id')['delay4_iter'].transform('mean')
#     # df_delays_long['delay3_cf_avg'] = df_delays_long.groupby('simulation_run_id')['delay3_cf_iter'].transform('mean')
#     # df_delays_long['delay4_cf_avg'] = df_delays_long.groupby('simulation_run_id')['delay4_cf_iter'].transform('mean')
#     df_delays_long = df_delays_long[['simulation_run_id', 'reserve_scenario', 'AOG_scale_weight',
#                                      'reserve_scenario_string']+['delay'+str(dl)+'_cf'+suffix_delay_avg for dl in delay_durations]]
#     df_delays_long = df_delays_long.drop_duplicates('simulation_run_id')
#
#     ############################### PLOTS COMPLETION FACTOR ###############################
#     colors = {'1 reserve': 'C0', '2 reserves': 'C1'}
#     # CANCELLATION FACTOR
#     fig, ax = plt.subplots()
#     plt.suptitle('Cancellation factor for different AOG duration scenarios')
#     for reserve in ['1 reserve', '2 reserves']:
#         rows_rs = df_cf[df_cf['reserve_scenario_string'] == reserve]
#         ax.scatter(rows_rs['AOG_scale_weight'], rows_rs['cancf_avg'], c=colors[reserve], label=reserve)
#     ax.set(ylabel='Cancellation factor [%]')
#     ax.set(xlabel=scale_weight_name)
#     ax.grid(axis='y')
#     ax.set_ylim(ymin=0)
#     ax.legend()
#
#     # REDUCED CANCELLATION FACTOR
#     # fig, ax = plt.subplots()
#     # plt.suptitle('Cancellation factor reduction through the use of an extra reserve')
#     # ax.scatter(df_reduced_cf['AOG_scale_weight'], df_reduced_cf['cancf_reduced'], c='C6')
#     # ax.set(ylabel='Cancellation factor reduction [%]')
#     # ax.set(xlabel=scale_weight_name)
#     # ax.grid(axis='y')
#     # ax.set_ylim(ymin=0)
#
#     # AVERAGE DELAY
#     fig, ax = plt.subplots()
#     plt.suptitle('Flights arrival delay for different AOG duration scenarios')
#     for reserve in ['1 reserve', '2 reserves']:
#         rows_rs = df_delays_run[df_delays_run['reserve_scenario_string'] == reserve]
#         ax.scatter(rows_rs['AOG_scale_weight'], rows_rs['delay_arrival_avg'], c=colors[reserve], label=reserve)
#     ax.set(ylabel='Long delays [%]')
#     ax.set(xlabel=scale_weight_name)
#     ax.grid(axis='y')
#     ax.set_ylim(ymin=0, ymax=14)
#     ax.legend()
#
#     # LONG DELAY
#     fig, ax = plt.subplots()
#     plt.suptitle('Occurrences of flights with arrival delay over three hours')
#     for reserve in ['1 reserve', '2 reserves']:
#         rows_rs = df_delays_long[df_delays_long['reserve_scenario_string'] == reserve]
#         ax.scatter(rows_rs['AOG_scale_weight'], rows_rs['delay180_cf_avg'], c=colors[reserve], label=reserve)
#     ax.set(ylabel='Long delays [%]')
#     ax.set(xlabel=scale_weight_name)
#     ax.grid(axis='y')
#     ax.set_ylim(ymin=0, ymax=0.4)
#     ax.legend()
#
#     # DELAY BRACKETS
#     fig, axs = plt.subplots(ncols=len(delay_durations), sharex=True, sharey=True)
#     plt.suptitle('Percentage of flights within delay bracket')
#     for columns_count in range(len(delay_durations)):
#         column = delay_durations[columns_count]
#         column_name = 'delay' + str(column) + '_cf' + suffix_delay_avg
#         for reserve in ['1 reserve', '2 reserves']:
#             rows_rs = df_delays_long[df_delays_long['reserve_scenario_string'] == reserve]
#             axs[columns_count].scatter(rows_rs['AOG_scale_weight'], rows_rs[column_name], c=colors[reserve], label = reserve)
#         axs[columns_count].grid(axis='y')
#         axs[columns_count].set_title(str(column) + ' minutes')
#         axs[columns_count].set(xlabel=scale_weight_name)
#
#     axs[0].set(ylabel='[%]')
#     axs[columns_count].set_ylim(ymin=0)
#     axs[columns_count].legend()
#     axs[columns_count].set_xticks(np.linspace(1,2,6))
#     axs[columns_count].set_yticks(np.linspace(0, 2.5, 11))
#
#
#     ############################### PLOTS COSTS ###############################
#
#     colors = {'1 reserve': 'C0', '2 reserves': 'C1'}
#     # ALL COSTS
#     fig, axs = plt.subplots(ncols=len(columns_costs), sharex=True, sharey=True)
#     plt.suptitle('Cost of disruptions for different AOG duration scenarios')
#     for columns_count in range(len(columns_costs)):
#         column = columns_costs[columns_count]
#         column_name = column + suffix_scenario_avg
#         for reserve in ['1 reserve', '2 reserves']:
#             rows_rs = df_costs[df_costs['reserve_scenario_string'] == reserve]
#             axs[columns_count].scatter(rows_rs['AOG_scale_weight'], rows_rs[column_name], c=colors[reserve], label = reserve)
#         axs[columns_count].grid(axis='y')
#         axs[columns_count].set_title(column.replace('_', ' ').replace('cost', ''))
#         axs[columns_count].set(xlabel=scale_weight_name)
#
#     axs[0].set(ylabel='[Million €]')
#     axs[columns_count].set_ylim(ymin=0)
#     axs[columns_count].legend()
#     axs[columns_count].set_xticks(np.linspace(1,2,6))
#     axs[columns_count].set_yticks(np.linspace(0, 40, 21))
#
#     # TOTAL AND CANCELLATIONS
#     columns_to_plot = ['cost_total', 'cost_cancellation']
#     fig, axs = plt.subplots(nrows=len(columns_to_plot), sharex=True)
#     plt.suptitle('Cost of disruptions for different AOG duration scenarios')
#     for columns_count in range(len(columns_to_plot)):
#         column = columns_to_plot[columns_count]
#         column_name = column + suffix_scenario_avg
#         for reserve in ['1 reserve', '2 reserves']:
#             rows_rs = df_costs[df_costs['reserve_scenario_string'] == reserve]
#             axs[columns_count].scatter(rows_rs['AOG_scale_weight'], rows_rs[column_name], c=colors[reserve], label = reserve)
#         axs[columns_count].set(ylabel=column+' [Million €]')
#         axs[columns_count].grid(axis='y')
#         axs[columns_count].set_title(column.replace('_', ' ').replace('cost', ''))
#         axs[columns_count].set_ylim(ymin=0)
#     axs[columns_count].set(xlabel=scale_weight_name)
#     axs[columns_count].set_xticks(np.linspace(1, 2, 6))
#     axs[0].legend()
#
#
#     # RESERVE VALUE SCATTER
#     fig, ax = plt.subplots()
#     plt.suptitle('Avoided costs of disruptions through an extra reserve')
#     ax.scatter(df_value['AOG_scale_weight'], df_value['cost_total'+suffix_value], c='C6')
#     ax.set(ylabel='[Million €]')
#     ax.set(xlabel=scale_weight_name)
#     ax.grid(axis='y')
#     ax.set_ylim(ymin=0)
#     ax.set_xticks(np.linspace(1, 2, 6))
#
#     # RESERVE VALUE BAR
#     fig, ax = plt.subplots()
#     plt.suptitle('Avoided costs of disruptions through an extra reserve (per component)')
#     bottom = [0 for el in list(set(df_value['AOG_scale_weight'].to_list()))]
#     for cost_col in [cl for cl in columns_costs if cl!='cost_total']:
#         print(cost_col+suffix_value)
#         ax.bar(df_value['AOG_scale_weight'], df_value[cost_col+suffix_value],
#                label= 'avoided '+ cost_col.replace('_',' '), width=0.15, bottom=bottom)
#         bottom = list(map(add, bottom, df_value[cost_col+suffix_value].to_list()))
#     ax.set(ylabel='[Million €]')
#     ax.set(xlabel=scale_weight_name)
#     ax.grid(axis='y')
#     ax.set_ylim(ymin=0)
#     ax.set_xticks(np.linspace(1, 2, 6))
#     ax.legend()
#
#
#     # Box plot total costs
#     # weights = list(set(df_costs['AOG_scale_weight'].to_list()))
#     # fig, axs = plt.subplots(nrows=1, ncols=len(weights), sharey=True, sharex=True)
#     # weights_count = 0
#     # for weight in weights:
#     #     rows_weight = df_costs[df_costs['AOG_scale_weight'] == weight]
#     #     axs[weights_count].scatter(rows_weight['reserve_scenario_string'], rows_weight['cost_scenario_avg'],)
#     #
#     #     weights_count += 1
#
#     plt.show()
#     print('hello')



class AOG_duration_scenario:
    def __init__(self):
        self.model = {}

def generate_AOG_scenarios():
    ''' Generate new pickles of the distribution_AOG file to simulate different scenarios of AOG duration. Do so by
    modifying the scale parameter of the fitted lognormal distribution describing AOG duration'''
    name_file = 'distributions_AOG_2022'
    distr_aog_orig = read_pickle(name_file, directories.aog_distributions)
    fleets = [el['fleet'] for el in distr_aog_orig]
    fleets_list = [{'fleet': fl, 'distr':[]} for fl in fleets]

    for scale_weight in SCALE_WEIGHTS:
        aog_scenario_new = copy.deepcopy(distr_aog_orig)
        for fleet_data in aog_scenario_new:
            # Empy list of empirical data
            fleet_data['AOG_duration_empirical'] = []
            # Find parameters of new distribution
            fleet_distr = fleet_data['AOG_duration_fitted']
            scale_new = scale_weight * fleet_distr.model['scale']
            distr_new = lognorm
            loc = fleet_distr.model['loc']
            arg = fleet_distr.model['arg']
            # Generate new object for fitted data
            AOG_duration_new = AOG_duration_scenario()
            AOG_duration_new.model['distr'] = distr_new
            AOG_duration_new.model['scale'] = scale_new
            AOG_duration_new.model['arg'] = arg
            AOG_duration_new.model['loc'] = loc
            # Update AOG distribution data
            fleet_data['AOG_duration_fitted'] = AOG_duration_new
            # Add reference to scale weight
            fleet_data['scale_weight'] = scale_weight

            # Add to list for plotting
            fleet_dict = next(fl for fl in fleets_list if fl['fleet'] == fleet_data['fleet'])
            fleet_dict['distr'].append((scale_weight, AOG_duration_new))

        # Save new AOG disribution found
        scale_weight_string = str(scale_weight).replace('.','_')
        name_new = name_file + '_' + scale_weight_string
        write_pickle(aog_scenario_new, name_new, directories.aog_distributions)

    # Plot distributions
    for fleet_dict in fleets_list:
        data_analytical_duration = []
        data_means = []
        for fleet_distr in fleet_dict['distr']:
            scale_weight = fleet_distr[0]
            distribution = fleet_distr[1].model
            data_analytical_duration.append({'distr': distribution, 'label': 'scale_weight=' + str(scale_weight)})
            mean_new = distribution['distr'].mean(s=distribution['arg'], loc=distribution['loc'],
                                                  scale=distribution['scale'])
            data_means.append({'x': mean_new, 'label': 'scale_weight=' + str(scale_weight)})

    print('hello')

def plot_AOG_scenarios():
    distr_aog_orig = read_pickle('distributions_AOG_2022', directories.aog_distributions)
    for distr in distr_aog_orig:
        duration_fitted = distr['AOG_duration_fitted']

        scale = duration_fitted.model['scale']
        loc = duration_fitted.model['loc']
        arg = duration_fitted.model['arg']

        # duration_mean = duration_fitted.model['distr'].mean(s=arg, loc=loc, scale=scale)[0]
        # data_analytical_duration = [{'distr': duration_fitted, 'label': 'AOG duration fitted'}]

        data_analytical_duration = []
        data_means = []
        for scale_weight in SCALE_WEIGHTS:
            scale_new = scale*scale_weight
            distr_new = lognorm

            AOG_duration_new = AOG_duration_scenario()
            AOG_duration_new.model['distr'] = distr_new
            AOG_duration_new.model['arg'] = arg
            AOG_duration_new.model['loc'] = loc
            AOG_duration_new.model['scale'] = scale_new

            data_analytical_duration.append({'distr': AOG_duration_new, 'label':scale_weight_name+'='+str(scale_weight)})
            mean_new = distr_new.mean(s=arg, loc=loc, scale=scale_new)
            data_means.append({'x': mean_new, 'label': ''})# 'scale_weight='+str(scale_weight)})

        print_hist(full_data_empirical=[], full_data_analytical=data_analytical_duration,
                   title='AOG duration ' + distr['fleet'], x_label='[hours]', min_max=(0, 75),
                   vertical_lines=data_means)

    plt.show()
    print('hello')



