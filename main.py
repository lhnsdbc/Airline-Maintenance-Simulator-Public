
from data_import.distributions import find_distributions_network
from data_import.distributions_maintenance_faults import find_distributions_maintenance_faults, \
                                                            find_distributions_maintenance_AOG
from simulation.run_anemos import run_anemos_scenarios
from output.verification_map import simulation_map
from config import RUN_CONFIG
from output.results_manage_files import concatenate_results_df_for_dashboard #, compute_flights_cost
from validation.historical_KPIs import compute_historical_KPIs
from validation.validation_network import validate_results
from validation.validation_maintenance import validate_results_maintenance
from output.case_reserve_aircraft import case_reserve_aircraft
from output.results_manage_files import change_simulation_id_in_results_files
from output.case_health import case_health

if __name__ == '__main__':
    match(RUN_CONFIG.MODE):
        case 0 | 1000: # Simulation
            run_anemos_scenarios()
        case 1: # Map
            simulation_map()
        case 2: # Generate output for dashboard
            concatenate_results_df_for_dashboard()

        # DATA IMPORT AND ELABORATION
        case 10 | 11: # Distributions disruptions
            find_distributions_network()
        case 12 | 13: # Distributions maintenance
            find_distributions_maintenance_faults()
        case 14: # Distributions AOG
            find_distributions_maintenance_AOG()

        # VALIDATION
        case 20: # Historical KPIs for validation
            compute_historical_KPIs()
        case 21:
            validate_results()
        case 22:
            validate_results_maintenance()


        # CASE STUDIES
        case 30: # Reserve aircraft
            case_reserve_aircraft()
        case 31:
            case_health()
        case 32:
            change_simulation_id_in_results_files('20230222_134230', '18_07_22_sw1_rs2_20230222_134230')
        case _:
            raise Exception('RUN_CONFIG.MODE value not supported')