from simulation._module_tail_assignment import module_tail_assignment_functions
from simulation._module_maintenance_scheduler import module_maintenance_scheduler_functions
from simulation._module_recovery import module_disruptions_recovery
from config import MODULES


def initialize_simulation_modules():
    '''
    Initialize the simulation module for planning, scheduling and rescheduling tail assignment and maintenance
    task execution
    '''
    def decorator(Class):
        # Tail assignment
        tail_assignment_fun = module_tail_assignment_functions[MODULES.TAIL_ASSIGNMENT]
        setattr(Class, 'tail_assignment', tail_assignment_fun)
        # Maintenance tasks and slots scheduler
        maintenance_schedule_fun = module_maintenance_scheduler_functions[MODULES.MAINTENANCE_SCHEDULE]
        setattr(Class, 'schedule_maintenance', maintenance_schedule_fun)
        # Recovery module
        disruption_recovery_fun = module_disruptions_recovery[MODULES.DISRUPTIONS_RECOVERY]
        setattr(Class, 'recover_disruption', disruption_recovery_fun)
        return Class
    return decorator