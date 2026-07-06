import csv
from config import directories
import os
import logging

def directory_output(filename, simulation_id, output_type):
    ''' Returns the directory for an output file located in the output folder'''
    if output_type == 'log_map':
        directory_folder = directories.logs_map
    else:
        directory_folder = os.path.join(directories.output, simulation_id)
    directory_file = os.path.join(directory_folder,filename)
    return directory_file


def check_error_gantt(directory):
    ''' If a csv file cannot be accessed for logging, check if it is the gantt log file. If it is not, then raise a
    warning. '''
    if 'log_sim' not in directory:
        logging.warning('Unable to access directory '+directory )

def csv_generate_file(directory, list):
    ''' Generates a csv file including a first line from a list given as input'''
    try:
        with open(directory, 'w',newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(list)
    except:
        check_error_gantt()

def csv_append_line(directory, list):
    ''' Append a line to a csv file when a list is given as input'''
    try:
        with open(directory, 'a',newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(list)
    except:
        check_error_gantt(directory)


def csv_generate_or_append(filename, list, simulation_id, output_type='output', extension='.csv'):
    '''Generates a csv file or appends a line to it, when the file already exists'''
    # Find path given a file name
    directory_file = directory_output(filename+extension, simulation_id, output_type)
    # If the file does not exist, generate it
    if os.path.isfile(directory_file) == False:
        csv_generate_file(directory_file, list)
    # If file already exists, append line to it
    else:
        csv_append_line(directory_file, list)



def write_csv_from_dataframe(filename, dataframe, simulation_id, output_type='output', extension='.csv', index=False):
                             #date_format='%d/%m/%Y %H:%M'):
    filename_with_extension = filename + extension
    directory_file = directory_output(filename_with_extension, simulation_id, output_type)
    dataframe.to_csv(directory_file, index=index)#, date_format=date_format)

def log_info(*log_elements):
    log_string = ' '.join(str(ls) for ls in log_elements)
    logging.info(log_string)

def log_warning(*log_elements):
    log_string = ' '.join(str(ls) for ls in log_elements)
    log_string = 'WARNING: ' + log_string
    logging.warning(log_string)

def log_error(*log_elements, print_error=True):
    log_string = ' '.join(str(ls) for ls in log_elements)
    if print_error == True:
        log_string = 'ERROR: ' + log_string
    logging.error(log_string)
