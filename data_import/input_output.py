import os
import pandas as pd
import pickle

from config import directories, G



def load_csv(filename, file_extension='.csv', decimal='.', parse_dates=False, encoding='utf-8-sig',low_memory=True,
             directory_input=directories.input, localize_time=True, select_columns=None, skiprows=0, skipfooter=0,
             directory_full = None, columns_exclude_utc_localization=None, nrows=None):
    if directory_full == None:
        directory = os.path.join(directory_input, filename + file_extension)
    else:
        directory = directory_full
    df = pd.read_csv(directory, decimal=decimal, parse_dates=parse_dates, encoding=encoding, low_memory=low_memory,
                     skiprows=skiprows, skipfooter=skipfooter, nrows=nrows)
    # Reduce columns, if requested
    if select_columns != None:
        df = df[select_columns]
    df = detect_date_columns(df)
    if localize_time==True:
        df = localize_time_utc(df, columns_excluded=columns_exclude_utc_localization)
    df = strip_whitespace(df)
    return df


def load_excel(filename, file_extension='.xlsx', parse_dates=False, localize_time=True, dtype=None):
    directory = os.path.join(directories.input, filename + file_extension)
    df = pd.read_excel(directory, parse_dates=parse_dates, dtype=dtype)
    df = detect_date_columns(df)
    if localize_time == True:
        df = localize_time_utc(df)
    df = strip_whitespace(df)
    return df

def dataframe_from_blueLagoon_query(query, localize_time=True, columns_rename=None,
                                    columns_exclude_utc_localization=None):
    '''
    Returns dataframe including data specified in a SQL query executed on Blue Lagoon database

    :param query: SQL query to be executed
    :return: dataframe with requested data
    '''

    import pyodbc
    from sqlalchemy.engine import URL
    connection_string = "DRIVER={SQL Server};SERVER=YS002XSL;DATABASE=BlueLagoonMart"
    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

    from sqlalchemy import create_engine
    engine = create_engine(connection_url)

    # Generate dataframe
    df = pd.read_sql_query(query, engine)
    # If required, change column names
    if columns_rename!=None:
        df = df.rename(columns=columns_rename)

    df = detect_date_columns(df)
    if localize_time == True:
        df = localize_time_utc(df, columns_exclude_utc_localization)
    df = strip_whitespace(df)
    return df

# UNUSED: not called anywhere in the active pipeline (legacy from old repo).
def write_csv(filename, df, file_extension='.csv', index=True, directory_output=directories.output):
    directory = os.path.join(directory_output, filename + file_extension)
    df.to_csv(directory, index=index)


def strip_whitespace(df):
    columns = df.columns[df.dtypes == 'object']
    for obj_col in columns:
        if isinstance(df[obj_col].iloc[0], str):
            df[obj_col] = df[obj_col].str.strip()
    return df


def write_pickle(struct, filename, directory=directories.pickle):
    directory_pickle = os.path.join(directory, filename)
    with open(directory_pickle, 'wb') as file:
        pickle.dump(struct,file)

def read_pickle(filename, directory=directories.pickle):
    directory_pickle = os.path.join(directory, filename)
    with open(directory_pickle, 'rb') as file:
        struct = pickle.load(file)
    return struct


def detect_date_columns(df):
    """ Make the columns that contain a date in datetime format """
    date_columns = df.columns[df.columns.str.lower().str.contains('date')]
    for column in date_columns:
        df[column] = pd.to_datetime(df[column], errors='coerce')
    return df

def localize_time_utc(df, columns_excluded=None):
    ''' Localize Utc dates included in a dataframe. Specific columns can be excluded by specifying them in the list
    columns_excluded'''
    columns_utc = df.columns[df.columns.str.lower().str.contains('utc')]
    # Exclude specified columns
    if columns_excluded != None:
        columns_utc = [col for col in columns_utc if col not in columns_excluded]
    # columns_local = df.columns[df.columns.str.lower().str.contains('local')]
    for column in columns_utc:
        df[column] = pd.to_datetime(df[column], errors='coerce')
        df[column] = df[column].dt.tz_localize(G.TIMEZONE_UTC)
    return df
