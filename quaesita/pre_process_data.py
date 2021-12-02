#!/usr/bin/python

from datetime import time, datetime, timedelta
import os
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

DEBUG = False 

def generateDataJobEventsCSV(path='../job_events/'):

    r"""Returns job arrival count CSV from job events table from the google cluster 2011 dataset"""
    columns = ['time','missing info', 'job ID', 'event type', 'user', 'scheduling class', 'job name', 'logical job name']
    data_job = pd.DataFrame(columns=columns) # empty pandas dataframe with set columns 
    
    file_names = []
    for file in os.listdir(path): 
        file_names.append(file)

    if '.DS_Store' in file_names:
        file_names.remove('.DS_Store')
    file_names.sort()

    for file in file_names:
        file_path = path + file
        with gzip.open(file_path) as f:
            # create a buffer pandas dataframe 
            # this step is time consuming -- give it a few minutes to unzip and process ~500 job events file
            buffer = pd.read_csv(file_path, compression='gzip',
                        error_bad_lines=False)
            # add columns to concat data
            buffer.columns = ['time','missing info', 'job ID', 'event type', 'user', 'scheduling class', 'job name', 'logical job name']
            if DEBUG:
                print(buffer.head())  
            swap = [data_job, buffer]
            data_job = pd.concat(swap)  # concat the read data to buffer data
            
    # sort the job events values based on time        
    data_job = data_job.sort_values(['time'])

    # job_arrival_rate subset data
    job_arrival_rate = data_job[['time','job ID','event type']]

    if DEBUG:
        print(job_arrival_rate.head())

    # filter the data with time = 0 to remove first 10 mins data before start of trace time
    data_bool = job_arrival_rate['time'] >= 600000000       # <---- this is a fixed value as per google cluster dataset
    job_arrival_rate = job_arrival_rate[data_bool]

    job_arrival_rate.to_csv('output.csv',index=True)

def getSampledData(job_arrival_rate, time_interval=30, event_type=0, path = None):
    r""" Return job arrival count filtered data for event_type at input sample rate 
    Args:time_interval: sample rate for the data set. The dataset we are handling is currently sampled at seconds. In order to smaple it more we provide value in minutes.
    event_type: type of events we want to process"""

    # filter the data with event type = 0 'SUBMIT (0): A task or job became eligible for scheduling'
    data_bool = job_arrival_rate['event type'] == 0

    if DEBUG:
        print('data_bool')
        print(data_bool)

    job_arrival_rate = job_arrival_rate[data_bool]
    job_arrival_rate = job_arrival_rate.drop(columns='event type')  # drop the event_type column

    if DEBUG:
        print('job_arrival_rate')
        print(job_arrival_rate.head())

    # create a interval. The timestamp is represented in microseconds hence we divide it by 1e6 
    job_arrival_rate['time'] = job_arrival_rate['time']/ (1e6 * 60 * time_interval) # for every <time_interval> mins
    job_arrival_rate['time'] = job_arrival_rate['time'].apply(np.floor)

    if DEBUG:
        print('print head for job_arrival_rate')
        print(job_arrival_rate.head())
        
    job_arrival_rate = job_arrival_rate.groupby('time').count() #nunique() - to be used if we want unique jobs at given timestamp

    if DEBUG:
        print(job_arrival_rate.columns)
    # drop extra column 'Unnamed: 0'
    job_arrival_rate = job_arrival_rate.drop(columns='Unnamed: 0')

    # change the column name to job_count 
    job_arrival_rate.columns = ['job_count']

    # capture moving average of the data 
    # moving average over the period of one week
    job_arrival_rate['1-week'] = job_arrival_rate['job_count'].rolling(int(24 * 7 * 60 / time_interval), min_periods=1).mean()

    # moving average over the period of 24 hours
    job_arrival_rate['daily'] = job_arrival_rate['job_count']. rolling(int(24 * 60 / time_interval), min_periods=1).mean()

    if DEBUG:
        print('after we sample data for given time interval ')
        print(job_arrival_rate.head())

    # change the data to float as first step of preprocessing
    job_arrival_count = job_arrival_rate['job_count'].values.astype(float)

    return job_arrival_count

def generateGoogleClusterJOBEVENTTBLCSV(path='../job_events/'):

    r"""Returns job arrival count CSV from job events table from the google cluster 2011 dataset"""
    columns = ['time','missing info', 'job ID', 'event type', 'user', 'scheduling class', 'job name', 'logical job name']
    data_job = pd.DataFrame(columns=columns) # empty pandas dataframe with set columns 
    
    file_names = []
    for file in os.listdir(path): 
        file_names.append(file)

    if '.DS_Store' in file_names:
        file_names.remove('.DS_Store')
    file_names.sort()

    for file in file_names:
        file_path = path + file
        with gzip.open(file_path) as f:
            # create a buffer pandas dataframe 
            # this step is time consuming -- give it a few minutes to unzip and process ~500 job events file
            buffer = pd.read_csv(file_path, compression='gzip',
                        error_bad_lines=False)
            # add columns to concat data
            buffer.columns = ['time','missing info', 'job ID', 'event type', 'user', 'scheduling class', 'job name', 'logical job name']
            if DEBUG:
                print(buffer.head())  
            swap = [data_job, buffer]
            data_job = pd.concat(swap)  # concat the read data to buffer data
            
    # sort the job events values based on time        
    data_job = data_job.sort_values(['time'])

    # job_arrival_rate subset data
    job_arrival_rate = data_job[['time','job ID','event type','scheduling class']]

    if DEBUG:
        print(job_arrival_rate.head())

    # filter the data with time = 0 to remove first 10 mins data before start of trace time
    data_bool = job_arrival_rate['time'] >= 600000000       # <---- this is a fixed value as per google cluster dataset
    job_arrival_rate = job_arrival_rate[data_bool]

    job_arrival_rate.to_csv('output.csv',index=True)

def getSampledDataFiltered(data_job, time_interval, event_type, scheduling_class):
    """
    This method is similar to getSampledData. In addition to consider event_type, this method also considers scheduling_class as 
    filter criteria. 
    """
    # job_arrival_rate subset data
    job_arrival_rate = data_job[['time','job ID','event type','scheduling class']]

    # filter the data with time = 0 to remove first 10 mins data before start of trace time
    data_bool = job_arrival_rate['time'] >= 600000000
    job_arrival_rate = job_arrival_rate[data_bool]

    job_arrival_rate['time'] = job_arrival_rate['time'] / 1e6
    job_arrival_rate['time'] = job_arrival_rate['time'].apply(np.floor)
    job_arrival_rate['time'] = job_arrival_rate['time'] - 600

    # # filter the data with event type = 0 'SUBMIT (0): A task or job became eligible for scheduling'
    data_bool = job_arrival_rate['event type'] == event_type

    job_arrival_rate = job_arrival_rate[data_bool]
    job_arrival_rate = job_arrival_rate.drop(columns='event type')

    # filter the data with event type = 0 'SUBMIT (0): A task or job became eligible for scheduling'
    data_bool = job_arrival_rate['scheduling class'] == scheduling_class

    job_arrival_rate = job_arrival_rate[data_bool]
    job_arrival_rate = job_arrival_rate.drop(columns='scheduling class')  # drop the event_type column

    # create a interval. The timestamp is represented in microseconds hence we divide it by 1e6 
    job_arrival_rate['time'] = job_arrival_rate['time']/ (60 * time_interval) # for every <time_interval> mins
    job_arrival_rate['time'] = job_arrival_rate['time'].apply(np.floor)

    job_arrival_rate = job_arrival_rate.groupby('time').count()
    job_arrival_rate.columns = ['job_count']

    # moving average over the period of one week
    job_arrival_rate['1-week'] = job_arrival_rate['job_count'].rolling(int(24 * 7 * 60 / time_interval), min_periods=1).mean()

    # moving average over the period of 24 hours
    job_arrival_rate['daily'] = job_arrival_rate['job_count']. rolling(int(24 * 60 / time_interval), min_periods=1).mean()

    # moving average over the period of 12 hours
    job_arrival_rate['day_night'] = job_arrival_rate['job_count']. rolling(int(12 * 60 / time_interval), min_periods=1).mean()

    # change the data to float as first step of preprocessing
    job_arrival_count = job_arrival_rate['job_count'].values.astype(float)

    return job_arrival_count

def gen_covariates(times, dims):
    covariates = np.zeros((times.shape[0], dims))
    for i, input_time in enumerate(times):
        input_time = datetime.strptime(input_time, '%Y-%m-%d %H:%M:%S')
        covariates[i, 0] = input_time.weekday() #6 weekly value 
        if dims > 1:
            covariates[i, 1] = input_time.hour #24  hourly info
    
    for i in range(1,dims):
        covariates[:,i] = stats.zscore(covariates[:,i])
    return covariates

def get_train_test_data_cov_dt(filename = 'data/data-jc-30-mins.csv.csv'):

    num_covariates = 1 # week 
    job_event_csv_data = pd.read_csv(filename, sep=",",parse_dates=True)
    # print(job_event_csv_data.columns)
    # Index([u'DATETIME', u'JOBCOUNT'], dtype='object')
    job_arrival_count = job_event_csv_data['JOBCOUNT'].values.astype(float)

    # # min-max scaling 
    # scaler, job_arrival_count_transformed = minmax(job_arrival_count)
    scaler = MinMaxScaler(feature_range=(-1,1))
    job_arrival_count_transformed= scaler.fit_transform(job_arrival_count.reshape(-1,1))

    # # create test/validation/test 
    split_size = 20 # to create 80-20 split for the training/testing dataset
    test_data_size = int(math.ceil(len(job_arrival_count_transformed) * split_size / 100))

    _train_data = job_arrival_count_transformed[:-test_data_size]
    _test_data = job_arrival_count_transformed[-test_data_size:]

    covariates = gen_covariates(job_event_csv_data['DATETIME'], num_covariates)
    
    _train_cov = covariates[:-test_data_size]
    _test_cov = covariates[-test_data_size:]

    # concaternating covariates data
    _train_data = np.concatenate((_train_data,_train_cov), axis = 1)
    _test_data = np.concatenate((_test_data,_test_cov), axis = 1)

    return _train_data, _test_data, scaler