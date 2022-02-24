import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy import stats
import matplotlib as mpl
from matplotlib.markers import TICKDOWN
import os
import datetime
from scipy.interpolate import UnivariateSpline
import csv
import bisect
import time
import pathlib
from plotting import savefig

def get_times(dataframe, filename):
    times = pandas.to_datetime(dataframe['Timestamp'], format = "%Y-%m-%d  %H:%M:%S")
    null_time = times.iloc[0].timestamp()
 
    for index, value in times.iteritems():
        times.iloc[index] = (value.timestamp() - null_time) /(60*60)
        print(index)
    #times = times /24 # convert to days
    times.to_csv(filename, header = False, index=False)
    return times


def get_timeseries_influx(series, start_datetime = None, filename ='RelativeTimeseries.csv', timestring ="%Y-%m-%dT%H:%M:%S.000Z"):
    ''' this function converts a timeseries from the influxdb to a series in hours, starting with the first datapoint'''
    timelist = []
    # perform conversion from string to datetime object
    for index, value in series.items():
        times = datetime.datetime.strptime(value, timestring)
        timelist.append(times)
    abs_time_series  = pandas.Series(timelist, name = "Time, absolute")

    # perform conversion from datetime object to relative time in hours
    rel_time_list = []
    if start_datetime != None:
        start = start_datetime
    else:
        start = abs_time_series[0]
    for index, value in abs_time_series.items():
        diff = value - start # this creates a datetime timedelta object implicitly
        diff_hours = diff.total_seconds() /(60*60)
        rel_time_list.append(diff_hours)
    rel_time_series = pandas.Series(rel_time_list, name ='Time, relative, h')
    rel_time_series.to_csv(filename,header = True, index=False)
    return rel_time_series, abs_time_series

def btm_overview():
    correlation_factor = 0.85
    dilution_factor = 100
    blank = 0.035
    fig1, ax = plt.subplots(nrows=3, ncols=1, sharex= True, sharey = True, figsize=(5,4))

    # first, plot 4.8 g/l*h feed rate from carboxylase 19 and 20
    def feed_rate_4_8_carb_20(ax):
        starttime = datetime.datetime (2021,9, 22, 15, 26, 0)
        full_dataframe = pandas.read_csv(fr'{data_path}\Carboxylase_20\\OD_2021-09-23-09-34 Chronograf Data.csv')
        strings = [] # inductor concentration 3.2 µM, feed rate 4.8 g/l*h
        
        for i in [10]:
            for k in 'EFGH':
                strings.append(f'{k}{str(i)}')
        #strings.remove('C12')
        print(strings)
        OD_data = full_dataframe[full_dataframe['Position'].isin(strings) == True]
        
        OD_data.reset_index(inplace = True)
        rel_time, abs_time = get_timeseries_influx (OD_data['time'], starttime)
        times = rel_time.unique()
        data = pandas.concat([OD_data, rel_time.to_frame()], axis = 1)
        print(data)
        cell_density = ((data['Readerdata.Value']-blank) * dilution_factor) * correlation_factor
        cell_density.rename('DCW, g/L', inplace = True)
        data_ = pandas.concat([data, cell_density.to_frame()], axis = 1)
        data_std = data_.groupby(['time']).std()
        data_mean = data_.groupby(['time']).mean()
        print(data_mean)
        ax[0].errorbar(times, data_mean['DCW, g/L'], yerr = data_std['DCW, g/L'], label='IPTG = 0.48 \xb5M, F$_{in}$ = 4.8 g L$^{-1}$ h$^{-1}$', linestyle ='', marker='o') # plot stuff
        return ax
    ax = feed_rate_4_8_carb_20(ax)

    def feed_rate_3_0_carb_23(ax):
        starttime = datetime.datetime (2021,10, 14, 15, 22, 0)
        full_dataframe = pandas.read_csv(fr'{data_path}\Carboxylase_23\\OD_2021-10-15-09-10 Chronograf Data.csv')
        strings = [] # 6 µM IPTG, 3 g/l*h feed
        for i in [10]:
            for k in 'ABCD':
                strings.append(f'{k}{str(i)}')
        print(strings)
        OD_data = full_dataframe[full_dataframe['Position'].isin(strings) == True]
        OD_data.reset_index(inplace = True)
        rel_time, abs_time = get_timeseries_influx (OD_data['time'], starttime)
        times = rel_time.unique()
        data = pandas.concat([OD_data, rel_time.to_frame()], axis = 1)
        cell_density = ((data['Readerdata.Value']-blank) * dilution_factor) * correlation_factor
        cell_density.rename('DCW, g/L', inplace = True)
        data_ = pandas.concat([data, cell_density.to_frame()], axis = 1)
        data_std = data_.groupby(['time']).std()
        data_mean = data_.groupby(['time']).mean()

        
        ax[1].errorbar(times, data_mean['DCW, g/L'], yerr = data_std['DCW, g/L'], label='IPTG = 6 \xb5M, F$_{in}$ = 3 g L$^{-1}$ h$^{-1}$', linestyle ='', marker='o') # plot stuff
        return ax
    ax = feed_rate_3_0_carb_23(ax)

    def feed_rate_1_0_carb_21(ax):
        starttime = datetime.datetime (2021,9, 30, 15, 24, 0)
        full_dataframe = pandas.read_csv(fr'{data_path}\Carboxylase_21\\OD_2021-10-01-10-59 Chronograf Data.csv')
        strings = [] # 12 µM IPTG, 1 g/l*h feed
        for i in [12]:
            for k in 'EFGH':
                strings.append(f'{k}{str(i)}')
        print(strings)
        OD_data = full_dataframe[full_dataframe['Position'].isin(strings) == True]
        OD_data.reset_index(inplace = True)
        rel_time, abs_time = get_timeseries_influx (OD_data['time'], starttime)
        times = rel_time.unique()
        data = pandas.concat([OD_data, rel_time.to_frame()], axis = 1)
        cell_density = ((data['Readerdata.Value']-blank) * dilution_factor) * correlation_factor
        cell_density.rename('DCW, g/L', inplace = True)
        data_ = pandas.concat([data, cell_density.to_frame()], axis = 1)
        data_std = data_.groupby(['time']).std()
        data_mean = data_.groupby(['time']).mean()
        
        ax[2].errorbar(times, data_mean['DCW, g/L'], yerr = data_std['DCW, g/L'], label='IPTG = 12 \xb5M, F$_{in}$ = 1 g L$^{-1}$ h$^{-1}$', marker ='o', linestyle='') # plot stuff
        return ax
    ax = feed_rate_1_0_carb_21(ax)


    ax[0].set_ylim(0, 50)
    ax[0].yaxis.set_ticks(np.arange(0, 50.000001, 10))
    ax[2].xaxis.set_ticks(np.arange(0, 18.001, 3))
    ax[2].set_xlim(0, 18)

    ax[0].set_ylabel('DCW, g L$^{-1}$')
    ax[1].set_ylabel('DCW, g L$^{-1}$')
    ax[2].set_ylabel('DCW, g L$^{-1}$')
    ax[2].set_xlabel('Time, h')
    #ax[0].legend()
    #ax[1].legend()
    #ax[2].legend()

    savefig(fig1, "btm_overview")
    #fig1.savefig(fr'{figure_path}\btm_overview.png', dpi = 800, bbox_inches='tight')


def ph_plot(low_feed_position, medium_feed_position, high_feed_position):
    fig1, ax = plt.subplots(nrows=3, ncols=1, sharex= True, sharey = True, figsize=(5,4))
    

    def pH_plot_data(ax, plot_number, filepath, reactor_position, starttime, label):
        pH_raw_data = pandas.read_csv(filepath)
        rel_timedata, abs_time_data = get_timeseries_influx(pH_raw_data['time'], starttime)
        full_data = pandas.concat([rel_timedata, pH_raw_data], axis = 1)
        active_position = full_data[full_data.position == reactor_position] 
        ax[plot_number].errorbar(active_position['Time, relative, h'], active_position['Presens_data.pH_value'], marker = '', linestyle = '-')
        return ax
    ax = pH_plot_data(ax=ax, plot_number=0, filepath=fr'{data_path}\Carboxylase_20\pH_2021-09-23-09-33 Chronograf Data.csv',
                      reactor_position = high_feed_position, starttime = datetime.datetime (2021,9, 22, 15, 26, 0), label='IPTG = 0.48 \xb5M, F$_{in}$ = 4.8 g L$^{-1}$ h$^{-1}$')#e2
    ax = pH_plot_data(ax=ax, plot_number=1, filepath=fr'{data_path}\Carboxylase_23\pH_2021-10-15-10-26 Chronograf Data.csv',
                      reactor_position = medium_feed_position, starttime = datetime.datetime (2021,10, 14, 15, 22, 0), label='IPTG = 6 \xb5M, F$_{in}$ = 3 g L$^{-1}$ h$^{-1}$')
    ax = pH_plot_data(ax=ax, plot_number=2, filepath=fr'{data_path}\Carboxylase_21\pH_2021-10-01-10-58 Chronograf Data.csv',
                      reactor_position = low_feed_position, starttime = datetime.datetime (2021,9, 30, 15, 24, 0), label='IPTG = 12 \xb5M, F$_{in}$ = 1 g L$^{-1}$ h$^{-1}$')

    ax[0].set_ylim(5,9)
    ax[2].set_xlim(0, 18)

    ax[0].yaxis.set_ticks(np.arange(5,10,1))
    ax[2].xaxis.set_ticks(np.arange(0, 18.001, 3))

    
    ax[0].set_ylabel("pH, -")
    ax[1].set_ylabel("pH, -")
    ax[2].set_ylabel("pH, -")
    ax[2].set_xlabel("Time, h")

    savefig(fig1, "pH_overview")
    #fig1.savefig(fr'{figure_path}\pH_overview.png', dpi = 800, bbox_inches='tight')

def o2_plot(low_feed_position, medium_feed_position, high_feed_position):
    fig1, ax = plt.subplots(nrows=3, ncols=1, sharex= True, sharey = True, figsize=(5,4))

    def plot_o2_data(ax, label, path, ax_position, reactor_number, starttime):
        o2_raw_data = pandas.read_csv(path)
        rel_timedata, abs_time_data = get_timeseries_influx(o2_raw_data['time'], starttime)
        full_data = pandas.concat([rel_timedata, o2_raw_data], axis = 1)
        active_position = full_data[full_data.position == reactor_number] 
        ax[ax_position].errorbar(active_position['Time, relative, h'], active_position['Presens_data.O2_value'], marker = '', linestyle = '-', label=label)
        return ax
    ax = plot_o2_data(ax=ax, label='IPTG = 0.48 \xb5M, F$_{in}$ = 4.8 g L$^{-1}$ h$^{-1}$', path = fr'{data_path}\Carboxylase_20\O2_2021-09-23-09-34 Chronograf Data.csv',
                      ax_position=0, reactor_number= high_feed_position, starttime= datetime.datetime (2021,9, 22, 15, 26, 0)) 
    ax = plot_o2_data(ax=ax, label='IPTG = 6 \xb5M, F$_{in}$ = 3 g L$^{-1}$ h$^{-1}$', path = fr'{data_path}\Carboxylase_23\O2_2021-10-15-10-26 Chronograf Data.csv',
                      ax_position=1, reactor_number= medium_feed_position, starttime= datetime.datetime (2021,10, 14, 15, 22, 0)) 
    ax = plot_o2_data(ax=ax, label='IPTG = 12 \xb5M, F$_{in}$ = 1 g L$^{-1}$ h$^{-1}$', path = fr'{data_path}\Carboxylase_21\O2_2021-10-01-10-59 Chronograf Data.csv',
                      ax_position=2, reactor_number= low_feed_position, starttime= datetime.datetime (2021,9, 30, 15, 24, 0))  
        
    ax[0].set_ylim(0,100)
    ax[2].set_xlim(0, 18)
    ax[0].set_ylabel("O$_{2}$, %")
    ax[1].set_ylabel("O$_{2}$, %")
    ax[2].set_ylabel("O$_{2}$, %")

    ax[2].set_xlabel("Time, h")
    ax[2].xaxis.set_ticks(np.arange(0, 18.001, 3))
    ax[0].yaxis.set_ticks(np.arange(0, 100.001, 20))

    savefig(fig1, "O2_overview")
    
if __name__ == "__main__":
    low_feed_position =15 # f2
    medium_feed_position = 25 #b4
    high_feed_position = 12 # e2

    # create the necessary paths
    parent_path= pathlib.Path(__file__).absolute().parent.parent
    data_path = fr'{parent_path}\data'
    figure_path = fr'{parent_path}\results'
    
    btm_overview()
    ph_plot(low_feed_position = low_feed_position, medium_feed_position = medium_feed_position, high_feed_position = high_feed_position)
    o2_plot(low_feed_position = low_feed_position, medium_feed_position = medium_feed_position, high_feed_position = high_feed_position)
