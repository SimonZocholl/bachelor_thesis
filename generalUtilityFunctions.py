"""
This file contains all usefull utility functions, that are used in the Notebooks/Scripts for my Master Thesis.
These Functions are excluded from the Notebooks and Scripts so this module has to be imported at the beginning of each Notebook/Script

@author: Leon Fiedler

"""
# necessary libraries
import os
#import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#import math
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
import seaborn as sns

# uncomment next line if you are running a python skript and you want to show figures
# plt.matplotlib.use("TKAgg", force=True)
#plt.style.use('default')
#plt.rcParams.update(plt.rcParamsDefault)
plt.style.use("seaborn")
size = 16
params = {'font.weight': 'normal',
        'legend.fontsize': 'large',
        'legend.fancybox': True,
        'axes.labelsize': size,
        'axes.titlesize': size,
        'xtick.labelsize': size*0.75,
        'ytick.labelsize': size*0.75,
        'axes.titlepad': 25}
 
# ----------- Function to Load CSV-files-----------------------#
def load_csv(file, header, index_col, header_name, index_name):
    parameter = os.path.basename(file).split(".")[0]
    table = pd.read_csv(
        file, sep=";", header=header, index_col=index_col, parse_dates=True
    )
    # table.to_datetime()
    table.name = parameter
    table.rename_axis(columns=header_name, index=index_name, inplace=True)
    return table


# ----------Function to look for any Value in Station_INFOS Df ----------#
def search_Station_INFOS(MetaData_to_search, val_to_search):
    Station_INFOS = pd.read_pickle(
        "C:/Users/leonf/Documents/Python/Masterthesis-Leon/DATA/Created_tables/Station_INFOS.pkl"
    )
    temp = Station_INFOS.dropna(axis=0)
    if MetaData_to_search == "Zip_IDs":
        for par, df in temp.items():
            for k, n in df["Zip_IDs"]:
                if n == val_to_search:
                    print(par, ":", k)
                    return k
                elif k == val_to_search:
                    print(par, ":", n)
                    return n
    else:
        for row, col in temp.items():
            if val_to_search in temp.loc[MetaData_to_search][row]:
                print("Match in:", row)
            else:
                print("NO Match in:", row)

# _______________________________________________Functions for creating Datasets__________________________________________________________________#

# Function to split dataframe into training and test datasets either by given length or time interval
# datetime_interval(tuple): (start_date_train(str), end_date_train(str), end_date_val(str)) --> all dates are inclusive
# periods(tuple of integers): (length of training_dataset, length of validation_datset) --> index numbers, where to split (start index is 0 by default)
# test interval = length(dataframe) - length(training period) --> no explicit definition of the length
def train_val_test_split(
    dataframe, period=None, datetime_interval=None, validation=False
):
    if period:
        if validation:
            start_index_train = 0
            end_index_train = period[0]
            start_index_val = period[0]
            end_index_val = start_index_val + period[1]
            training_split = dataframe[start_index_train:end_index_train]
            validation_split = dataframe[start_index_val:end_index_val]
            testing_split = dataframe[end_index_val:]
            return training_split, validation_split, testing_split

        else:
            start_index = 0
            end_index = period[0]
            training_split = dataframe[start_index:end_index]
            testing_split = dataframe[end_index:]
            return training_split, testing_split

    if datetime_interval:
        if validation:
            start_index_train = dataframe.index.get_loc(datetime_interval[0])
            end_index_train = dataframe.index.get_loc(datetime_interval[1]) + 1
            start_index_val = end_index_train + 1
            end_index_val = dataframe.index.get_loc(datetime_interval[2]) + 1
            training_split = dataframe[start_index_train:end_index_train]
            validation_split = dataframe[start_index_val:end_index_val]
            testing_split = dataframe[end_index_val:]
            return training_split, validation_split, testing_split

        else:
            start_index = dataframe.index.get_loc(datetime_interval[0])
            end_index = dataframe.index.get_loc(datetime_interval[1]) + 1
            training_split = dataframe[start_index:end_index]
            testing_split = dataframe[end_index:]
            return training_split, testing_split


# Function, that creates a dataset of tensors, with specific number of shuffled batches
# Input --> use results of the function "windowed_data"
# windowed_data_X: dataset(np.array), that contains the input features
# data_Y: dataset(np.array), that contains the target values/ labels


def batched_dataset(
    windowed_data_X,
    data_Y,
    batch_size,
    shuffle_buffer,
    type_of_data,
    repeat=False,
):

    if repeat == True:
        if type_of_data == "training":
            train_data = tf.data.Dataset.from_tensor_slices(
                (windowed_data_X, data_Y)
            )
            train_data = (
                train_data.cache()
                .shuffle(shuffle_buffer, seed=7)
                .batch(batch_size, drop_remainder=True)
                .repeat()
            )
            return train_data

        if type_of_data == "validation" or "testing":
            val_data = tf.data.Dataset.from_tensor_slices(
                (windowed_data_X, data_Y)
            )
            val_data = val_data.batch(batch_size, drop_remainder=True).repeat()
            return val_data

    else:
        if type_of_data == "training":
            train_data = tf.data.Dataset.from_tensor_slices(
                (windowed_data_X, data_Y)
            )
            train_data = (
                train_data.cache()
                .shuffle(shuffle_buffer, seed=7)
                .batch(batch_size, drop_remainder=True)
            )
            return train_data

        if type_of_data == "validation" or "testing":
            val_data = tf.data.Dataset.from_tensor_slices(
                (windowed_data_X, data_Y)
            )
            val_data = val_data.batch(batch_size, drop_remainder=True)
            return val_data




def windowed_data(
    df_features,
    df_label,
    window_size,
    target_size,
    start_idx=0,
    end_idx=None,
    step=1,
    single_step=True,
    ):
    """
    Function to create two windowed Datasets for predicting several timesteps into the future (single shot) --> returns two arrays(one for the input features, one for the target values)
    This function does NOT take into account feature observations as "forecasted values" --> target values are shifted into the future

    Parameters:
    df_features: columns of input features in input-df OR dataframe, that consists of the input features
    df_labels: column name of target value/ label in input-df OR dataframe, that consists of the target values/ label (i.e. discharge at specific station)
    start_index(integer): index location at which the dataset should start --> usefull for splitting the data into training-validation-testing
    end_index(integer): index location at which the dataset should end --> usefull for splitting the data into training-validation-testing
    window_size(integer): number of timesteps, that should be considered as a "look back window" of past input features used for prediction
    target_size(integer): how far(number of timesteps) to predcit into the future --> targets are selected laying ahead of the lookback window
    step(integer): The period, in timesteps, at which data is sampled (naive assumption: the data within these timesteps does not change --> take the value every step)
    """
    data = []
    labels = []
    if type(df_features) == pd.core.frame.DataFrame:
        input_features = df_features.values  # convert dataframe to np.array
        target_values = df_label.values  # convert dataframe to np.array
    else:
        input_features = df_features
        target_values = df_label

    start_index = start_idx + window_size
    end_index = end_idx

    if end_index is None:
        end_index = len(input_features) - target_size + 1

    for i in range(start_index, end_index):
        indices = range(i - window_size, i, step)
        data.append(input_features[indices])

        if single_step:
            labels.append(target_values[i + target_size - 1])
        else:
            labels.append(target_values[i : i + target_size])

    return np.array(data), np.array(labels)


def windowed_data_with_forecast(
    df_features,
    df_label,
    window_size,
    target_size,
    step=1,
    single_step=True,
    ):
    """
    Function to create two windowed Datasets for predicting several timesteps into the future (single shot) --> returns two arrays(one for the input features, one for the target values)
    This function DOES take into account feature observations as "forecasted values" --> target values are NOT shifted into the future

    Parameters:
    df_features: columns of input features in input-df OR dataframe, that consists of the input features
    df_labels: column name of target value/ label in input-df OR dataframe, that consists of the target values/ label (i.e. discharge at specific station)
    window_size(integer): number of timesteps, that should be considered as a "look back window" of past input features used for prediction
    target_size(integer): how far(number of timesteps) to predcit into the future --> targets are selected laying within the lookback window
    step(integer): The period, in timesteps, at which data is sampled (naive assumption: the data within these timesteps does not change --> take the value every step)
    """
    data = []
    labels = []
    if type(df_features) == pd.core.frame.DataFrame:
        input_features = df_features.values  # convert dataframe to np.array
        target_values = df_label.values  # convert dataframe to np.array
    else:
        input_features = df_features
        target_values = df_label


    # Create Windows
    for i in range(window_size, input_features.shape[0] + 1):
        indices = range(i - window_size, i, step)
        data.append(input_features[indices])

    # Create Labels/target values
        if single_step:
            labels.append(target_values[i - target_size])
        else:
            labels.append(target_values[i - target_size : i])

    return np.array(data), np.array(labels)


# def windowed_data_sequence_model(
#     df_features,
#     df_label,
#     window_size,
#     target_shift,
#     start_idx=0,
#     end_idx=None,
#     step=1,
#     ):
#     """
#     Function to create two windowed Datasets for sequence models (return_sequence = True in last layer)) 
#     --> returns two arrays: one for the input features (input window with window_size), one for the target values(target window with window_size)
   
#    Parameters:
#     df_features: columns of input features in input-df OR dataframe, that consists of the input features
#     df_labels: column name of target value/ label in input-df OR dataframe, that consists of the target values/ label (i.e. discharge at specific station)
#     start_index(integer): index location at which the dataset should start --> usefull for splitting the data into training-validation-testing
#     end_index(integer): index location at which the dataset should end --> usefull for splitting the data into training-validation-testing
#     window_size(integer): number of timesteps, that should be considered as a "look back window" of past input features used for prediction
#     target_shift(integer): how far(number of timesteps) the target labels are shifted compared to the input window -> target window is selected laying ahead of the lookback window by shifted steps
#     step(integer): The period, in timesteps, at which data is sampled (naive assumption: the data within these timesteps does not change --> take the value every step)
#     """
#     data = []
#     labels = []
#     if type(df_features) == pd.core.frame.DataFrame:
#         input_features = df_features.values  # convert dataframe to np.array
#         target_values = df_label.values  # convert dataframe to np.array
#     else:
#         input_features = df_features
#         target_values = df_label

#     start_index = start_idx + window_size
#     end_index = end_idx

#     if end_index is None:
#         end_index = len(input_features) - target_shift + 1
    
#     for i in range(start_index, end_index):
#         if target_shift == 0:
#             indices_data = range(i - window_size, i, step)
#             data.append(input_features[indices_data])
#             labels.append(target_values[indices_data])
#         else:
#             indices_data = range(i - window_size, i, step)
#             data.append(input_features[indices_data])

#             indices_label = range(i - window_size + target_shift, i + target_shift, step)
#             labels.append(target_values[indices_label])

#     return np.array(data), np.array(labels)

# _______________________________________________Functions for Plotting__________________________________________________________________#

# ------- Function for plotting Data out of Pandas Dataframes (does not work for Discharge_unstack/Discharge_multi)-----------#

# time_range and stations must be passed as a list
def plot_series(
    df,
    time_range,
    parameter,
    stations,
    linestyle="-",
    formater=None,
    tight=False,
):

    # extract start and end Time of Time range, i.e. inteval of interest
    start = time_range[0]
    end = time_range[-1]

    # Create Figure
    fig, ax = plt.subplots(figsize=(25, 12))

    # Set parameter value and plot axes
    if parameter == None:
        parameter = df.name
        table = pd.DataFrame(df.loc[start:end, stations])
        for idx, station in enumerate(stations):
            ax.plot(table.index, stations[idx], data=table, linestyle=linestyle)

    else:
        parameter = parameter
        table = pd.DataFrame(df[parameter].loc[start:end, stations])
        for idx, station in enumerate(stations):
            ax.plot(table.index, stations[idx], data=table, linestyle=linestyle)

    # Set Lables
    ax.set_xlabel("Date", fontsize=15, fontstyle="italic")
    # TODO: add unit --> maybe use station_INFOS somehow
    ax.set_ylabel("Unit", fontsize=15, fontstyle="italic")
    ax.set_title(parameter, fontweight="bold", fontsize=15)

    # Set Legend
    ax.legend(prop={"size": 15})

    # optional settings
    if tight == True:
        fig.tight_layout()

    if formater == "hours":
        date_format = mdates.DateFormatter("%Y-%m-%d %H:%M")
        ax.xaxis.set_major_formatter(date_format)
        hours = mdates.HourLocator()
        ax.xaxis.set_major_locator(hours)
    elif formater == "days":
        date_format = mdates.DateFormatter("%Y-%m-%d")
        ax.xaxis.set_major_formatter(date_format)
        days = mdates.DayLocator()
        ax.xaxis.set_major_locator(days)
    elif formater == "months":
        date_format = mdates.DateFormatter("%Y-%m")
        ax.xaxis.set_major_formatter(date_format)
        months = mdates.MonthLocator()
        ax.xaxis.set_major_locator(months)
    elif formater == "years":
        date_format = mdates.DateFormatter("%Y")
        ax.xaxis.set_major_formatter(date_format)
        years = mdates.YearLocator()
        ax.xaxis.set_major_locator(years)
    else:
        date_format = mdates.DateFormatter("%Y-%m-%d")
        ax.xaxis.set_major_formatter(date_format)

    # additional settings
    # ax.set_xlim(start,end)
    # xticks = ax.xaxis.get_major_ticks()  # get xticks
    # xticks[-1].label1.set_visible(False)  # modify visibility of xticks at specific positions
    ax.tick_params(
        axis="x",
        which="major",
        colors="r",
        labelsize=15.0,
        grid_color="black",
        grid_alpha=0.5,
    )
    fig.autofmt_xdate()
    plt.grid(True)
    plt.show()


# Function to Plot the Measured and Predicted Runoff in one plot

# data_measured: must be a dataframe (Training/Validation/Test Dataframe)
# data_predicted: must be a np.array of predicted and inversed target data
# data_simulated: must be a dataframe (Larsim Simulations)
def compare_discharge_plot(data_measured, data_predicted, data_simulated, title, WINDOW_SIZE):
    # 'M': Measured
    # 'P': Predicted
    # 'S': Simulated

    # time dimension
    x_M = data_measured.index
    x_P = data_measured[WINDOW_SIZE:].index
    x_S = data_simulated.index
    interval = [x_M[0], x_P[-1]]

    fig = plt.figure(figsize=(25, 12))
    ax = fig.add_subplot(111)
    ax.set_xlim(interval)

    # format your data to desired format
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))

    # Plot the Data
    (Simulated,) = ax.plot_date(x_S, data_simulated, "-.", c="black", linewidth=2.0)  
    (Measured,) = ax.plot_date(x_M, data_measured.values, '-', dashes=[6,2], linewidth=3.0) 
    (Predicted,) = ax.plot_date(x_P, data_predicted, "-", c="orange", markersize=2)

    # Figure layout
    plt.title(title)
    plt.ylabel("$m^3/s$")
    plt.xlabel("Date")
    plt.legend([Measured, Predicted, Simulated], ["Measured", "Prediction", "Larsim Simulation"], loc="upper left")
    plt.grid(True)
    plt.rcParams.update(params)
    fig.autofmt_xdate()
    fig = plt.gcf()
    return fig


# Function to Plot the Measured and Predicted runoff at specifc steps within the target step window

# data_measured: must be a dataframe (Training/Validation/Test Dataframe)
# data_predicted: must be a np.array of predicted and inversed target data
# data_simulated: must be a dataframe (Larsim Simulations)
def compare_discharge_plot_multistep(data_measured, data_predicted, data_simulated, title, WINDOW_SIZE, step, target_step):
    # 'M': Measured
    # 'P': Predicted
    # 'S': Simulated

    # time dimension
    x_M = data_measured.index
    x_S = data_simulated.index   
    if step < target_step - 1:
        x_P = data_measured[WINDOW_SIZE + step: - target_step + step + 1].index
    else:
        x_P = data_measured[WINDOW_SIZE + step:].index
    
    interval = [x_M[0], x_P[-1]]

    data_predicted = data_predicted[:,step][:]

    fig = plt.figure(figsize=(25, 12))
    ax = fig.add_subplot(111)
    ax.set_xlim(interval)

    # format your data to desired format
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))

    # Plot the Data
    (Simulated,) = ax.plot_date(x_S, data_simulated, "-.", c="black", linewidth=2.0, zorder= 1)  
    (Measured,) = ax.plot_date(x_M, data_measured.values, '-', dashes=[6,2], linewidth=3.0, zorder= 2)
    Predicted  = ax.scatter(x_P, data_predicted, marker= "o", label="Prediction", edgecolors='k', s= 5, c="orange", zorder= 3) 
    # = ax.plot_date(x_P, data_predicted, "o", c="orange", markersize=5)

    # Figure layout
    plt.title(title)
    plt.ylabel("$m^3/s$")
    plt.xlabel("Date")
    plt.legend([Measured, Predicted, Simulated], ["Measured", "Prediction", "Larsim Simulation"], loc="upper left")
    plt.grid(True)
    plt.rcParams.update(params)
    fig.autofmt_xdate()
    fig = plt.gcf()
    return fig


# Function to plot Training/Validation/Test Split of Discharge Timeseries and corresponding Violing/Box plots

def plot_validation_splits(df, validation_method, number_splits ,dataset_size_list, stat_plot_method, scale_violin= 'area' ,log_transform= False ,color_palette = {'Training': 'royalblue', 'Validation': 'violet', 'Test': 'brown'}, seed = 7):
    n_splits = number_splits

    if validation_method == 'Holdout':
        training_size = dataset_size_list[0]
        validation_size = dataset_size_list[1]
        train_dataset, val_dataset, test_dataset = train_val_test_split(df, (training_size, validation_size), None, validation=True)
        
        data = [train_dataset['q']['MARI'], val_dataset['q']['MARI'], test_dataset['q']['MARI']]
        df_sets = pd.concat(data, keys= ['Training', 'Validation', 'Test'], join='outer', axis=1)
        if log_transform == True:
            df_sets = np.log1p(df_sets)
        
        fig = plt.subplots(figsize=(25, 10))
        ax1 = plt.subplot2grid((1, 6), (0, 0), colspan=4)
        ax2 = plt.subplot2grid((1, 6), (0, 4), colspan=2)

        # Discharge Plot
        ax1.plot_date(train_dataset.index, train_dataset['q']['MARI'].values, '-', c=color_palette['Training'], label='Training Period')
        ax1.plot_date(val_dataset.index, val_dataset['q']['MARI'].values, '-', c=color_palette['Validation'], label='Validation Period')
        ax1.plot_date(test_dataset.index, test_dataset['q']['MARI'].values, '-', c=color_palette['Test'], label='Test Period')

        # Violin plot/ Boxplots
        if stat_plot_method == 'violin':
            sns.violinplot(data=df_sets, ax= ax2, gridsize=1000, cut=0, scale=scale_violin, inner='box', palette=color_palette)
        if stat_plot_method == 'box':
            sns.boxplot(data= df_sets, ax= ax2, showfliers=True, linewidth=1.5, fliersize=2, whis=None, palette=color_palette)

        # format your data to desired format
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
        ax1.set_ylabel("$m^3/s$")
        ax1.set_xlabel("Date")

        # Set titles
        ax1.set_title('Discharge Timeseries at Marienthal Station')
        ax1.legend(loc="upper right")
        if stat_plot_method == 'violin':
            ax2.set_title('Violin Plots of different Periods')
        if stat_plot_method == 'box':
            ax2.set_title('Box Plots of different Periods')

        #plt.style.use("seaborn")
        plt.grid(True)
        plt.rcParams.update(params)
        fig = plt.gcf()

    if validation_method == 'Rep-Holdout':
        training_size = dataset_size_list[0] + dataset_size_list[1]
        train_dataset , test_dataset = train_test_split(df, train_size= training_size, shuffle=False) 
        
        index_window_start = int(len(train_dataset) * 0.60)
        index_window_end  = int(len(train_dataset) * 0.70)
        index_window = range(index_window_start, index_window_end)
        random.seed(seed)
        random_split_points_1 = random.choices(index_window, k=n_splits//2)

        # Use other 5 splits after 70% of the data
        index_window_start = int(len(train_dataset) * 0.70)
        index_window_end  = int(len(train_dataset) * 0.80)
        index_window = range(index_window_start, index_window_end)
        random_split_points_2 = random.choices(index_window, k=n_splits//2)
        random_split_points = random_split_points_1 + random_split_points_2
        random_split_points.sort()

        fig = plt.figure(figsize=(25, 15))
        axes1 = []
        axes2 = []
        for n in range(n_splits):
            # Create axes 1 for Time Series
            ax1 = fig.add_subplot(plt.subplot2grid((n_splits, 6), (n, 0), colspan=4))
            ax1.set_ylabel("$m^3/s$")
            axes1.append(ax1)

            # Create axes 2 for Violin Plot
            ax2 = fig.add_subplot(plt.subplot2grid((n_splits, 6), (n, 4), colspan=2))
            axes2.append(ax2)


        for i, k in enumerate(random_split_points):
            # Split the Training Dataset into Train and Validation Set
            train_dataset_temp , val_dataset_temp = train_test_split(train_dataset, train_size= k, shuffle=False)
        
            data = [train_dataset_temp['q']['MARI'], val_dataset_temp['q']['MARI'], test_dataset['q']['MARI']]
            df_sets = pd.concat(data, keys= ['Training', 'Validation', 'Test'], join='outer', axis=1)
            if log_transform == True:
                df_sets = np.log1p(df_sets)

            # Discharge Plot
            axes1[i].plot_date(train_dataset.index, train_dataset['q']['MARI'].values, '-', c=color_palette['Training'], label='Training Period')
            axes1[i].plot_date(val_dataset_temp.index, val_dataset_temp['q']['MARI'].values, '-', c=color_palette['Validation'], label='Validation Period')
            axes1[i].plot_date(test_dataset.index, test_dataset['q']['MARI'].values, '-', c=color_palette['Test'], label='Test Period')

            # Violin plot/ Boxplots
            if stat_plot_method == 'violin':
                sns.violinplot(data= df_sets, ax=axes2[i] , gridsize=1000, cut=0, scale= scale_violin, inner='box', palette=color_palette)
            if stat_plot_method == 'box':
                sns.boxplot(data= df_sets, ax = axes2[i], showfliers=True, linewidth=1.5, fliersize=2, whis=None, palette=color_palette)

            axes1[i].xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
            axes1[i].legend(loc="upper right")
            axes1[0].set_title('Discharge Timeseries at Marienthal Station')
            axes1[-1].set_xlabel("Date")

            if stat_plot_method == 'violin':
                axes2[0].set_title('Violin Plots of different Periods')
            if stat_plot_method == 'box':
                axes2[0].set_title('Box Plots of different Periods')

            #plt.style.use("seaborn")
            plt.grid(True)
            plt.rcParams.update(params)
            fig = plt.gcf()

    if validation_method == 'k-fold':
        training_size = dataset_size_list[0] + dataset_size_list[1]
        train_dataset , test_dataset = train_test_split(df, train_size= training_size, shuffle=False) 

        kf = KFold(n_splits= n_splits, shuffle=False, random_state=seed)
        #n_splits = kf.get_n_splits()

        fig = plt.figure(figsize=(25, 15))
        axes1 = []
        axes2 = []
        for n in range(n_splits):
            # Create axes 1 for Time Series
            ax1 = fig.add_subplot(plt.subplot2grid((n_splits, 6), (n, 0), colspan=4))
            ax1.set_ylabel("$m^3/s$")
            axes1.append(ax1)

            # Create axes 2 for Violin Plot
            ax2 = fig.add_subplot(plt.subplot2grid((n_splits, 6), (n, 4), colspan=2))
            axes2.append(ax2)

        for i, (train_index, val_index) in enumerate(kf.split(train_dataset)):
            
            train_dataset_temp = train_dataset.iloc[train_index]
            val_dataset_temp = train_dataset.iloc[val_index]
            
            data = [train_dataset_temp['q']['MARI'], val_dataset_temp['q']['MARI'], test_dataset['q']['MARI']]
            df_sets = pd.concat(data, keys= ['Training', 'Validation', 'Test'], join='outer', axis=1)
            if log_transform == True:
                df_sets = np.log1p(df_sets)

            # Discharge Plot
            axes1[i].plot_date(train_dataset.index, train_dataset['q']['MARI'].values, '-', c=color_palette['Training'], label='Training Period')
            axes1[i].plot_date(val_dataset_temp.index, val_dataset_temp['q']['MARI'].values, '-', c=color_palette['Validation'], label='Validation Period')
            axes1[i].plot_date(test_dataset.index, test_dataset['q']['MARI'].values, '-', c=color_palette['Test'], label='Test Period')

            # Violin plot/ Boxplots
            if stat_plot_method == 'violin':
                sns.violinplot(data= df_sets, ax= axes2[i], gridsize=1000, cut=0, scale= scale_violin, inner='box', palette=color_palette)
            if stat_plot_method == 'box':
                sns.boxplot(data= df_sets, ax = axes2[i], showfliers=True, linewidth=1.5, fliersize=2, whis=None, palette=color_palette)
            
            #ax2 = sns.boxplot(data= df_sets, showfliers=True, linewidth=1.5, fliersize=2, whis=None)

            axes1[i].xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
            axes1[i].legend(loc="upper right")
            axes1[0].set_title('Discharge Timeseries at Marienthal Station')
            axes1[-1].set_xlabel("Date")

            if stat_plot_method == 'violin':
                axes2[0].set_title('Violin Plots of different Periods')
            if stat_plot_method == 'box':
                axes2[0].set_title('Box Plots of different Periods')

            #plt.style.use("seaborn")
            plt.grid(True)
            plt.rcParams.update(params)
            fig = plt.gcf()

    if validation_method == 'time-series-split':
        training_size = dataset_size_list[0] + dataset_size_list[1]
        train_dataset , test_dataset = train_test_split(df, train_size= training_size, shuffle=False) 

        tss = TimeSeriesSplit(n_splits= n_splits, max_train_size=None)
        #n_splits = tss.get_n_splits()

        fig = plt.figure(figsize=(25, 15))
        axes1 = []
        axes2 = []
        for n in range(n_splits):
            # Create axes 1 for Time Series
            ax1 = fig.add_subplot(plt.subplot2grid((n_splits, 6), (n, 0), colspan=4))
            ax1.set_ylabel("$m^3/s$")
            axes1.append(ax1)

            # Create axes 2 for Violin Plot
            ax2 = fig.add_subplot(plt.subplot2grid((n_splits, 6), (n, 4), colspan=2))
            axes2.append(ax2)

        for i, (train_index, val_index) in enumerate(tss.split(train_dataset)):
            
            train_dataset_temp = train_dataset.iloc[train_index]
            val_dataset_temp = train_dataset.iloc[val_index]
            
            data = [train_dataset_temp['q']['MARI'], val_dataset_temp['q']['MARI'], test_dataset['q']['MARI']]
            df_sets = pd.concat(data, keys= ['Training', 'Validation', 'Test'], join='outer', axis=1)
            if log_transform == True:
                df_sets = np.log1p(df_sets)

            # Discharge Plot
            axes1[i].plot_date(train_dataset_temp.index, train_dataset_temp['q']['MARI'].values, '-', c=color_palette['Training'], label='Training Period')
            axes1[i].plot_date(val_dataset_temp.index, val_dataset_temp['q']['MARI'].values, '-', c=color_palette['Validation'], label='Validation Period')
            axes1[i].plot_date(test_dataset.index, test_dataset['q']['MARI'].values, '-', c=color_palette['Test'], label='Test Period')

            # Violin plot/ Boxplots
            if stat_plot_method == 'violin':
                sns.violinplot(data= df_sets, ax= axes2[i], gridsize=1000, cut=0, scale= scale_violin, inner='box', palette=color_palette)
            if stat_plot_method == 'box':
                sns.boxplot(data= df_sets, ax = axes2[i], showfliers=True, linewidth=1.5, fliersize=2, whis=None, palette=color_palette)
            
            #ax2 = sns.boxplot(data= df_sets, showfliers=True, linewidth=1.5, fliersize=2, whis=None)

            axes1[i].xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
            axes1[i].legend(loc="upper right")
            axes1[0].set_title('Discharge Timeseries at Marienthal Station')
            axes1[-1].set_xlabel("Date")

            if stat_plot_method == 'violin':
                axes2[0].set_title('Violin Plots of different Periods')
            if stat_plot_method == 'box':
                axes2[0].set_title('Box Plots of different Periods')

            #plt.style.use("seaborn")
            plt.grid(True)
            plt.rcParams.update(params)
            fig = plt.gcf()

    return fig

# Function to Plot the Measured and Predicted Runoff áfter different Epochs in one plot

# data_measured: must be a dataframe (Training/Validation/Test Dataframe)
# data_predicted_after epochs: must be a dataframe of predicted and inversed target data for different epochs 
#   -->(each columns represents predicted data after different epoch)
def compare_discharge_plot_epochs(data_measured, data_predicted_after_epochs, title, WINDOW_SIZE):
    # 'M': Measured
    # 'P': Predicted

    # time dimension
    x_M = data_measured.index
    x_P = data_predicted_after_epochs.index
    
    interval = [x_M[0], x_P[-1]]

    fig = plt.figure(figsize=(25, 12))
    ax = fig.add_subplot(111)
    ax.set_xlim(interval)

    # format your data to desired format
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))

    # Plot the Data
    (Measured,) = plt.plot_date(x_M, data_measured.values, "-", c="blue")
    (Predicted_0,) = plt.plot_date(x_P, data_predicted_after_epochs.iloc[:,0], "--", c="orange")
    (Predicted_1,) = plt.plot_date(x_P, data_predicted_after_epochs.iloc[:,1], "-", c="fuchsia")
    (Predicted_2,) = plt.plot_date(x_P, data_predicted_after_epochs.iloc[:,2], "-.", c="black")


    # Figure layout
    plt.title(title)
    plt.ylabel("$m^3/s$")
    plt.xlabel("Date")
    plt.legend([Measured, Predicted_0, Predicted_1, Predicted_2], 
    ["Measured", 
    "Prediction after " + data_predicted_after_epochs.columns[0], 
    "Prediction after " + data_predicted_after_epochs.columns[1],
    "Prediction after " + data_predicted_after_epochs.columns[2]],               
    loc="upper left")
    #plt.style.use("seaborn")
    plt.grid(True)
    plt.rcParams.update(params)
    fig.autofmt_xdate()
    fig = plt.gcf()
    return fig


# Df: pandas Dataframes with predictions[observations, predictions, larsim simulations] for training/validation/test period
# index: Date at which the multi-step predictions should be plotted (str-format)
# lookback: size of lookback window, that should be plotted (int)
# terget_step: how many steps the model has predicted into the future (int)
# title_add_on: what kind of dataset is used (training/validation/test) (str)

def plot_window_with_predictions(df, index, lookback, target_step, title):
    fig, ax =  plt.subplots(figsize=(25, 12))
    labels = ["Window", "True Future", "Model Prediction", "Larsim Prediction"]
    marker_size = 12

    # for index conversion
    index = get_date_from_str(index)

    # select data from dataframe
    x = df['observations'].loc[index - timedelta(hours = lookback) : index]                            # measured discharge in lookback window
    y = df['observations'].loc[index + timedelta(hours= 1) : index + timedelta(hours= target_step)]       # ground truth discharge of future target steps
    p = pd.Series(data= df['predictions'].loc[index], index= df['predictions'].loc[index + timedelta(hours= 1) : index + timedelta(hours= target_step)].index) # predicted dischargte of target steps
    L = df['Larsim'].loc[index + timedelta(hours= 1) : index + timedelta(hours= target_step)]             # larsim prediction of target steps
    
    #plotting 
    plt.plot(x, marker='.', linestyle='dashed', linewidth=2, markersize=marker_size, label="past discharge")
    plt.scatter(y.index, y, marker= "o", label="true future", edgecolors='k', s= marker_size * 10)
    plt.scatter(p.index, p, marker= "X", label="model predictions", edgecolors='k', s= marker_size * 10)
    plt.scatter(L.index, L, marker= "X", label="larsim predictions", edgecolors='k', s= marker_size* 10)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("$m^3/s$")
    plt.legend()
    plt.grid(True)

    #ax.xaxis.set_major_locator(mdates.DayLocator()) 
    ax.xaxis.set_major_locator(mdates.HourLocator(range(0, 25, 6)))
    date_format = mdates.DateFormatter("%Y-%m-%d %H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    plt.rcParams.update(params)
    return fig

# # Function to create timestamps for plotting
# def create_time_steps(length):
#     return list(range(-length, 0))


# Function for creating a plot to viszualize model prediction
# plot_data: list of data to be plottet --> [true discharge values of the window in 1 batch, True discharge of the next step/future, predicted discharge of this batch]
# delta: how many steps to look into the future (0 is the next step)
# tf_dataset: set to True, if the input is taken from a tensor dataset. Set to False, if the input as a normal numpy array.

# def single_step_prediction(plot_data, title, delta=None, tf_dataset=True):
#     labels = ["Window", "True Future", "Model Prediction", "Larsim Prediction"]
#     marker = [".-", "o", "X", "X"]
#     fig = plt.figure(figsize=(25, 12))

#     if tf_dataset:
#         x = plot_data[0][0][
#             :, -1
#         ].numpy()  # ground truth discharge values of the window
#         y = plot_data[1][
#             0
#         ].numpy()  # ground truth discharge at specific timestep
#         p = plot_data[2]  # predicted dischargte at specific timestep
#         data = [x, y, p]
#         time_steps = create_time_steps(x.shape[0])

#         if delta:
#             future = delta
#         else:
#             future = 0

#         for i, a in enumerate(data):
#             if i:
#                 plt.plot(
#                     future, data[i], marker[i], markersize=10, label=labels[i]
#                 )
#             else:
#                 plt.plot(
#                     time_steps, data[i].flatten(), marker[i], label=labels[i]
#                 )

#         plt.title(title)
#         plt.legend()
#         plt.xlim([time_steps[0], (future + 5) * 2])
#         plt.xlabel("Time-Step")
#         plt.grid(True)
#         plt.rcParams.update(params)
#         plt.show()
#         return plt

#     else:
#         x = plot_data[0][:, -1]  # ground truth discharge values of the window
#         y = plot_data[1]  # ground truth discharge at specific timestep
#         p = plot_data[2]  # predicted dischargte at specific timestep
#         L = plot_data[3]  # larsim prediction as baseline method
#         data = [x, y, p, L]
#         time_steps = create_time_steps(x.shape[0])

#         if delta:
#             future = delta
#         else:
#             future = 0

#         for i, a in enumerate(data):
#             if i:
#                 plt.scatter(
#                     future, data[i], marker=marker[i], s=120, label=labels[i], edgecolors='k'
#                 )
#             else:
#                 plt.plot(
#                     time_steps, data[i].flatten(), marker[i], label=labels[i]
#                 )

#         plt.title(title)
#         plt.legend()
#         plt.xlim([time_steps[0], (future + 5) * 2])
#         plt.xlabel("Time-Step")
#         plt.grid(True)
#         plt.rcParams.update(params)
#         #plt.style.use("seaborn")
#         plt.show()
#         return plt

# Function for plotting the history data of a model
def plot_train_history(history, title, loss_untrained, loss_trained):
    fig = plt.figure(figsize=(10, 8))
    # plt.style.use('default')
    # plt.style.use("seaborn")
    if isinstance(history, dict):
        loss_untrained = list(loss_untrained.values())
        loss = history['loss']
        val_loss = history['val_loss']
        loss.insert(0, loss_untrained[0])
        val_loss.insert(0, loss_untrained[1])
    else:     
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        loss.insert(0, loss_untrained[0])
        val_loss.insert(0, loss_untrained[1])
    
    epochs = range(0, len(loss))

    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "g", label="Validation loss") 
    plt.plot(
        [0, 25],
        [loss_untrained[2], loss_trained],
        "rX",
        ms=10,
        label="Test loss (untrained vs. trained)",
    )
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.yticks(np.arange(0.05, np.max([np.max(loss), np.max(val_loss)]), 0.05))
    plt.legend()
    plt.grid(True)
    plt.rcParams.update(params)
    return fig

def plot_train_history_normal(history, title):
    fig = plt.figure(figsize=(10, 8))
    if type(history) is dict: 
        plt.plot(range(25), history['loss'], label="Training loss")
        plt.plot(range(25), history['val_loss'], label="Validation loss")
    else: 
        plt.plot(range(25), history.history['loss'], label="Training loss")
        plt.plot(range(25), history.history['val_loss'], label="Validation loss")
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.rcParams.update(params)
    return fig


# Function to look at batch sizes
def plot_batch_sizes(ds):
    plt.figure(figsize=(10, 8))
    batch_sizes = [batch.shape[0] for batch in ds]
    plt.bar(range(len(batch_sizes)), batch_sizes)
    plt.xlabel("Batch number")
    plt.ylabel("Batch size")
    plt.show()


# Plot Learning Rate over Fitting history
def plot_lr(history):
    plt.plot(history.history["lr"], history.history["loss"])
    plt.axis([1e-4, 2e-3, 0, 1])
    plt.grid(True, which="both")
    plt.show()


# _______________________________________________Pre-defined models__________________________________________________________________#
def get_model_vishakh(num_LSTM_units, SHAPE, loss):
    # Create and Compile the LSTM Model
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.LSTM(
            units=num_LSTM_units,
            activation="tanh",
            return_sequences=True,
            use_bias=True,
            kernel_initializer= tf.keras.initializers.RandomUniform(seed=7),
            bias_initializer= tf.keras.initializers.RandomUniform(seed=7),
            input_shape=SHAPE,
            name="LSTM_1",
        )
    )
    model.add(tf.keras.layers.Dropout(0.45, name="Dropout_1"))
    model.add(tf.keras.layers.LSTM(units=num_LSTM_units, name="LSTM_2"))
    model.add(tf.keras.layers.Dense(1))

    # adapt metrics with respect to loss function
    if loss == 'NSE':
        loss = NSE_loss
        observed_metrics = ["mae", "mse", tf.keras.metrics.RootMeanSquaredError()]
    if loss == 'MAE':
        observed_metrics = ["mse", tf.keras.metrics.RootMeanSquaredError()]
    if loss == 'MSE':
        observed_metrics = ["mae", tf.keras.metrics.RootMeanSquaredError()]

    # Compile the Model
    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1)
    model.compile(
        loss=loss,
        optimizer="adagrad",
        metrics= observed_metrics,
    )

    return model


def get_model_vishakh_masked(num_LSTM_units, SHAPE, loss, fill_value):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=fill_value, input_shape=SHAPE))
    model.add(
        tf.keras.layers.LSTM(
            units=num_LSTM_units,
            activation="tanh",
            return_sequences=True,
            use_bias=True,
            kernel_initializer="RandomUniform",
            bias_initializer="RandomUniform",
            name="LSTM_1",
        )
    )
    model.add(tf.keras.layers.Dropout(0.45, name="Dropout_1"))
    model.add(
        tf.keras.layers.LSTM(
            units=num_LSTM_units, activation="tanh", name="LSTM_2"
        )
    )
    model.add(tf.keras.layers.Dense(1))

    # adapt metrics with respect to loss function
    if loss == 'NSE':
        loss = NSE_loss
        observed_metrics = ["mae", "mse", tf.keras.metrics.RootMeanSquaredError()]
    if loss == 'MAE':
        observed_metrics = ["mse", tf.keras.metrics.RootMeanSquaredError()]
    if loss == 'MSE':
        observed_metrics = ["mae", tf.keras.metrics.RootMeanSquaredError()]

    # Compile the Model
    model.compile(
        loss=loss,
        optimizer="adagrad",
        metrics= observed_metrics,
    )

    return model


def get_changed_model(lstm_layers, num_LSTM_units, SHAPE, loss, fill_value, masked):

    if lstm_layers == 1 and masked == False:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(
            units=num_LSTM_units,
            activation="tanh",
            use_bias=True,
            kernel_initializer= tf.keras.initializers.RandomUniform(seed=7),
            bias_initializer= tf.keras.initializers.RandomUniform(seed=7),
            input_shape=SHAPE,
            name="LSTM_1",
            )
        )
        model.add(tf.keras.layers.Dropout(0.45, name="Dropout_1"))
        model.add(tf.keras.layers.Dense(1))        


    if lstm_layers == 1 and masked == True:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Masking(mask_value=fill_value, input_shape=SHAPE))
        model.add(tf.keras.layers.LSTM(
            units=num_LSTM_units,
            activation="tanh",
            use_bias=True,
            kernel_initializer="RandomUniform",
            bias_initializer="RandomUniform",
            name="LSTM_1",
            )
        )
        model.add(tf.keras.layers.Dropout(0.45, name="Dropout_1"))
        model.add(tf.keras.layers.Dense(1))


    if lstm_layers == 3 and masked == False:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(
            units=num_LSTM_units,
            activation="tanh",
            return_sequences=True,
            use_bias=True,
            kernel_initializer="RandomUniform",
            bias_initializer="RandomUniform",
            name="LSTM_1",
            )
        )
        model.add(tf.keras.layers.Dropout(0.45, name="Dropout_1"))
        model.add(tf.keras.layers.LSTM(
            units=num_LSTM_units, 
            activation="tanh",
            return_sequences=True, 
            name="LSTM_2"
            )
        )
        model.add(tf.keras.layers.Dropout(0.45, name="Dropout_2"))
        model.add(tf.keras.layers.LSTM(
            units=num_LSTM_units, 
            activation="tanh", 
            name="LSTM_3"
            )
        )
        model.add(tf.keras.layers.Dense(1))


    if lstm_layers == 3 and masked == True:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Masking(mask_value=fill_value, input_shape=SHAPE))
        model.add(tf.keras.layers.LSTM(
            units=num_LSTM_units,
            activation="tanh",
            return_sequences=True,
            use_bias=True,
            kernel_initializer="RandomUniform",
            bias_initializer="RandomUniform",
            name="LSTM_1",
            )
        )
        model.add(tf.keras.layers.Dropout(0.45, name="Dropout_1"))
        model.add(tf.keras.layers.LSTM(
            units=num_LSTM_units, 
            activation="tanh",
            return_sequences=True, 
            name="LSTM_2"
            )
        )
        model.add(tf.keras.layers.Dropout(0.45, name="Dropout_2"))
        model.add(tf.keras.layers.LSTM(
            units=num_LSTM_units, 
            activation="tanh", 
            name="LSTM_3"
            )
        )
        model.add(tf.keras.layers.Dense(1))

    # adapt metrics with respect to loss function
    if loss == 'NSE':
        loss = NSE_loss
        observed_metrics = ["mae", "mse", tf.keras.metrics.RootMeanSquaredError()]
    if loss == 'MAE':
        observed_metrics = ["mse", tf.keras.metrics.RootMeanSquaredError()]
    if loss == 'MSE':
        observed_metrics = ["mae", tf.keras.metrics.RootMeanSquaredError()]

    # Compile the Model
    model.compile(
        loss=loss,
        optimizer="adagrad",
        metrics= observed_metrics,
    )

    return model

# _______________________________________________Other usefull functions__________________________________________________________________#

# Function to show a example batch of a Tensor Dataset
def show_batch(dataset):
    for batch, label in dataset.take(1):
        print("Batched Input Features:", batch)
        print("Corresponding Labels:", label)


# change the learning rate during fitting process
def lr_schedule(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))


def NSE_loss(y_true, y_pred):
    NSE = 1 - (tf.math.reduce_sum((y_pred - y_true) ** 2, axis=0)) / tf.math.reduce_sum((y_true - tf.math.reduce_mean(y_true, axis=0)) ** 2, axis=0)
    return 1 - NSE

def kgenp(simulation_s, evaluation):
    # calculate error in timing and dynamics r (Spearman's correlation coefficient)
    sim_rank = np.argsort(np.argsort(simulation_s, axis=0), axis=0)
    obs_rank = np.argsort(np.argsort(evaluation, axis=0), axis=0)
    r = np.sum(
        (obs_rank - np.mean(obs_rank, axis=0, dtype=np.float64))
        * (sim_rank - np.mean(sim_rank, axis=0, dtype=np.float64)),
        axis=0,
    ) / np.sqrt(
        np.sum(
            (obs_rank - np.mean(obs_rank, axis=0, dtype=np.float64)) ** 2,
            axis=0,
        )
        * (
            np.sum(
                (sim_rank - np.mean(sim_rank, axis=0, dtype=np.float64)) ** 2,
                axis=0,
            )
        )
    )
    # calculate error in timing and dynamics alpha (flow duration curve)
    sim_fdc = np.sort(
        simulation_s
        / (
            simulation_s.shape[0]
            * np.mean(simulation_s, axis=0, dtype=np.float64)
        ),
        axis=0,
    )
    obs_fdc = np.sort(
        evaluation
        / (evaluation.shape[0] * np.mean(evaluation, axis=0, dtype=np.float64)),
        axis=0,
    )
    alpha = 1 - 0.5 * np.sum(np.abs(sim_fdc - obs_fdc), axis=0)
    # calculate error in volume beta (bias of mean discharge)
    beta = np.mean(simulation_s, axis=0) / np.mean(
        evaluation, axis=0, dtype=np.float64
    )
    # calculate the non-parametric Kling-Gupta Efficiency KGEnp
    kgenp_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return np.vstack((kgenp_, r, alpha, beta))


# Function to compute different performance metrices
# Watch out: both input datasets should have the same dimensions! (i.e. flatten the datasets first)
def performance_metrics(
    ground_truth_data, predicted_data, category, precision=4, mulitoutput=False
):
    if mulitoutput:
        y_true = ground_truth_data#.flatten()
        y_pred = predicted_data#.flatten()

        # Max Error
        MAX_err = np.amax(np.abs(y_true - y_pred), axis=0)
        #MAX_err = metrics.max_error(y_true, y_pred)

        # MAE
        MAE = metrics.mean_absolute_error(
            y_true, y_pred, multioutput="raw_values"
        )
        # MSE
        MSE = metrics.mean_squared_error(
            y_true, y_pred, multioutput="raw_values", squared=True
        )
        # RMSE
        RMSE = metrics.mean_squared_error(
            y_true, y_pred, multioutput="raw_values", squared=False
        )
        # Median Absolute Error
        MeAE = metrics.median_absolute_error(
            y_true, y_pred, multioutput="raw_values"
        )
        # Relativ Error in Volume
        REV = ((np.sum((y_pred), axis=0) - np.sum((y_true), axis=0)) / np.sum((y_true), axis=0)) * 100
        
        # NSE
        NSE = 1 - (np.sum((y_pred - y_true) ** 2, axis=0)) / np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)

        # Kling Gupta Efficiency
        KGNP = kgenp(y_pred, y_true)

        metrics_dict = {
            "MAX_ERROR": np.round(MAX_err, precision),
            "MAE": np.round(MAE, precision),
            "MSE": np.round(MSE, precision),
            "RMSE": np.round(RMSE, precision),
            "MeAE": np.round(MeAE, precision),
            "REV": np.round(REV, precision),
            "NSE": np.round(NSE, precision),
            "KGNP": np.round(KGNP[0], precision),
        }

        # if single argument is passed:
        if type(category) == str:
            if category == "MAX_Error":
                return metrics_dict["MAX_ERROR"]
            elif category == "MAE":
                return metrics_dict["MAE"]
            elif category == "MSE":
                return metrics_dict["MSE"]
            elif category == "RMSE":
                return metrics_dict["RMSE"]
            elif category == "MeAE":
                return metrics_dict["MeAE"]
            elif category == "REV":
                return metrics_dict["REV"]
            elif category == "NSE":
                return metrics_dict["NSE"]
            elif category == "KGNP":
                return metrics_dict["KGNP"]
            elif category == "All":
                return metrics_dict

        metrics_out = []
        # if list of criteria is passed:
        if type(category) == list:
            for i in range(len(category)):
                if category[i] == "MAX_Error":
                    metrics_out.append(np.round(MAX_err, precision))
                if category[i] == "MAE":
                    metrics_out.append(np.round(MAE, precision))
                elif category[i] == "MSE":
                    metrics_out.append(np.round(MSE, precision))
                elif category[i] == "RMSE":
                    metrics_out.append(np.round(RMSE, precision))
                elif category[i] == "MeAE":
                    metrics_out.append(np.round(MeAE, precision))
                elif category[i] == "REV":
                   metrics_out.append(np.round(REV, precision))
                elif category[i] == "NSE":
                    metrics_out.append(np.round(NSE, precision))
                elif category[i] == "KGNP":
                    metrics_out.append(np.round(KGNP[0][0], precision))
            return metrics_out
    else:
        y_true = ground_truth_data.flatten()
        y_pred = predicted_data.flatten()

        # Max Error
        MAX_err = metrics.max_error(y_true, y_pred)
        # MAE
        MAE = metrics.mean_absolute_error(
            y_true, y_pred, multioutput="uniform_average"
        )
        # MSE
        MSE = metrics.mean_squared_error(
            y_true, y_pred, multioutput="uniform_average", squared=True
        )
        # RMSE
        RMSE = metrics.mean_squared_error(
            y_true, y_pred, multioutput="uniform_average", squared=False
        )
        # Median Absolute Error
        MeAE = metrics.median_absolute_error(
            y_true, y_pred, multioutput="uniform_average"
        )
        # Relativ Error in Volume
        REV = ((np.sum(y_pred) - np.sum(y_true)) / np.sum(y_true)) * 100
        
        # NSE
        NSE = 1 - (np.sum((y_pred - y_true) ** 2, axis=0)) / np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)

        # Kling Gupta Efficiency
        KGNP = kgenp(y_pred, y_true)

        metrics_dict = {
            "MAX_ERROR": round(MAX_err, precision),
            "MAE": round(MAE, precision),
            "MSE": round(MSE, precision),
            "RMSE": round(RMSE, precision),
            "MeAE": round(MeAE, precision),
            "REV": round(REV, precision),
            "NSE": round(NSE, precision),
            "KGNP": round(KGNP[0][0], precision),
        }

        # if single argument is passed:
        if type(category) == str:
            if category == "MAX_Error":
                return metrics_dict["MAX_ERROR"]
            elif category == "MAE":
                return metrics_dict["MAE"]
            elif category == "MSE":
                return metrics_dict["MSE"]
            elif category == "RMSE":
                return metrics_dict["RMSE"]
            elif category == "MeAE":
                return metrics_dict["MeAE"]
            elif category == "REV":
                return metrics_dict["REV"]
            elif category == "NSE":
                return metrics_dict["NSE"]
            elif category == "KGNP":
                return metrics_dict["KGNP"]
            elif category == "All":
                return metrics_dict

        metrics_out = []
        # if list of criteria is passed:
        if type(category) == list:
            for i in range(len(category)):
                if category[i] == "MAX_Error":
                    metrics_out.append(round(MAX_err, precision))
                elif category[i] == "MAE":
                    metrics_out.append(round(MAE, precision))
                elif category[i] == "MSE":
                    metrics_out.append(round(MSE, precision))
                elif category[i] == "RMSE":
                    metrics_out.append(round(RMSE, precision))
                elif category[i] == "MeAE":
                    metrics_out.append(round(MeAE, precision))
                elif category[i] == "REV":
                    metrics_out.append(round(REV, precision))
                elif category[i] == "NSE":
                    metrics_out.append(round(NSE, precision))
                elif category[i] == "KGNP":
                    metrics_out.append(round(KGNP[0][0], precision))
            return metrics_out


def get_date_from_str(s_date):
    date_patterns = [
        "%d.%m.%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%Y-%m-%d %H:%M",
        "%m/%d/%y %H:%M",
        "%m/%d/%y %H:%M:%S",
    ]

    for pattern in date_patterns:
        try:
            return datetime.strptime(s_date, pattern)  # .date()
        except:
            pass

    print("Date is not in expected format: %s" % (s_date))


def str_to_timestamp(s_date):

    try:
        s_datetime = get_date_from_str(s_date)
        s_timestamp = pd.Timestamp(s_datetime)
        return s_timestamp
    except:
        pass

    print("Date is not in expected format: %s" % (s_date))