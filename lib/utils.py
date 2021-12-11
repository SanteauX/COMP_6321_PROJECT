import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.metrics
from sktime.forecasting.model_selection import ExpandingWindowSplitter


def load_station_data(filepath, remove_zero_periods=False, remove_leap_day = False, add_seasonality=False, add_weather=False):
    station_bike_demand = pd.read_csv(filepath, index_col='start_date', parse_dates=['start_date']).asfreq('D')

    # Take only data until 2019
    station_bike_demand = station_bike_demand[:'2019-12-31']

    # Add zeros for 2014 before April to have consistent period windows
    df = pd.DataFrame(index=pd.date_range(start='2014-01-01', end='2014-04-14', freq='D'), columns=['trip_count'])
    df['trip_count'] = 0
    station_bike_demand = pd.concat([df, station_bike_demand])

    if add_seasonality:

        year=365*24*60*60
        week=7*24*60*60

        timestamp = station_bike_demand.index.map(pd.Timestamp.timestamp)

        station_bike_demand['year_sin'] = np.sin(timestamp * (2 * np.pi / year))
        station_bike_demand['year_cos'] = np.cos(timestamp * (2 * np.pi / year))

        station_bike_demand['week_sin'] = np.sin(timestamp * (2 * np.pi / week))
        station_bike_demand['week_cos'] = np.cos(timestamp * (2 * np.pi / week))

    if add_weather:
        weather_df = pd.read_csv('data/weather.csv', index_col='Date/Time', parse_dates=['Date/Time'])
        station_bike_demand = pd.concat([station_bike_demand ,weather_df], axis=1)

    # In order to have consistent data period each year, take measurementes from April 15 to October 31 only
    selection = (((station_bike_demand.index.month == 4) & (station_bike_demand.index.day < 15)) | (station_bike_demand.index.month < 4))         
    station_bike_demand.loc[selection, 'trip_count'] = 0
    station_bike_demand.loc[(station_bike_demand.index.month > 10), 'trip_count'] = 0

    # Remove the leap year day
    if remove_leap_day:
      station_bike_demand = station_bike_demand[~((station_bike_demand.index.month == 2) & (station_bike_demand.index.day == 29))]
    
    #-----------------------------------------
    # Remove zero periods
    if remove_zero_periods:
      station_bike_demand = station_bike_demand[~selection]
      station_bike_demand = station_bike_demand[~(station_bike_demand.index.month > 10)]
    
    return station_bike_demand


def zero_out_outside_season(series):
    # In order to have consistent data period each year, take measurementes from April 15 to October 31 only
    selection = (((series.index.month == 4) & (series.index.day < 15)) | (series.index.month < 4))         
    series.loc[selection, 'trip_count'] = 0
    series.loc[(series.index.month > 10), 'trip_count'] = 0
    
    return series  

def evaluate(y_true, y_pred):
    # Zero out predictions outside of season where we have demand data
    y_pred = zero_out_outside_season(y_pred)
    
    # Find the intersection before true values and predictions
    y_true_eval = y_true[y_true.index.isin(y_pred.index)]
    
    # Evaluate and print metrics
    rmse = sklearn.metrics.mean_squared_error(y_true_eval, y_pred, squared=False)
    mae = sklearn.metrics.mean_absolute_error(y_true_eval, y_pred)
    
    return (rmse, mae)

def plot_and_evaluate(y_true, y_pred, title=''):  
    # Evaluate and print metrics
    rmse, mae = evaluate(y_true, y_pred)  
    print(f'Error metrics for {title}:\n')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    
    # Plot the prediction against the actual series
    ax = y_true.plot(label='actual', figsize=(16,8))
    y_pred.plot(ax=ax, label='forecast')
    
    ax.legend(["True", "Predictions"]);    
    plt.xlabel('Time')
    plt.ylabel('Trip Count')    
    plt.title(title)
    
    return (rmse, mae)

  
def cv_predict(forecaster, train_data, window_size, training_function=None, return_data = True, verbose=False):
    # Define the forecast horizon
    fh = np.arange(1, window_size+1)
    
    # Create a splitter for computing split thresholds
    splitter = ExpandingWindowSplitter(initial_window = window_size, 
                                        step_length = window_size, 
                                        fh = window_size)
    
    # Create a DataFrame for storing the results
    results = pd.DataFrame()
    
    # Train and evaluate the model for each split, store the results
    for (train_cutoff, validation_cutoff) in splitter.split(train_data):
        # Extract train and test cutoff sample numbers
        train_cutoff = train_cutoff[-1] + 1
        validation_cutoff = validation_cutoff[0] + 1

        # Partition the data into training and validation sets
        y_train = train_data.iloc[0:train_cutoff]
        y_val   = train_data.iloc[train_cutoff:validation_cutoff]

        if verbose:
            train_end = y_train.tail(1).index.strftime('%Y-%m-%d')[0]
            val_end = y_val.tail(1).index.strftime('%Y-%m-%d')[0]

            print(f"Train cutoff: {train_end}")
            print(f"Validation cutoff: {val_end}")
            print()

        if training_function == None:
            # Train the model
            forecaster.fit(y_train)

            # Make predictions and evaluate
            y_pred = forecaster.predict(fh)
        else:
            y_pred = training_function(y_train, y_val)   
        
        # Store the data for the split
        data = {}
        if return_data:
            data = {
                    "y_train": y_train,
                    "y_val": y_val,
                    "y_pred": y_pred,
                    }
        
        
        # Add split data to results
        results = results.append({**data},
            ignore_index=True,
        )
              
    return results

def merge_cv_predictions(cv_results):
    return pd.concat(cv_results['y_pred'].tolist())