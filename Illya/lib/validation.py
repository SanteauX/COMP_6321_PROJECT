import numpy as np
import pandas as pd

from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError


def evaluate_cv(forecaster, train_data, window_size, 
                scoring = (MeanAbsolutePercentageError(), MeanSquaredError(square_root=True)),
                return_data = True):
    # Define the forecast horizon
    fh = np.arange(1, window_size+1)
    
    # Create a splitter for computing split thresholds
    splitter = ExpandingWindowSplitter(initial_window = window_size, 
                                        step_length = window_size, 
                                        fh = window_size)
    
    # Create a DataFrame for storing the results
    results = pd.DataFrame()
    #score_name = scoring.name
    
    # Train and evaluate the model for each split, store the results
    for (train_cutoff, validation_cutoff) in splitter.split(train_data):
        # Extract train and test cutoff sample numbers
        train_cutoff = train_cutoff[-1] + 1
        validation_cutoff = validation_cutoff[0] + 1

        # Partition the data into training and validation sets
        y_train = train_data.iloc[0:train_cutoff]
        y_val   = train_data.iloc[train_cutoff:validation_cutoff]

        # Train the model
        forecaster.fit(y_train)

        # Make predictions and evaluate
        y_pred = forecaster.predict(fh)   
        
        scores = {}
        if not isinstance(scoring, tuple):
            scoring = (scoring, )
        for scorer in scoring:
            score_name = scorer.name
            score = scorer(y_val, y_pred, y_train=y_train)
            scores[score_name] = score
        
        data = {}
        if return_data:
            data = {
                    "y_train": y_train,
                    "y_val": y_val,
                    "y_pred": y_pred,
                    }
        
        
        # Add results
        results = results.append({**scores, **data},
            ignore_index=True,
        )
              
    return results
                       
def plot_cv_results(results, train_data):
    ax = train_data.resample('D').sum().plot(figsize=(16, 8))
    
    for split in results.itertuples():
        sub_plot = split.y_pred.resample('D').sum().plot(ax=ax, color='orange', legend=False)

    # Set legend
    ax.legend(["True", "Predictions"])


def eval_forecast(forecaster, fh, y_true, scoring = (MeanAbsolutePercentageError(), MeanSquaredError(square_root=True))):
    # Make predictions 
    y_pred = forecaster.predict(fh)

    # Downsample data to daily and plot
    ax = y_true.plot(figsize=(16, 8))
    y_pred.plot(ax=ax)

    # Compute RMSE
    if not isinstance(scoring, tuple):
        scoring = (scoring, )
    for scorer in scoring:
        print(scorer.name + ':', scorer(y_true, y_pred))