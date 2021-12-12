import pandas as pd

def zero_out(timestamp, value):
    before_april = (timestamp.month == 4 and timestamp.day < 15) or (timestamp.month < 4)
    after_october = (timestamp.month > 10)
    if before_april or after_october:
        value = 0.0
    return value    

def perform_cv(model, input_series, covariates):
    return model.historical_forecasts(input_series,
                                      future_covariates=covariates,
                                      forecast_horizon=365,
                                      start= input_series.start_time()+pd.DateOffset(years=1),
                                      retrain=False,
                                      verbose=True)