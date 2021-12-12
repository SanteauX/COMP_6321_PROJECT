import pandas as pd

# Run historical forecasts on the input series and return predictions
def cv_predict(model, input_series, covariates):
    return model.historical_forecasts(input_series,
                                      future_covariates=covariates,
                                      forecast_horizon=365,
                                      start= input_series.start_time()+pd.DateOffset(years=1),
                                      retrain=False,
                                      verbose=True)