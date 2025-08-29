import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

class Model():

    def __init__(self) -> None:
            pass

    def invert_differencing(self, forecast, differencing_count, series):
        reverted = forecast
        for i in range(1, differencing_count + 1):
            reverted = reverted.cumsum() + series["Close"].iloc[-i]
        return reverted

    def check_stationarity(self, data):
        result = adfuller(data)
        if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
            return True
        else:
            return False
                
    def arima_model(self, series, parameters):
        order = tuple(parameters[0])
        seasonal_order = tuple(parameters[1])
        series = series.reset_index(drop=True)
        model = sm.tsa.statespace.SARIMAX(series, order=order, seasonal_order=seasonal_order, trend="c")
        res = model.fit(disp=False)
        forecast = res.get_prediction(steps=1)
        expected = forecast.predicted_mean.values[0]
        conf_int = forecast.conf_int(alpha=0.05).values.tolist()[0]
        return expected, conf_int

    def model(self, asset_data, parameters):
        closing_prices = asset_data["Close"]
        differencing_count = 0
        is_data_stationary = self.check_stationarity(closing_prices)
        while is_data_stationary == False:
            closing_prices = closing_prices.diff()
            closing_prices = closing_prices.dropna()
            differencing_count += 1
            is_data_stationary = self.check_stationarity(closing_prices)
        forecast, conf_int = self.arima_model(closing_prices, parameters)
        conf_int = np.array(conf_int, ndmin=2)  
        forecast = pd.Series(forecast)
        forecast = self.invert_differencing(forecast, differencing_count, asset_data)
        conf_int[:, 0] = self.invert_differencing(conf_int[:, 0], differencing_count, asset_data)
        conf_int[:, 1] = self.invert_differencing(conf_int[:, 1], differencing_count, asset_data)
        return conf_int