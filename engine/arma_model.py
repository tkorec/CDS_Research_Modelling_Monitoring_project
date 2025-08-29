import pandas as pd
import numpy as np
from scipy.stats import boxcox
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
import statsmodels.api as sm


class ModelARIMA():

      def __init__(self) -> None:
            pass

      def hyndman_khandakar(self, series):
            auto_model = auto_arima(series)
            forecast, conf_int = auto_model.predict(n_periods=1, return_conf_int=True, alpha=0.05)
            conf_int = conf_int[0]
            return forecast, conf_int
      
      def check_stationarity(self, data):
            result = adfuller(data)
            if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
                  return True
            else:
                  return False
        
      def invert_differencing(self, forecast, differencing_count, data):
            reverted = forecast
            for i in range(1, differencing_count + 1):
                  reverted = reverted.cumsum() + data["Close"].iloc[-i]
            return reverted

      def arima_model(self, series, parameters):
            order = tuple(parameters[0])
            seasonal_order = tuple(parameters[1])
            model = sm.tsa.statespace.SARIMAX(series, order=order, seasonal_order=seasonal_order, trend="c")
            res = model.fit(disp=False)
            forecast = res.get_prediction(steps=1)
            expected = forecast.predicted_mean.values[0]
            conf_int = forecast.conf_int(alpha=0.05).values.tolist()[0]
            return expected, conf_int

      def get_parameters(self, data):
            close_p = data["Close"]

            is_data_stationary = self.check_stationarity(close_p)
            while is_data_stationary == False:
                  close_p = close_p.diff()
                  close_p = close_p.dropna()
                  is_data_stationary = self.check_stationarity(close_p)

            parameters = self.hyndman_khandakar(close_p)
            return parameters
      
      def model(self, data, parameters):
            data = data.sort_values(by="Date", ascending=True)
            close_p = data["Close"]
            differencing_count = 0
            is_data_stationary = self.check_stationarity(close_p)
            while is_data_stationary == False:
                  close_p = close_p.diff()
                  close_p = close_p.dropna()
                  differencing_count += 1
                  is_data_stationary = self.check_stationarity(close_p)

            forecast, conf_int = self.arima_model(close_p, parameters)

            conf_int = np.array(conf_int, ndmin=2)  
            forecast = pd.Series(forecast)

            forecast = self.invert_differencing(forecast, differencing_count, data)
            conf_int[:, 0] = self.invert_differencing(conf_int[:, 0], differencing_count, data)
            conf_int[:, 1] = self.invert_differencing(conf_int[:, 1], differencing_count, data)

            return forecast, conf_int
