import pandas as pd
from pandas import DataFrame
import numpy as np
import os
from alpha_vantage.timeseries import TimeSeries
import requests
import yfinance as yf
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import scipy.stats as st


class LoadAssetsData():


    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_folder_path = os.path.join(self.script_dir, "..", "data")
        self.ALPHA_VANTAGE_API_KEY = "6JKWIK2RFT5121M1"
        self.ts = TimeSeries(key=self.ALPHA_VANTAGE_API_KEY, output_format="pandas")
        self.tickers, self.files = self.get_assets_tickers()
        self.ten_years_ago = pd.Timestamp.today() - pd.DateOffset(years=10)
        self.risk_free_rates = self.get_risk_free_rates()


    def hyndman_khandakar(self, series):
        auto_model = auto_arima(series)
        #forecast, conf_int = auto_model.predict(n_periods=1, return_conf_int=True, alpha=0.05)
        #conf_int = conf_int[0]
        order = list(auto_model.order)
        seasonal_order = list(auto_model.seasonal_order)
        parameters = []
        parameters.append(order)
        parameters.append(seasonal_order)
        return parameters


    def check_stationarity(self, data):
        result = adfuller(data)
        if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
            return True
        else:
            return False


    def get_parameters(self, data):
        close_p = data["Close"]

        is_data_stationary = self.check_stationarity(close_p)
        while is_data_stationary == False:
                close_p = close_p.diff()
                close_p = close_p.dropna()
                is_data_stationary = self.check_stationarity(close_p)

        parameters = self.hyndman_khandakar(close_p)
        return parameters
    
      
    def get_innovation_std_dev(self, data):
        close_p = data["Close"]

        is_data_stationary = self.check_stationarity(close_p)
        while is_data_stationary == False:
                close_p = close_p.diff()
                close_p = close_p.dropna()
                is_data_stationary = self.check_stationarity(close_p)

        parameters = self.hyndman_khandakar(close_p)
        order = tuple(parameters[0])
        seasonal_order = tuple(parameters[1])
        model = sm.tsa.statespace.SARIMAX(close_p, order=order, seasonal_order=seasonal_order, trend="c")
        fitted = model.fit(disp=False)

        #sigma_hat = fitted.scale**0.5
        sigma_hat = fitted.resid.std(ddof=1)
        alpha = 0.05
        z = st.norm.ppf(1 - alpha/2)
        delta = z * sigma_hat

        return delta


    def get_assets_tickers(self):
        if not os.path.exists(self.data_folder_path):
            raise FileNotFoundError(f"Data folder not found at {self.data_folder_path}")
        files = [
            f for f in os.listdir(self.data_folder_path)
            if f.endswith(".csv") and f != "observed_assets.csv"
        ]
        assets_tickers = [f.replace("_data.csv", "").upper() for f in files]
        return assets_tickers, files
    

    def get_risk_free_rates(self):
        url = f"https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=10year&apikey={self.ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        risk_free_rate_data = response.json()
        # At this point, the unit is percent and interval is daily
        risk_free_rate_data = pd.DataFrame(risk_free_rate_data["data"])
        risk_free_rate_data = risk_free_rate_data.rename(columns={
            "date": "Date",
            "value": "Value"
        })
        risk_free_rate_data = risk_free_rate_data.sort_values(by="Date")
        risk_free_rate_data = risk_free_rate_data.reset_index()
        risk_free_rate_data = risk_free_rate_data.drop(columns="index")
        # This changes string values of Value column to numbers only if possible and divide by 100 to get the
        # percentage values as a fraction of 1
        risk_free_rate_data["Value"] = pd.to_numeric(risk_free_rate_data["Value"], errors="coerce") / 100
        risk_free_rate_data["Date"] = pd.to_datetime(risk_free_rate_data["Date"], errors="coerce")
        # The U.S. Department of Treasury posts rates daily on business days, however, for weekends and holidays
        # the rates stay unchanged, therefore ->
        # This will replace each missing value (NaN) with the most recent previous non-missing value
        risk_free_rate_data["Value"] = risk_free_rate_data["Value"].ffill()
        ten_years_risk_free_rate_data = risk_free_rate_data[risk_free_rate_data["Date"] >= self.ten_years_ago]
        return ten_years_risk_free_rate_data
    

    def get_asset_data(self, asset) -> DataFrame:
        data, _ = self.ts.get_daily(
            symbol=asset,
            outputsize="full" # returns full history (compact = last 100 days)
        )
        data = data.reset_index()
        data = data.rename(columns={
            "date": "Date",
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        })
        data = data.sort_values(by="Date")
        data = data.reset_index()
        data = data.drop(columns=["index"])
        data = data[data["Date"] >= self.ten_years_ago]
        data["Date"] = data["Date"].dt.tz_localize("America/New_York")

        splits = yf.Ticker(asset.upper()).splits
        splits.index = splits.index.tz_convert("America/New_York")
        splits = splits[splits.index > data["Date"].min()]

        risk_free_rate_data = self.get_risk_free_rates()

        if splits.shape[0] > 0:
            factor = 1
            for date, ratio in splits.items():
                factor *= ratio
                data.loc[data["Date"] < date, "Adjustment_factor"] = factor

            # For dates for which no split applied, dates after the last split, there is
            # no adjustment factor, factor of 1, that doesn't change stock's prices, is added.
            data.loc[data["Adjustment_factor"].isna(), "Adjustment_factor"] = 1

            # Stock prices are adjusted based on the adjustment factor that says how many new
            # shares were created from the old previous one
            data["Close"] = data["Close"] * (1 / data["Adjustment_factor"])

            # If there were splits, 50-days, 100-days Moving Averages, 
            # and log returns are calculated from adjusted prices
            data["50_day_MA"] = data["Close"].rolling(window=50).mean() # 50-days Moving average
            data["100_day_MA"] = data["Close"].rolling(window=100).mean() # 100-days Moving average
            data["Log_return"] = np.log(data["Close"] / data["Close"].shift(1)) # Daily log returns
            # Annualized daily volatility computed from rolling standard deviation of 20 days 
            data["Volatility"] = data["Log_return"].rolling(window=20).std() * np.sqrt(252)
        else:
            data["50_day_MA"] = data["Close"].rolling(window=50).mean() # 50-days Moving average
            data["100_day_MA"] = data["Close"].rolling(window=100).mean() # 100-days Moving average
            data["Log_return"] = np.log(data["Close"] / data["Close"].shift(1)) # Daily log returns
            # Annualized daily volatility computed from rolling standard deviation of 20 days 
            data["Volatility"] = data["Log_return"].rolling(window=20).std() * np.sqrt(252) 

        risk_free_rate_data["Date"] = risk_free_rate_data["Date"].dt.tz_localize("America/New_York")
        merged_data = pd.merge(data, risk_free_rate_data, on="Date", how="inner")
        merged_data = merged_data.rename(columns={"Value": "Risk-free Rate"})
        merged_data["Date"] = merged_data["Date"].dt.tz_localize(None)
        return merged_data
    

    def drop_and_load_data(self):
        for file, ticker in zip(self.files, self.tickers):
            # Drop files
            file_path = os.path.join(self.data_folder_path, file)
            if os.path.exists(file_path):
                os.remove(file_path)
            # Load files
            data = self.get_asset_data(ticker)
            data.to_csv(file_path, index=False)


    def get_new_parameters(self):
        file_path = os.path.join(self.data_folder_path, "observed_assets.csv")
        deltas_data = pd.read_csv(file_path, sep=",")
        updated_deltas = []
        for asset in deltas_data["asset"]:
            delta_data = pd.read_csv(os.path.join(self.data_folder_path, f"{asset.lower()}_data.csv"))
            delta = self.get_innovation_std_dev(delta_data)
            delta = round(delta)
            if delta < 2:
                 delta = 2
            updated_deltas.append({"asset": asset, "delta": delta})
        updated_deltas = pd.DataFrame(updated_deltas)
        os.remove(file_path)
        updated_deltas.to_csv(file_path, sep=",", index=False)

    
    def run_data_pipeline(self) -> None:
        self.drop_and_load_data()
        self.get_new_parameters()



            
    
# Data Pipeline execution
load_assets_data = LoadAssetsData()
print(load_assets_data.run_data_pipeline())


    

