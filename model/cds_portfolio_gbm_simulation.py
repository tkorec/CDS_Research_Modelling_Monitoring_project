import numpy as np
import pandas as pd
import os

class CDSPortfolioGBMSimulation():


    def __init__(self):
        self.MU = {}
        self.M = 110 # Number of simulations
        self.sim_years = 4
        self.sim_years_steps = self.sim_years * 365 # Number of steps in trading days per year
        self.data, self.files = self.load_data()
        self.portfolio_gbm, self.portfolio_50ma, self.portfolio_100ma = self.simulate_portfolio_price_paths()
        self.portfolio_iv = self.get_proxy_implied_volatility()
        self.observed_assets = self.load_observed_assets()


    def load_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder_path = os.path.join(script_dir, '..', 'data')

        if not os.path.exists(data_folder_path):
            raise FileNotFoundError(f"Data folder not found at {data_folder_path}")

        files = []
        data = {}

        for filename in os.listdir(data_folder_path):
            # Skip .py files and observed_assets.csv
            if filename.endswith(".py") or filename == "observed_assets.csv":
                continue

            if filename.endswith(".csv"):
                filepath = os.path.join(data_folder_path, filename)
                df = pd.read_csv(filepath)
                data[filename] = df
                files.append(filename)
        return data, files
    

    def load_observed_assets(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, '..', 'data', 'observed_assets.csv')
        observed_assets = pd.read_csv(data_path, sep=',')
        return observed_assets


    
    def simulate_portfolio_price_paths(self):
        portfolio_gbm = {}
        portfolio_50ma = {}
        portfolio_100ma = {}

        for asset in self.files:
            asset_data = self.data[asset]
            asset_data["Date"] = pd.to_datetime(asset_data["Date"])
            one_year_ago = asset_data["Date"].iloc[-1] - pd.Timedelta(days=365)
            last_year_data = asset_data[asset_data["Date"] >= one_year_ago]
            S0 = last_year_data["Close"].iloc[-1]
            mu = np.mean(last_year_data["Log_return"]) * 365
            self.MU[asset] = mu
            sigma = np.std(last_year_data["Log_return"], ddof=1) * np.sqrt(365)
            dt = self.sim_years / self.sim_years_steps # Calculate each time step

            St = np.exp(
                (mu - sigma ** 2 / 2) * dt
                + sigma * np.random.normal(0, np.sqrt(dt), size=(self.M, self.sim_years_steps)).T
            )
            St = np.vstack([np.ones(self.M), St])
            St = S0 * St.cumprod(axis=0)

            steps, simulations = St.shape
            Ma_50 = np.full((steps, simulations), np.nan)
            Ma_100 = np.full((steps, simulations), np.nan)

            window_50 = 50
            weights_50 = np.ones(window_50) / window_50

            window_100 = 100
            weights_100 = np.ones(window_100) / window_100


            for i in range(simulations):
                series_50ma = np.concatenate([asset_data["Close"][-(window_50-1):], St[:, i]])
                series_100ma = np.concatenate([asset_data["Close"][-(window_100-1):], St[:, i]])

                # Compute valid part
                valid_ma_50 = np.convolve(series_50ma, weights_50, mode="valid")
                valid_ma_100 = np.convolve(series_100ma, weights_100, mode="valid")

                # Fill result (pad initial values with NaN)
                Ma_50[:, i] = valid_ma_50[-steps:]
                Ma_100[:, i] = valid_ma_100[-steps:]

            portfolio_gbm[asset] = St
            portfolio_50ma[asset] = Ma_50
            portfolio_100ma[asset] = Ma_100

        return portfolio_gbm, portfolio_50ma, portfolio_100ma
    

    def get_proxy_implied_volatility(self):
        portfolio_iv = {}
        for asset in self.files:
            steps_count = len(self.portfolio_gbm[asset])
            simulations_count = self.portfolio_gbm[asset][0].shape[0]
            prices = self.portfolio_gbm[asset]
            historical_close = self.data[asset]["Close"].to_numpy()
            portfolio_iv[asset] = np.zeros((steps_count, simulations_count))
            for step in range(steps_count):
                start_index = step - 252
                if start_index >= 0:
                    series_window = prices[start_index:step, :]
                else:
                    missing_steps = -start_index
                    historical_part = historical_close[missing_steps:, None]
                    historical_part = np.repeat(historical_part, simulations_count, axis=1)
                    simulation_part = prices[0:step, :]
                    series_window = np.vstack([historical_part, simulation_part])
                log_returns = np.log(series_window[1:] / series_window[:-1])
                # Annualized volatility per simulation
                sigma = np.sqrt((252 / log_returns.shape[0]) * np.sum(log_returns**2, axis=0))
                portfolio_iv[asset][step, :] = sigma
        return portfolio_iv





