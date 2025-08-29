import numpy as np
import pandas as pd
import os
import json
import math
import random
from scipy.stats import norm
from cds_portfolio_gbm_simulation import CDSPortfolioGBMSimulation
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")



class CDSPortfolioMCSimulation():


    def __init__(self, sim_num):
        self.simulated_portfolio = CDSPortfolioGBMSimulation()
        self.portfolio = pd.DataFrame(columns=["asset", "expiration", "long_strike", "short_strike", "long_price", "short_price", "spread_cost", 
             "max_profit", "position_size", "contracts_bought"])
        self.account_size = 100000 # $100'000
        # This means we can open with max of 0.6%, i.e., we can open up to 3 positions and not loosing more than 1% if we close them all with 50% loss
        self.trade_size = self.account_size * 0.006
        self.number_of_simulations = sim_num
        self.YEARS_TO_EXP = 2
        self.PREMATURE_PERC_PROFIT = 0.35
        self.PREMATURE_PERC_LOSS = -0.5
        self.log = []
        self.all_portfolio_simulations = []


    def calculate_call_option_price(self, S, K, sigma, T, r):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)      
        return call_price


    def calculate_historical_volatility(self, historical_volatility_data):
        sum_squared_returns = historical_volatility_data["log_returns_squared"].sum()
        n = historical_volatility_data["log_returns"].count()
        sigma_historical = np.sqrt((252 / n) * sum_squared_returns)
        return round(sigma_historical, 6)


    def get_proxy_implied_volatility(self, gbm_path, step, asset):
        # If 252 steps (trading days) is deducted from the current step, but there isn't enough steps/days in the GBM simulation,
        # the start of the interval is from non-existing index, so the series is empty
        series = self.simulated_portfolio.portfolio_gbm[asset][step - 252:step, gbm_path]
        if len(series) == 0:
            missing_steps = abs(step - 252)
            series = np.concatenate([
                self.simulated_portfolio.data[asset]["Close"][-missing_steps:].to_numpy(),
                self.simulated_portfolio.portfolio_gbm[asset][0:step, gbm_path]
            ])
        series = pd.Series(series)
        volatility_data = pd.DataFrame()
        volatility_data["log_returns"] = np.log(series / series.shift(1))
        volatility_data["log_returns_squared"] = volatility_data["log_returns"] ** 2
        step_sigma = self.calculate_historical_volatility(volatility_data)
        return step_sigma, series

        

    def check_opened_positions(self, gbm_path, step, portfolio_pnl):
        for index, row in self.portfolio.iterrows():
            price = self.simulated_portfolio.portfolio_gbm[row["asset"]][step, gbm_path]
            step_sigma, _ = self.get_proxy_implied_volatility(gbm_path, step, row["asset"])
            # For consistency, datasets of underlying assets should be created on the same day, so the risk-free rates are same for the modelling
            r = self.simulated_portfolio.data[row["asset"]]["Risk-free Rate"].iloc[-1]
            time_to_expiration = (row["expiration"] - step) / 365
            long_call_price = self.calculate_call_option_price(price, row["long_strike"], step_sigma, time_to_expiration, r)
            short_call_price = self.calculate_call_option_price(price, row["short_strike"], step_sigma, time_to_expiration, r)
            spread_value = long_call_price - short_call_price
            spread_return = (spread_value - row["spread_cost"]) / row["spread_cost"]

            step_threshold = step + 365 * self.YEARS_TO_EXP - 100
            if (spread_return >= self.PREMATURE_PERC_PROFIT) and (row["expiration"] - step_threshold) <= 100:
                profit_in_usd = row["max_profit"] * self.PREMATURE_PERC_PROFIT * row["contracts_bought"]
                portfolio_pnl.append(portfolio_pnl[-1] + profit_in_usd)
                self.portfolio = self.portfolio.drop(index=index)
                self.log.append(f"**** Step: {step}; Asset: {row['asset']}; Spread value: {spread_value}; Spread return: {spread_return} ****")
                self.log.append(f"Profit (percentual) at step {step}: {self.portfolio}")

            if spread_return <= self.PREMATURE_PERC_LOSS and (row["expiration"] - step_threshold) <= 100:
                loss_in_usd = row["spread_cost"] * self.PREMATURE_PERC_LOSS * row["contracts_bought"]
                # Loss in USD is added as a sum because it was calculated from the negative share of premature percentage loss value
                portfolio_pnl.append(portfolio_pnl[-1] + loss_in_usd)
                self.portfolio = self.portfolio.drop(index=index)
                self.log.append(f"**** Step: {step}; Asset: {row['asset']}; Spread value: {spread_value}; Spread return: {spread_return} ****")
                self.log.append(f"Loss (percentual) at step {step}: {self.portfolio}")

            if row["expiration"] == step:
                if price > row["short_strike"]:
                    profit_in_usd = row["max_profit"] * row["contracts_bought"]
                    portfolio_pnl.append(portfolio_pnl[-1] + profit_in_usd)
                    self.portfolio = self.portfolio.drop(index=index)
                    self.log.append(f"**** Step: {step}; Asset: {row['asset']}; Spread value: {spread_value}; Spread return: {spread_return} ****")
                    self.log.append(f"Profit (at expiration) at step {step}: {self.portfolio}")
                elif price < row["long_strike"]:
                    loss_in_usd = row["spread_cost"] * row["contracts_bought"]
                    # Minus sign is here because we don't multiply by negative share of return
                    portfolio_pnl.append(portfolio_pnl[-1] - loss_in_usd)
                    self.portfolio = self.portfolio.drop(index=index)
                    self.log.append(f"**** Step: {step}; Asset: {row['asset']}; Spread value: {spread_value}; Spread return: {spread_return} ****")
                    self.log.append(f"Loss (at expiration) at step {step}: {self.portfolio}")
                else:
                    # Whether the profit or loss is added to the portfolio here, it is added up because spread's max profit is divided
                    # by spread return share that is either positive or negative for profit or loss, respectively
                    portfolio_pnl.append(portfolio_pnl[-1] + spread_return * row["max_profit"] * row["contracts_bought"])
                    self.portfolio = self.portfolio.drop(index=index)
                    self.log.append(f"**** Step: {step}; Asset: {row['asset']}; Spread value: {spread_value}; Spread return: {spread_return} ****")
                    self.log.append(f"P/L (at expiration) at step {step}: {self.portfolio}")

        return portfolio_pnl
    

    def get_spread(self, series, step, step_sigma, asset, price):
        r = self.simulated_portfolio.data[asset]["Risk-free Rate"].iloc[-1]
        delta = self.simulated_portfolio.observed_assets.loc[
            self.simulated_portfolio.observed_assets["asset"] == asset.split("_")[0].upper(), "delta"
        ].item()
        strike_list_lower_boundary = price - delta
        strike_list_upper_boundary = price + delta
        strikes = np.arange(strike_list_lower_boundary, strike_list_upper_boundary + 1)
        strikes = strikes[strikes > price]
        # Create all (long, short) combinations where short strike > long strike, i.e., Call Debit Spreads
        long_strikes, short_strikes = np.meshgrid(strikes, strikes, indexing="ij")
        mask_upper_triangle = short_strikes > long_strikes
        long_strikes = long_strikes[mask_upper_triangle]
        short_strikes = short_strikes[mask_upper_triangle]
        # Calculating option prices for all strikes (vectorized)
        long_prices = np.array([
            self.calculate_call_option_price(price, ls, step_sigma, self.YEARS_TO_EXP, r)
            for ls in long_strikes
        ])
        short_prices = np.array([
            self.calculate_call_option_price(price, ss, step_sigma, self.YEARS_TO_EXP, r)
            for ss in short_strikes
        ])
        spread_costs = long_prices - short_prices
        max_profits = short_strikes - long_strikes - spread_costs
        # Applying conditions in one vectorized step
        valid_mask = (short_strikes > price)
        if not np.any(valid_mask):
            return None
        # Random pick of valid CDS
        valid_indices = np.flatnonzero(valid_mask)
        choice_idx = random.choice(valid_indices)
        return {
            "asset": asset,
            "expiration": step + 365 * 2,
            "long_strike": int(long_strikes[choice_idx]),
            "short_strike": int(short_strikes[choice_idx]),
            "long_price": float(long_prices[choice_idx]),
            "short_price": float(short_prices[choice_idx]),
            "spread_cost": float(spread_costs[choice_idx]),
            "max_profit": float(max_profits[choice_idx])
        }
        

    def check_observed_underlying_assets(self, gbm_path, step):
        for asset in self.simulated_portfolio.files:
            price = self.simulated_portfolio.portfolio_gbm[asset][step, gbm_path]
            ma50_level = self.simulated_portfolio.portfolio_50ma[asset][step, gbm_path]
            ma100_level = self.simulated_portfolio.portfolio_100ma[asset][step, gbm_path]

            condition = (
                (ma50_level > ma100_level) and
                (price < ma50_level) and
                ((self.simulated_portfolio.sim_years_steps - step) > (365 * 2)) #and # There must be at least two years available in the future steps of the simulation
                # This needs to be uncomented once I start simulating
                #(asset not in portfolio["asset"].values) # Returning single True/False based on whether the asset is already in the portfolio or not
            )

            if condition == True:
                #step_sigma, series = self.get_proxy_implied_volatility(gbm_path, step, asset)
                step_sigma = self.simulated_portfolio.portfolio_iv[asset][step, gbm_path]
                starting_step = step - 252
                if starting_step >= 0:
                    series = self.simulated_portfolio.portfolio_gbm[asset][starting_step:step, gbm_path]
                else:
                    series = np.concatenate([
                        self.simulated_portfolio.data[asset]["Close"][starting_step:].to_numpy(),
                        self.simulated_portfolio.portfolio_gbm[asset][0:step, gbm_path]
                    ])
                spread = self.get_spread(pd.Series(series), step, step_sigma, asset, price)
                # There were no spreads because early 50-days and 100-days Moving Averages had Nan values
                try:
                    spread_cost = spread["spread_cost"]
                    number_of_contracts = round(self.trade_size / (spread_cost * 100))
                    position_cost = spread_cost * 100 * number_of_contracts
                    current_allocation = self.portfolio.loc[self.portfolio["asset"] == asset, "position_size"].sum()
                    max_allocation = self.account_size * 0.01

                    # Some spreads are too expensive they cannot be bought with the current account size
                    # or its portion dedicated to Call Debit Spreads + 
                    if (number_of_contracts > 0) and ((current_allocation + position_cost) < max_allocation):
                        spread["position_size"] = position_cost
                        spread["contracts_bought"] = number_of_contracts
                        self.portfolio = pd.concat([self.portfolio, pd.DataFrame([spread])], ignore_index=True)
                        self.log.append(f"Opening position at step {step}: {self.portfolio}")
                except Exception as e:
                    self.log.append(f"Error for asset {asset} at step {step}: {e}")
                    continue


    def export_simulation_report(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path_report = os.path.join(script_dir, "simulations", "mc_simulation_results.txt")
        os.makedirs(os.path.dirname(output_path_report), exist_ok=True)
        with open(output_path_report, "w") as file:
            for item in self.log:
                file.write(f"{item}\n")
        output_path_data = os.path.join(script_dir, "simulations", "mc_simulation_pnls.json")
        os.makedirs(os.path.dirname(output_path_data), exist_ok=True)
        with open(output_path_data, "w") as file:
            json.dump(self.all_portfolio_simulations, file)         


    def run_monte_carlo_simulation(self):
        self.log.append(f"Monte Carlo simulation performed on {pd.Timestamp.now()}")
        self.log.append(f"Underlying assets involved: {self.simulated_portfolio.files}")
        for i in range(self.number_of_simulations):
            self.log.append(f"---- Simulation {i} ----")
            portfolio_pnl = [0]
            gbm_path = random.randint(0, self.simulated_portfolio.M - 1) # Number of GBM simulations passed from CDSPortfolioGBMSimulation class
            for step in range(self.simulated_portfolio.sim_years_steps): # Number of steps in GBM paths passed from CDSPortfolioGBMSimulation class
                # Checking the currently opened Call Debit Spread positions whether they are to be closed or exercised
                # Randomly selected GBM path for each simulation and time step on it are passed to the method
                portfolio_pnl = self.check_opened_positions(gbm_path, step, portfolio_pnl)
                # Checking the all observed underlying asset at each step whether the condition for opening Call Debit Spread position occured
                # Randomly selected GBM path for each simulation and time step on it are passed to the method
                self.check_observed_underlying_assets(gbm_path, step)

            portfolio_pnl = [x * 100 for x in portfolio_pnl]
            self.all_portfolio_simulations.append(portfolio_pnl)
            self.log.append(portfolio_pnl)
            # Portfolio should be empty at the end of each simulation
            if len(self.portfolio) != 0:
                self.log.append(f"Error on {gbm_path}")

        self.export_simulation_report()
    

    
mc_simulation = CDSPortfolioMCSimulation(100)
mc_simulation.run_monte_carlo_simulation()