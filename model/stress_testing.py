import random
import os
import pandas as pd
import numpy as np
import json
from cds_portfolio_mc_simulation import CDSPortfolioMCSimulation


# StressTesting class inherits from CDSPortfolioMCSimulation
class StressTesting(CDSPortfolioMCSimulation):

    def __init__(self, sim_num, *args, **kwargs):
        super().__init__(sim_num, *args, **kwargs)
        self.scenarios = {
            "price_shock": [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05]
        }
        self.absolute_losses = []


    def ST_check_observed_underlying_assets(self, ST_data, step):
        for asset in self.simulated_portfolio.files:
            price = ST_data[asset][0][step]
            ma50_level = ST_data[asset][1][step]
            ma100_level = ST_data[asset][2][step]

            condition = (
                (ma50_level > ma100_level) and
                (price < ma50_level) and
                ((self.simulated_portfolio.sim_years_steps - step) > (365 * 2))
            )

            if condition == True:
                step_sigma = ST_data[asset][3][step]
                starting_step = step - 252
                if starting_step >= 0:
                    series = ST_data[asset][0][starting_step:step]
                else:
                    series = np.concatenate([
                        self.simulated_portfolio.data[asset]["Close"][starting_step:].to_numpy(),
                        ST_data[asset][0][0:step]
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


    def ST_check_opened_positions(self, ST_data, step, portfolio_pnl):
        ST_event_loss = 0
        for index, row in self.portfolio.iterrows():
            price = ST_data[row["asset"]][0][step]
            step_sigma = ST_data[row["asset"]][3][step]
            #price = self.simulated_portfolio.portfolio_gbm[row["asset"]][step, gbm_path]
            #price = price * (1 + sampled["price_shock"]) if "price_shock" in sampled else price
            #step_sigma = self.simulated_portfolio.portfolio_iv[row["asset"]][step, gbm_path]
            #step_sigma = sampled["volatility_shock"] if "volatility_shock" in sampled else step_sigma
            r = self.simulated_portfolio.data[row["asset"]]["Risk-free Rate"].iloc[-1]
            time_to_expiration = (row["expiration"] - step) / 365
            long_call_price = self.calculate_call_option_price(price, row["long_strike"], step_sigma, time_to_expiration, r)
            short_call_price = self.calculate_call_option_price(price, row["short_strike"], step_sigma, time_to_expiration, r)
            spread_value = long_call_price - short_call_price
            spread_return = (spread_value - row["spread_cost"]) / row["spread_cost"]

            step_threshold = step + 365 * self.YEARS_TO_EXP - 100
            if (spread_return >= self.PREMATURE_PERC_PROFIT) and (row["expiration"] - step_threshold) <= 100:
                profit_in_usd = row["max_profit"] * self.PREMATURE_PERC_PROFIT * row["contracts_bought"]
                ST_event_loss -= profit_in_usd # Profit decreases a loss (thus minus)
                portfolio_pnl.append(portfolio_pnl[-1] + profit_in_usd)
                self.portfolio = self.portfolio.drop(index=index)
                self.log.append(f"Profit (percentual) at step {step}: {self.portfolio}")

            if (spread_return <= self.PREMATURE_PERC_LOSS) and (row["expiration"] - step_threshold) <= 100:
                loss_in_usd = row["spread_cost"] * self.PREMATURE_PERC_LOSS * row["contracts_bought"]
                # Loss in USD is added as a sum because it was calculated from the negative share of premature percentage loss value
                ST_event_loss += (-loss_in_usd) # Loss_in_usd increases total loss – loss_in_usd is negative, therefore minus
                portfolio_pnl.append(portfolio_pnl[-1] + loss_in_usd)
                self.portfolio = self.portfolio.drop(index=index)
                self.log.append(f"Loss (percentual) at step {step}: {self.portfolio}")

            if row["expiration"] == step:
                if price > row["short_strike"]:
                    profit_in_usd = row["max_profit"] * row["contracts_bought"]
                    ST_event_loss -= profit_in_usd
                    portfolio_pnl.append(portfolio_pnl[-1] + profit_in_usd)
                    self.portfolio = self.portfolio.drop(index=index)
                    self.log.append(f"Profit (at expiration) at step {step}: {self.portfolio}")
                elif price < row["long_strike"]:
                    loss_in_usd = row["spread_cost"] * row["contracts_bought"]
                    ST_event_loss += loss_in_usd # Loss_in_usd increases total loss, there plus
                    # Minus sign is here because we don't multiply by negative share of return
                    portfolio_pnl.append(portfolio_pnl[-1] - loss_in_usd)
                    self.portfolio = self.portfolio.drop(index=index)
                    self.log.append(f"Loss (at expiration) at step {step}: {self.portfolio}")
                else:
                    # Whether the profit or loss is added to the portfolio here, it is added up because spread's max profit is divided
                    # by spread return share that is either positive or negative for profit or loss, respectively
                    portfolio_pnl.append(portfolio_pnl[-1] + spread_return * row["max_profit"] * row["contracts_bought"])
                    # Loss increases total loss (product of return, max_profit, and num of contracts is negative), therfore minus
                    # Profit decreases total loss (product of return, max_profit, and num of contreacts is positive), therefore minus
                    ST_event_loss -= spread_return * row["max_profit"] * row["contracts_bought"]
                    self.portfolio = self.portfolio.drop(index=index)
                    self.log.append(f"P/L (at expiration) at step {step}: {self.portfolio}")

        return portfolio_pnl, ST_event_loss


    def ST_export_simulation_report(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path_report = os.path.join(script_dir, "simulations", "stress_testing_results.txt")
        os.makedirs(os.path.dirname(output_path_report), exist_ok=True)
        with open(output_path_report, "w") as file:
            for item in self.log:
                file.write(f"{item}\n")
        output_path_data = os.path.join(script_dir, "simulations", "stress_testing_pnls.json")
        os.makedirs(os.path.dirname(output_path_data), exist_ok=True)
        with open(output_path_data, "w") as file:
            json.dump(self.all_portfolio_simulations, file)


    def get_proxy_implied_volatility_STv(self, updated_gbm_path, asset):
        proxy_iv = []
        step_count = len(updated_gbm_path)
        historical_close = self.simulated_portfolio.data[asset]["Close"].to_numpy()
        for step in range(step_count):
            start_index = step - 252
            if start_index >= 0:
                series_window = updated_gbm_path[start_index:step]
            else:
                missing_steps = -start_index
                historical_part = historical_close[missing_steps:]
                simulation_part = updated_gbm_path[0:step]
                series_window = np.concatenate([historical_part, simulation_part])
            log_returns = np.log(series_window[1:] / series_window[:-1])
            sigma = np.sqrt((252 / log_returns.shape[0]) * np.sum(log_returns**2, axis=0))
            proxy_iv.append(sigma)
        return proxy_iv
        

    
    def run_stress_testing_simulation(self):
        self.log.append(f"Stress testing performed on {pd.Timestamp.now()}")
        self.log.append(f"Underlying assets involved: {self.simulated_portfolio.files}")
        for i in range(self.number_of_simulations):
            portfolio_pnl = [0]
            ST_data = {}
            # GBM paths of observed assets and step on which a stress testing event happens are chosen
            gbm_path = random.randint(0, self.simulated_portfolio.M - 1)
            event_step = random.randint(300, (self.simulated_portfolio.sim_years_steps - 365 * self.YEARS_TO_EXP))
            price_shock = self.scenarios["price_shock"][random.randint(0, len(self.scenarios["price_shock"]) - 1)]
            self.log.append(f"Stress testing simulation {i} starting at step {event_step}")

            for asset in self.simulated_portfolio.files:
                # Stress Testing GBM path of prices
                asset_gbm_path = self.simulated_portfolio.portfolio_gbm[asset][:, gbm_path]
                prior_ST_event = asset_gbm_path[:event_step]
                post_ST_event = asset_gbm_path[event_step:]
                price = self.simulated_portfolio.portfolio_gbm[asset][event_step, gbm_path]
                new_price = price * (1 + price_shock)
                new_implied_vol = self.simulated_portfolio.portfolio_iv[asset][event_step, gbm_path] - 1 * (new_price / price)
                dt = (len(post_ST_event) / 365) / len(post_ST_event)
                Z = np.random.normal(0, np.sqrt(dt), size=len(post_ST_event))
                St = np.exp((self.simulated_portfolio.MU[asset] - new_implied_vol**2 / 2) * dt + new_implied_vol * Z)
                St = new_price * St.cumprod()
                updated_gbm_path = np.concatenate([prior_ST_event, St])
                # Stress Testing 50- and 100- days Moving Averages for new prices GBM path
                window_50 = 50
                window_100 = 100
                weights_50 = np.ones(window_50) / window_50
                weights_100 = np.ones(window_100) / window_100
                series_50ma = np.concatenate([self.simulated_portfolio.data["aapl_data.csv"]["Close"][-(window_50-1):], updated_gbm_path])
                series_100ma = np.concatenate([self.simulated_portfolio.data["aapl_data.csv"]["Close"][-(window_100 - 1):], updated_gbm_path])
                valid_50ma = np.convolve(series_50ma, weights_50, mode="valid")[-len(updated_gbm_path):]
                valid_100ma = np.convolve(series_100ma, weights_100, mode="valid")[-len(updated_gbm_path):]
                # Stress Testing Volatility
                implied_volatilities = self.get_proxy_implied_volatility_STv(updated_gbm_path, asset)

                asset_ST_data = [updated_gbm_path, valid_50ma, valid_100ma, implied_volatilities]
                ST_data[asset] = asset_ST_data
                
            for step in range(self.simulated_portfolio.sim_years_steps):
                # Checking the all observed underlying asset at each step whether the condition for opening Call Debit Spread position occured
                # Randomly selected GBM path for each simulation and time step on it are passed to the method
                self.ST_check_observed_underlying_assets(ST_data, step)
                portfolio_pnl, ST_event_loss = self.ST_check_opened_positions(ST_data, step, portfolio_pnl)

            portfolio_pnl = [x * 100 for x in portfolio_pnl]
            self.absolute_losses.append(ST_event_loss * 100)
            self.log.append(portfolio_pnl)
            self.all_portfolio_simulations.append(portfolio_pnl)

        self.log.append(f"Absolute losses in USD on Stress testing events: {self.absolute_losses}")
        percentual_loasses = [x / (self.account_size * 0.1) for x in self.absolute_losses]
        self.log.append(f"Percentual losses on Stress testing events: {percentual_loasses}")
        self.ST_export_simulation_report()

    



# Execute Stress Testing simulation
stress_testing = StressTesting(1000)
stress_testing.run_stress_testing_simulation()





