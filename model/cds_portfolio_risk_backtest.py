import pandas as pd
import numpy as np
import os
from arma_model import ModelARIMA


class CDSPortfolioRiskBacktest():


    def __init__(self):
        self.model = ModelARIMA()
        self.data, self.files = self.load_data()
        # Even though the latest and earliest dates throughout all observed assets are found, the requirment of this model
        # is to have data for all assets starting and ending with the same dates to maintain the consistency for risk-free rates.
        # When new asset is researched and added to the portfolio, before portfolio risk simulation is executed, pipeline for
        # loading observed asset's data is triggered, so the latest data is observed.
        self.all_dates = pd.to_datetime(
            pd.concat([df["Date"] for df in self.data.values()])
        ).drop_duplicates().sort_values().reset_index(drop=True)
        self.all_dates = self.extract_considered_dates()

        self.portfolio = pd.DataFrame(columns=["asset", "expiration", "long_strike", "short_strike", "long_price", "short_price", "spread_cost", 
             "max_profit", "position_size", "contracts_bought"])
        self.account_size = 100000 # $100'000
        # This means we can open with max of 0.6%, i.e., we can open up to 3 positions and not loosing more than 1% if we close them all with 50% loss
        self.trade_size = self.account_size * 0.006
        self.all_portfolio_backtests = [0]
        self.log = [] 

        self.YEARS_TO_EXP = 2
        self.PREMATURE_PERC_PROFIT = 0.35
        self.PREMATURE_PERC_LOSS = -0.5


    def extract_considered_dates(self):
        min_considered_date = self.all_dates.iloc[0] + pd.DateOffset(years=1)
        max_considered_date = self.all_dates.iloc[-1] - pd.DateOffset(years=2)
        considered_dates = self.all_dates[
            (self.all_dates >= min_considered_date) &
            (self.all_dates <= max_considered_date)
        ]
        return considered_dates       


    def load_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder_path = os.path.join(script_dir, '..', 'data')

        if not os.path.exists(data_folder_path):
            raise FileNotFoundError(f"Data folder not found at {data_folder_path}")

        files = os.listdir(data_folder_path)
        data = {}
        for filename in os.listdir(data_folder_path):
            if filename.endswith(".csv"):
                filepath = os.path.join(data_folder_path, filename)
                df = pd.read_csv(filepath)
                data[filename] = df
        return data, files
    

    def calculate_historical_volatility(self, historical_volatility_data):
        sum_squared_returns = historical_volatility_data["log_returns_squared"].sum()
        n = historical_volatility_data["log_returns"].count()
        sigma_historical = np.sqrt((252 / n) * sum_squared_returns) # Should this be 252 or 365?
        return round(sigma_historical, 6)
    
    

    def get_proxy_implied_volatility(self, date, asset):
        # In comparison to this function in CDS MC simulation class, 
        asset_data = self.data[asset]
        series = asset_data[
            (asset_data["Date"] <= date) &
            (asset_data["Date"] >= (date - pd.DateOffset(years=1)))
        ]["Close"]
        volatility_data = pd.DataFrame()
        volatility_data["log_returns"] = np.log(series / series.shift(1))
        volatility_data["log_returns_squared"] = volatility_data["log_returns"] ** 2
        step_sigma = self.calculate_historical_volatility(volatility_data)
        return step_sigma, series
    

    def get_spreads(self, series, date, step_sigma, asset, price):
        _, conf_int = self.model(series, [[1, 0, 1], [0, 0, 0, 0]])
        strike_list_lower_boundary = round(round(conf_int[0][0]))
        strike_list_upper_boundary = round(round(conf_int[0][1]))
        strike_prices_list = [strike for strike in range(strike_list_lower_boundary, strike_list_upper_boundary + 1)]
        strike_price_chain = {
            "strike": strike_prices_list,
            "price": np.full(len(strike_prices_list), None),
            "implied_volatility": np.full(len(strike_prices_list), step_sigma)
        }
        strike_price_chain = pd.DataFrame(strike_price_chain)
        strike_price_chain["price"] = strike_price_chain.apply(
            lambda row: pd.Series(
                self.calculate_call_option_price(
                    price,
                    row["strike"],
                    step_sigma,
                    self.YEARS_TO_EXP,
                    self.simulated_portfolio.data[asset]["Risk-free Rate"].iloc[-1]
                )
            ),
            axis=1
        )
        spreads = []
        for _, long_call in strike_price_chain.iterrows():
            for _, short_call in strike_price_chain.iterrows():
                spread_cost = long_call["price"] - short_call["price"]
                max_profit = short_call["strike"] - long_call["strike"] - spread_cost

                spreads.append({
                    "asset": asset,
                    "expiration": date + pd.DateOffset(years=self.YEARS_TO_EXP), # Is there such a date? Shouldn't this be adjusted to existing date?
                    "long_strike": long_call["strike"],
                    "short_strike": short_call["strike"],
                    "long_price": long_call["price"],
                    "short_price": short_call["price"],
                    "spread_cost": spread_cost,
                    "max_profit": max_profit
                })
        spreads = pd. DataFrame(spreads)
        spreads = spreads[
            (spreads["long_strike"] < spreads["short_strike"]) &
            (spreads["short_strike"] > price)
        ] # Only OTM spreads
        spreads = spreads[spreads["max_profit"] > spreads["spread_cost"]]
        return spreads
    

    def check_opened_positions(self, date):
        for index, row in self.portfolio.iterrows():
            price = self.data.loc[self.data["Date"] == date, "Close"].item()
            step_sigma, _ = self.get_proxy_implied_volatility(date, row["asset"])
            r = self.data.loc[self.data["Date"] == date, "Risk-free Rate"].item()
            time_to_expiration = (row["expiration"] - date).days() / 365 # Time to expiration is share of remaining days divided by 365 in case of daily steps
            long_call_price = self.calculate_call_option_price(price, row["long_strike"], step_sigma, time_to_expiration, r)
            short_call_price = self.calculate_call_option_price(price, row["short_strike"], step_sigma, time_to_expiration, r)
            spread_value = long_call_price - short_call_price
            spread_return = (spread_value - row["spread_cost"]) / row["spread_cost"]

            if (spread_return >= self.PREMATURE_PERC_PROFIT) and ((row["expiration"] - date).days() <= 100):
                profit_in_usd = row["max_profit"] * self.PREMATURE_PERC_PROFIT * row["contracts_bought"]
                self.all_portfolio_backtests.append(self.all_portfolio_backtests[-1] + profit_in_usd)
                self.portfolio = self.portfolio.drop(index=index)

            if (spread_return <= self.PREMATURE_PERC_LOSS) and ((row["expiration"] - date).days() <= 100):
                loss_in_usd = row["spread_cost"] * self.PREMATURE_PERC_LOSS * row["contracts_bought"]
                # Loss in USD is added as a sum because it was calculated from the negative share of premature percentage loss value
                self.all_portfolio_backtests.append(self.all_portfolio_backtests[-1] + loss_in_usd)
                self.portfolio = self.portfolio.drop(index=index)

            if row["expiration"] == date:
                if price > row["short_strike"]:
                    profit_in_usd = row["max_profit"] * row["contracts_bought"]
                    self.all_portfolio_backtests.append(self.all_portfolio_backtests[-1] + profit_in_usd)
                    self.portfolio = self.portfolio.drop(index=index)
                elif price < row["long_strike"]:
                    loss_in_usd = row["spread_cost"] * row["contracts_bought"]
                    # Minus sign is here because we don't multiply by negative share of return
                    self.all_portfolio_backtests.append(self.all_portfolio_backtests[-1] - loss_in_usd)
                    self.portfolio = self.portfolio.drop(index=index)
                else:
                    # Whether the profit or loss is added to the portfolio here, it is added up because spread's max profit is divided
                    # by spread return share that is either positive or negative for profit or loss, respectively
                    self.all_portfolio_backtests.append(self.all_portfolio_backtests[-1] + spread_return * row["max_profit"] * row["contracts_bought"])
                    self.portfolio = self.portfolio.drop(index=index)

            


    def check_observed_underlying_assets(self, date):
        for asset in self.files:
            price = self.data[asset].query(f'Date == "{date}"')["Close"].item()
            ma50_level = self.data[asset].query(f'Date == "{date}"')["50_day_MA"].item()
            ma100_level = self.data[asset].query(f'Date == "{date}"')["100_day_MA"].item()
            condition = (
                (ma50_level > ma100_level) and
                (price < ma50_level) #and
                # This line is not needed any more as we are selecting from dates that still leaving us with 2 years of future data,
                # so we can open CDS positions with 2 years long expiration
                #((date + pd.DateOffset(years=2)) >= self.data["Date"].max())
            )
            if condition == True:
                step_sigma, series = self.get_proxy_implied_volatility(date, asset)
                # Should we run this script as a simulation if spread is chosen from spreads chain randomly?
                spreads = self.get_spreads(series, date, step_sigma, asset, price)
                try:
                    random_spread = spreads.sample(n=1)
                    spread_cost = random_spread["spread_cost"].item()
                    number_of_contracts = round(self.trade_size / (spread_cost * 100))
                    position_cost = spread_cost * 100 * number_of_contracts
                    current_allocation = self.portfolio.loc[self.portfolio["asset"] == asset, "position_size"].sum()
                    max_allocation = self.account_size * 0.01

                    # Some spreads are too expensive they cannot be bought with the current account size
                    # or its portion dedicated to Call Debit Spreads + 
                    if (number_of_contracts > 0) and ((current_allocation + position_cost) < max_allocation):
                        random_spread["position_size"] = position_cost
                        random_spread["contracts_bought"] = number_of_contracts
                        self.portfolio = pd.concat([self.portfolio, random_spread], ignore_index=True)
                except Exception as e:
                    #self.log.append(e)
                    continue
    
    
    def backtest_portfolio_risk(self):    
        for date in self.all_dates:
            #self.check_opened_positions(date)
            self.check_observed_underlying_assets(date)

        # Portfolio should be empty at the end of each simulation
        #if len(self.portfolio) != 0:
        #    self.log.append(f"Error on {}")

    
    
portfolio_risk_backtest = CDSPortfolioRiskBacktest()
print(portfolio_risk_backtest.files)