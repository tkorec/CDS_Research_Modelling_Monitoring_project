import pandas as pd
from ib_insync import Stock, Option
from datetime import datetime, timedelta
from itertools import chain
from model import Model
from order import Order
from report import Report
import config

class Monitoring():

    def __init__(self, ib):
        self.ib = ib
        #self.model = Model()
        #self.order = Order(ib)
        #self.report = Report()


    def followed_assets_monitor(self, asset, delta):
        # Historical prices for the one last year with daily interval are picked and 50- and 100-days moving averages are counted from them
        contract = Stock(asset, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        asset_prices = self.ib.reqHistoricalData(
            contract,
            endDateTime=datetime.today().strftime("%Y%m%d %H:%M:%S"),
            durationStr="1 Y",
            barSizeSetting="1 day",
            whatToShow="MIDPOINT", # Data type (use "MIDPOINT" for close price approximation)
            useRTH=True, # Regular trading hours
            formatDate=1 # Format as a string
        )
        
        asset_data = {
            "Date": [bar.date for bar in asset_prices],
            "Close": [bar.close for bar in asset_prices]
        }
        asset_data = pd.DataFrame(asset_data)
        asset_data["50_day_MA"] = asset_data["Close"].rolling(window=50).mean()
        asset_data["100_day_MA"] = asset_data["Close"].rolling(window=100).mean()

        # If the condition for possibility of opening a Call Debit Spread is met, on the one-year historical prices time series of an underlying asset, 
        # the prediction of a price one period ahead is made, so the 95% confidential interval is set. This interval determines the strike prices that
        # will be considered while looking for particular Call Debit Spreads.
        latest_close = asset_data["Close"].iloc[-1]
        latest_50_day_MA = asset_data["50_day_MA"].iloc[-1]
        latest_100_day_MA = asset_data["100_day_MA"].iloc[-1]
        condition = config.trading_condition(latest_close, latest_50_day_MA, latest_100_day_MA)
        if condition == True:
            strike_list_lower_boundary = round(latest_close - delta)
            strike_list_upper_boundary = round(latest_close + delta)

            # The expiration date closest to date one year from now is found
            option_parameters = self.ib.reqSecDefOptParams(contract.symbol, "", contract.secType, contract.conId)
            smart_params = [params for params in option_parameters if params.exchange == "SMART"]
            expiration_dates = set(chain.from_iterable(params.expirations for params in smart_params))
            expiration_dates = [datetime.strptime(date, "%Y%m%d") for date in expiration_dates]
            years_from_today = datetime.today() + timedelta(days=365*2)
            dates_within_next_year = [date for date in expiration_dates if date.year == years_from_today.year]
            closest_expiration = min(dates_within_next_year, key=lambda x: abs(x - years_from_today))

            # Strikes from the available ones within confidential boundaries are chosen.
            options = [params for params in smart_params if closest_expiration.strftime("%Y%m%d") in params.expirations]
            all_strikes = set(chain.from_iterable(params.strikes for params in options))
            selected_strikes = [strike for strike in all_strikes if strike_list_lower_boundary <= strike <= strike_list_upper_boundary]

            # Current prices of Call options with that paricular expiration date (closest available date to the date one year from now) and
            # available strike prices within 95% confidential interval given by time series prediction are extracted and DataFrame of all
            # such options is created
            options = []
            for i in range(0, len(selected_strikes)):
                try:
                    contract = Option(symbol=asset, lastTradeDateOrContractMonth=closest_expiration.strftime("%Y%m%d"), 
                                strike=selected_strikes[i], right="C", exchange="SMART", currency="USD")
                    ticker = self.ib.reqMktData(contract, genericTickList="", snapshot=False)
                    self.ib.sleep(1)
                    options.append({"Bid": ticker.bid, "Ask": ticker.ask, "Strike": ticker.contract.strike, "Expiration": ticker.contract.lastTradeDateOrContractMonth})
                except Exception:
                    continue
            options = pd.DataFrame(options)

            spreads = []
            for i, long_call in options.iterrows():
                for j, short_call in options.iloc[i+1:].iterrows():
                    spread_cost = (long_call["Ask"] - short_call["Bid"]) * 100
                    max_profit = (short_call["Strike"] - long_call["Strike"]) * 100 - spread_cost

                    spreads.append({
                        "Expiration": closest_expiration.strftime("%Y%m%d"),
                        "Long_Strike": long_call["Strike"],
                        "Short_Strike": short_call["Strike"],
                        "Long_Price": long_call["Ask"],
                        "Short_Price": short_call["Bid"],
                        "Spread_Cost": spread_cost,
                        "Max_Profit": max_profit,
                        "Max_Loss": spread_cost
                    })
            
            spreads = pd.DataFrame(spreads)
            """
            spreads = spreads[
                (spreads["Long_Strike"] < spreads["Short_Strike"]) &
                (spreads["Short_Strike"] > asset_data.iloc[-1]["Close"]) &
                (spreads["Max_Profit"] > spreads["Max_Loss"])
            ]
            """
            spreads = spreads.dropna()
            if not spreads.empty:
                message = f"Suitable Call Debit Spreads for{asset}\n<pre>{spreads.to_string(index=False)}</pre>"
                config.send_telegram_message(message)


    def opened_positions_monitoring(self, spread):
        #broker_commission = 1 if spread["long_call"].position == 1 else 0.25 * spread["long_call"].position * 2
        spread_cost = (spread["long_call"].avgCost - spread["short_call"].avgCost)
        asset_symbol = spread["long_call"].contract.symbol

        long_call_option = self.ib.reqMktData(spread["long_call"].contract)
        short_call_option = self.ib.reqMktData(spread["short_call"].contract)
        current_spread_price = long_call_option.bid - short_call_option.bid

        expiration_date = spread["long_call"].contract.lastTradeDateOrContractMonth
        days_to_exp = (expiration_date - datetime.today()).days

        # Condition for closing loosing position
        condition_for_closing_loosing_position = config.closing_loosing_position_condition(current_spread_price, spread_cost, days_to_exp)
        if condition_for_closing_loosing_position == True:
            message = f"Loosing CDS for premature closure: {asset_symbol} - long strike {spread['long_call']}, short strike {spread['short_call']}"
            config.send_telegram_message(message)

        # Condition for closing profitable position
        condition_for_closing_profitable_position = config.closing_profitable_position_condition(current_spread_price, spread_cost, days_to_exp)
        if condition_for_closing_profitable_position == True:
            message = f"Profitable CDS for premature closure: {asset_symbol} - long strike {spread['long_call']}, short strike {spread['short_call']}"
            config.send_telegram_message(message)

        if days_to_exp == 14:
            message = f"{asset_symbol} - long strike {spread['long_call']}, short strike {spread['short_call']} will expire in 14 days"
            config.send_telegram_message(message)

           

  





