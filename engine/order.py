from ib_insync import LimitOrder
from ib_insync import Option, Contract, ComboLeg
from report import Report
import config

class Order():

    def __init__(self, ib):
        self.ib = ib
        self.report = Report()


    # Closing Call Debit Spread requires closing the position as soon as possible because further waiting could lead to additional losses.
    # That's why the bid price is defined close to the ask one as long-term expiration options don't have such liquidity. For the future developments,
    # the distance from ask price will be adjusted according to liquidity of each followed asset. 
    def close_call_debit_spread(self, spread, p_or_l):
        try:
            if not self.ib.isConnected():
                self.ib.connect("127.0.0.1", 7497, clientId=config.CLIENT_ID)
            # Contracts' information from opened positions extracted from TWS IB platform aren't sufficient enough for sending an order to the market.
            # The contract information contains only the data about the call option in general, i.e., strike price and expiration date, but if we want to
            # create an instance of Option contract for passage into an active market, we need to specify a few more values.
            long_opt = spread["long_call"].contract
            short_opt = spread["short_call"].contract
            long_contract = Option(conId=long_opt.conId, symbol=long_opt.symbol, lastTradeDateOrContractMonth=long_opt.lastTradeDateOrContractMonth, strike=long_opt.strike,
                                right=long_opt.right, multiplier=long_opt.multiplier, currency=long_opt.currency, exchange="")
            short_contract = Option(conId=short_opt.conId, symbol=short_opt.symbol, lastTradeDateOrContractMonth=short_opt.lastTradeDateOrContractMonth, strike=short_opt.strike,
                                right=short_opt.right, multiplier=short_opt.multiplier, currency=short_opt.currency, exchange="") 
            long_detail = self.ib.reqContractDetails(long_contract)
            short_detail = self.ib.reqContractDetails(short_contract)
            long_listing_exchange = long_detail[0].contract.exchange
            short_listing_exchange = short_detail[0].contract.exchange
            long_contract = long_detail[0].contract
            long_contract.exchange = long_listing_exchange or "CBOE"
            short_contract = short_detail[0].contract
            short_contract.exchange = short_listing_exchange or "CBOE"
            long_call_option = self.ib.reqMktData(long_contract)
            short_call_option = self.ib.reqMktData(short_contract)
            self.ib.sleep(2)
            # While closing any Call Debit Spread, I buy a call option and sell a call option on a different strikes. Problem is that TWS IB
            # platform sees such an order as two separate orders and accounts other than margin one don't have permission to sell naked call.
            # To overcome this and send the order to the market via API, instance of Contract must be used and orders of buying and selling call 
            # options are passed as instances of ComboLeg. However, this approach has a significant limitation on its own as it doesn't allow to set 
            # separate prices for each combo leg which might be important especially when strike prices of long and short calls are distant, so the
            # spread between short and call options' prices is also significant. In this function, I use limit price based on real-time bid/ask prices
            # of each leg which should at least reflect the actual market pricing (not just a midpoint) and increase the chances of order filling.
            # Still, it depends on the difference between short/long calls' strike prices and asset's liquidity. As the legs of the combo order cannot
            # be edited separately, closing the position under the better conditions or closing it at all will still be dependent on personal involvement
            # of the trader.
            mid_long_price = (long_call_option.ask + long_call_option.bid) / 2 if long_call_option.ask and long_call_option.bid else None
            mid_short_price = (short_call_option.ask + short_call_option.bid) / 2 if short_call_option.ask and short_call_option.bid else None
            limit_price = round(mid_long_price - mid_short_price, 2)

            combo = Contract()
            combo.symbol = long_opt.symbol
            #combo.secId = "BAG"
            combo.currency = long_opt.currency
            combo.exchange = long_contract.exchange
            combo.comboLegs = [
                ComboLeg(conId=short_opt.conId, ratio=1, action="BUY", exchange=short_contract.exchange, openClose=1),
                ComboLeg(conId=long_opt.conId, ratio=1, action="SELL", exchange=long_contract.exchange, openClose=1)
            ]
            order = LimitOrder("SELL", long_opt.multiplier, limit_price, transmit=True)
            trade = self.ib.placeOrder(combo, order)
            self.ib.sleep(2)
            if p_or_l == -1:
                message = f"""✅ --- Message from close_call_debit_spread() function in Order() class ---\n
                {long_opt.symbol} – {long_opt.lastTradeDateOrContractMonth} Loosing Call Debit Spread Order was 
                successfully sent to the platform and was {trade.orderStatus.status}"""
                self.report.send_telegram_message(message)
            if p_or_l == 1:
                message = f"""✅ --- Message from close_call_debit_spread() function in Order() class ---\n
                {long_opt.symbol} - {long_opt.lastTradeDateOrContractMonth} Profitable Call Debit Spread Order was
                successfully sent to the platform and was {trade.orderStatus.status}"""
        except Exception as e:
            # In case the Call Debit Spread met the conditions for being closed as the loosing position and any error occured, I want to
            # know about it asap via Telegram, so I can react on such an event and close the position manually, but I also want to have a
            # log about this error.
            message = f"""{long_opt.symbol} – {long_opt.lastTradeDateOrContractMonth} Call Debit Spread Order hasn't been sent
            to the market because of following error: {e}"""
            self.report.send_telegram_message(message)
            print(f"""❌ --- Message from close_call_debit_spread() function in Order() class ---\nCall Debit Spread Order hasn't been sent
            to the market because of following error: {e}""")


