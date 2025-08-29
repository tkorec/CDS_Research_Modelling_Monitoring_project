import pandas as pd
from ib_insync import *
from datetime import datetime
import random


ib = IB()
if ib.isConnected():
    ib.disconnect()

client_id = random.randint(1000, 9999)

ib.connect("127.0.0.1", 7497, clientId=client_id)
#ib.reqMarketDataType(3)
print("Connected:", ib.isConnected())

contract = Stock(symbol="AAPL", exchange="SMART", currency="USD")
ib.qualifyContracts(contract)
print(contract)

bars = ib.reqHistoricalData(
    contract,
    endDateTime="",
    durationStr="1 Y",
    barSizeSetting="1 day",
    whatToShow="MIDPOINT", # Data type (use "MIDPOINT" for close price approximation)
    useRTH=True, # Regular trading hours
    formatDate=1 # Format as a string
)

print(bars)

ib.disconnect()