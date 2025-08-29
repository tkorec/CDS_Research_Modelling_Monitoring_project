"""
This engine relates on real-time market data for
- NYSE American, BATS, ARCA, IEX, and Regional Exchanges (Network B) (NP,L1) 
- OPRA (US Options Exchanges) (NP,L1)

– Checking logs of the engine executed via run_call_debit_spread_engine.sh shell script called by Launch Agent
when the system wakes up or is booted after a certain time. 
cat /Users/tomaskorec/logs/engine.log

– Confirmation of active Launch Agent
launchctl list | grep com.tomas.engine.monitor

– Unloading and reloading the Launch Agent, so I can execute it manually
launchctl unload ~/Library/LaunchAgents/com.tomas.engine.monitor.plist
launchctl load ~/Library/LaunchAgents/com.tomas.engine.monitor.plist

– Check the Launch Agent is loaded
plutil ~/Library/LaunchAgents/com.tomas.engine.monitor.plist

– Updating Launch Agent
nano /Users/tomaskorec/Library/LaunchAgents/com.tomas.engine.monitor.plist
"""

import pandas as pd
import ast
from ib_insync import IB, Option
from collections import defaultdict
from monitoring import Monitoring
from order import Order
import datetime
import traceback
import sys
import os
import config


ib = IB()
ib.RequestTimeout = 30
ib.connect("127.0.0.1", 4002, clientId=config.CLIENT_ID)

monitoring = Monitoring(ib)
order = Order(ib)

# Inserting All outputs – prints and exceptions – into engine.log
"""
log_dir = "/Users/tomaskorec/logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "engine.log")
sys.stdout = open(log_file_path, "a")
sys.stderr = sys.stdout
"""

# 1/ Monitoring of followed assets
def monitor_followed_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    followed_assets = os.path.join(base_dir, "..", "data", "observed_assets.csv")
    followed_assets = os.path.abspath(followed_assets)
    print(f"[{datetime.datetime.now()}] Running monitor_followed_assets")
    followed_assets = pd.read_csv(followed_assets, sep=",")
    for _, row in followed_assets.iterrows():
        try:
            monitoring.followed_assets_monitor(row["asset"], row["delta"])
        except Exception as e:
            print(f"Error monitoring {row['asset']}: {e}")
            traceback.print_exc()

# 2/ Monitoring of opened positions
def monitor_positions():
    print(f"[{datetime.datetime.now()}] Running monitor_positions()")
    try:
        if not ib.isConnected():
            print("Reconnecting to IB...")
            ib.connect("127.0.0.1", 4002, clientId=config.CLIENT_ID)

        # All opened positions are requested and only the option positions are selected (stocks, ETFs, and other assets might be at the account as well)
        positions = ib.positions()
        positions = [pos for pos in positions if isinstance(pos.contract, Option)]
        if not positions:
            print("No positions found.")
            return # Early exit!

        # As the positions requested from IB Gateway are received independently, i.e., it's not apparent they are part of option strategies,
        # they need to be paired based on ticker and expiration date, so the correct positions of call debit spreads are received
        key_position_dict = defaultdict(dict)
        for position in positions:
            key = (position.contract.lastTradeDateOrContractMonth, position.contract.symbol)

            if position.position > 0:
                key_position_dict[key]["long_call"] = position
            elif position.position < 0:
                key_position_dict[key]["short_call"] = position

        # Each opened Call Debit Spread is checked whether it meets conditions for closing such position because of loss or profit
        for key, spread in key_position_dict.items():
            try:
                print(spread)
                monitoring.opened_positions_monitoring(key, spread)
            except Exception as e:
                print(f"Error monitoring position {key}: {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"Error in monitor_positions_and_orders: {e}")
        traceback.print_exc()



def run_once_daily():
    today_str = datetime.date.today().isoformat()
    now = datetime.datetime.now()
    current_hour = now.hour
    print(today_str)

    if not (18 <= current_hour < 21):
        #print(f"[{now}] Skipping daily run — outside of time window.")
        return # Early exit! It doesn't run monitor_positions_and_orders()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    last_run_path = os.path.join(base_dir, "last_run_date")

    if os.path.exists(last_run_path):
        with open(last_run_path, "r") as file:
            last_run = file.read().strip()
        if last_run == today_str:
            return # Early exit! It doesn't run monitor_positions_and_orders()
        
    monitor_followed_assets()

    with open(last_run_path, "w") as file:
        file.write(today_str)


if __name__ == "__main__":
    monitor_positions()
    run_once_daily()
    sys.exit(0) # Successful execution end
    ib.disconnect()
