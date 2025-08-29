#!/bin/bash

LOG_FILE="/Users/tomaskorec/logs/engine.log"
PYTHON_SCRIPT="/Users/tomaskorec/Documents/Portfolio Modelling/Call Debit Spreads/engine/engine.py"
PYTHON_EXEC="/opt/homebrew/bin/python3.11"

# Get current hour and weekday
HOUR=$(date +%H)
DAY=$(date +%u)  # 1 = Monday, 7 = Sunday

# Get idle time in seconds
IDLE_TIME=$(ioreg -c IOHIDSystem | awk '/HIDIdleTime/ { print int($NF/1000000000); exit }')

echo "---- Script check at $(date) ----" >> "$LOG_FILE"

# Conditions: between 4pm and 10pm, weekday, and user active in last 5 mins
if [ "$HOUR" -ge 16 ] && [ "$DAY" -le 5 ] && [ "$IDLE_TIME" -lt 300 ]; then
echo "✅ Conditions met. Running Python script..." >> "$LOG_FILE"
    "$PYTHON_EXEC" "$PYTHON_SCRIPT" >> "$LOG_FILE" 2>&1
else
    echo "❌ Conditions NOT met. Skipping." >> "$LOG_FILE"
    echo "Hour: $HOUR | Day: $DAY | Idle: ${IDLE_TIME}s" >> "$LOG_FILE"
fi

echo "" >> "$LOG_FILE"