Tomas Korec | tk.korec@gmail.com | https://www.linkedin.com/in/tomas-korec-4419aa214/

Description of:
Call Debit Spread Modelled Strategy including Risk Quantification and Monitoring Engine Connected to IBKR TWS Platform:

The simple idea beyond this model is that if 95% confidential boundaries define the area within which a price of an underlying asset will
move with 95% probability while the drift will eventually move the prices towards the particular boundary or behind it, there is such combination
of Call Debit Spread lifespan, conditions for opening and closing positions, that might allow to create a significant profit within a shorter
period (due to options' leverage - represents 100 stocks – and neccessity to buy only a few contracts which lead to small fees) with berable 
risk (due to CDS' less significant time decay and longer lifespan that allows drift to take an effect). 
Note: The combination of lifespan and conditions and thresholds for opening and closing the positions was found by assumption and numerical changes 
of parameters without any optimisation -> there is a significant space for improvement.

Project Content:
- research folder – research scripts' purpose is to define the suitability of a particular underlying asset to become part of the CDS portfolio
– model folder – scripts model future probabilities of CDS portfolio hitting the cumulative loss of 10% of the entire account on all opened CDS positions at one time
- model folder - stress testing
- data incl. data pipeline
- engine – monitoring application connected to TWS IB platform

Phases:
(1) Research Phase
(2) Risk Quantification under the normal conditions
(3) Stress testing phase
(4) Monitoring phase

Research (research folder):
The purpose of the research is to decide what underlying asset to add to observed assets from which the CDS portfolio can be composed. The research
consists of two parts. Future simulation and backtesting on historical data:
1/ Underlying asset's historical prices are used for (1) drift and variance for future prices modelling via geometric Brownian Motion and (2) defining
95% confidential boundaries within which the options for Call Debit Spreads are chosen and spreads composed. As there are in average 3 opportunities a year for opening a CDS
position based on the conditions, Monte Carlo simulation is run for 3 CDS (3 randomly selected spreads and 3 randomly selected GBM price paths) per
iteration and these 3 CDS contracts are closed according to specified rules, resulting in various P&Ls per run.
2/ The importance of backtesting is that the future simulations are made from the ("current date") date on which the historical data was downloaded,
regardless of levels of 50-days and 100-days Moving Averages, which does represent the probabilities of profits and losses on the Call Debit Spreads,
but it doesn't represent the probabilities in relation to condition under which the Call Debit Spreads are to be opened.


Call Debit Spreads Portfolio's Risk Quantification (model folder):
Considering that the particular amount of funds is dedicated to this Call Debit Spread strategy and that there are conditions under which the CDS positions
are closed with a loss, e.g., particular percentage loss within the early stage of CDS position lifespan or the maximum loss at the expiration are achieved,
the question is what is the probability that at one particular point, the entire CDS portfolio experiences the loss equal to the dedicated amount of funds,
i.e., margin call occures for the CDS strategy. 
The Risk Quantification is perceived as a future risk simulation:
In case of future risk simulation, for each observed underlying asset, starting at the same day for each asset, several geometric Brownian Motion paths 
representing the future prices of that underlying assets are simulated. One GBM path for each asset considered for the CDS portfolio is picked randomly. Then,
all randomly picked GBM paths for all considered underlying assets are processed step by step and on each step, CDS position is either opened, enlarged, 
or closed based on the defined conditions. This process of random selection of GBM paths and processing them (opening and closing CDS positions) is repeated
a couple of times, so the statistical distribution of various gaines and losses on CDS portfolio throughout the continuous time is obtained. From this
statistical distribution is then deduced what is the probability of CDS portfolio's margin call occurence.


Stress Testing (model folder):
Normally, Stress Testing part is to test the portfolio's performance or maximum downfalls undr the extreme conditions. Stress Testing part has the same
objective here, but additionally to it, as Risk Quantification part simulation uses various randomly selected GBM paths that may go to different directions
even though two or more assets are correlated, so their prices development are to be same under given market (systematic risk) conditions, Stress Testing
part is to test the portfolio's behaviour for less significant parameters changes than usual for stress testing to see, how portfolio of correlated
assets behave. For this part (Stress Testing), scenario sudden change in price of an underlying asset followed by increase in implied volatility is used.



Data incl. Data Pipeline Script (data folder):
Data folder contains files with historical prices data of all observed underlying assets, i.e., those assets that were decided to be added to the CDS 
portfolio based on the research. Each file contains 10-years of historical Close prices, Risk-free rates, Logarithmic returns, 50-days and 100-days Moving 
Averages, Volumes, and Volatilities. Observed_assets.csv file contains ARMA model parameters for each observed underlying asset.
CDS portfolio risk simulation depends on historical data, i.e., historical prices of underlying assets are used for simulating future price paths
using gemoetric Brownian Motion on whose time steps are then calculated prices and returns of Call Debit Spread positions. To maintain the constancy of
calculated prices and returns on those CDS positions via Black Scholes model, constant value of risk-free rate is used, so every time a new underlying
asset is added to the observed assets from which the CDS portfolio can be built, Data Pipeline script is executed to load the latest historical data for
all underlying assets, so the starting date and ending date of the 10-years long period are the same, so the risk-free rate. The small discrepancies in
risk-free rates would probably didn't cause much of problems, but if one asset was added to the observed ones in differnt macroeconomic situation than
the other one, the possibility of risk-free rates being significantly different is higher. Together with executing Data Pipeline script, new parameters
for ARMA model are obtained.


Engine (engine folder):
The purpose of the Engine is to assist a trader with executing the CDS strategy. The engine doesn't trade itself, but it monitors the market for
conditions under which the new CDS position would be opened and monitor the opened CDS positions whether they are to be closed. Anytime the application
is to be taken, a trader receives a notifiction from the engine. Although there was the intention to implement automatic closure of CDS positions,
non-margin IBKR accounts don't allow naked put options and TWS API sees the legs (single options) of Call Debit Spread separately, not as parts of CDS
contract, so it prohibits Call Debit Spread's closure as one of the legs at closure is actually the naked put.


