"""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Atlanta, Georgia 30332  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
All Rights Reserved  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template code for CS 4646/7646  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
and other users of this template code are advised not to share it with others  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or to make it available on publicly viewable websites including repositories  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or edited.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
We do grant permission to share solutions privately with non-students such  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
as potential employers. However, sharing with other current or future  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT honor code violation.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
-----do not edit anything above this line---  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Student Name: Sujit Kanti Biswas (replace with your name)
GT User ID: sbiswas67 (replace with your User ID)
GT ID: 903549376 (replace with your GT ID)
"""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import datetime as dt  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import pandas as pd  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import util as ut  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import random
import QLearner as ql
from indicators import *
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
class StrategyLearner(object):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # constructor  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def __init__(self, verbose = False, impact = 0.0, commission=0.0):
        self.verbose = verbose  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self.impact = impact  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self.commission = commission
        self.epochs = 100
        self.disc_buckets = 10
        self.qlearner = ql.QLearner(num_states=1000, \
                                    num_actions=3, \
                                    alpha=0.2, \
                                    gamma=0.9, \
                                    rar=0.60, \
                                    radr=0.999, \
                                    dyna=0, \
                                    verbose=False)  # initialize the learner

    # this method should create a QLearner, and train it for trading  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def addEvidence(self, symbol = "AAPL", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        # add your code to do learning here
        symbol = [symbol]
        df_prices = self.getPrices(symbol, sd, ed)
        df_states = self.getDiscrIndicators(symbol, df_prices)

        # The below loop structure using 'epochs' is adapted from testqlearner.py module provided with the previous project
        for epoch in range(1, self.epochs + 1):

            port_val = sv
            net_holding = 0  # Start with nothing

            for day in range (df_prices.shape[0]):
                reward = 0.

                if df_prices.index[day] == df_prices.index[0]:      # First trading date

                    action = self.qlearner.querysetstate(df_states.iloc[0].item())    #df_states.iloc[day].item()
                    trade_units = self.getTradeAction(action, net_holding)
                else:
                    transaction_cost = self.commission + self.impact * net_holding    # Transaction cost

                    reward = (((df_prices[symbol].iloc[day].item() / df_prices[symbol].iloc[day-1].item()) - 1.0) \
                             * net_holding) - transaction_cost

                    action = self.qlearner.query(df_states.iloc[day].item(), reward)
                    trade_units = self.getTradeAction(action, net_holding)

                port_val += reward
                net_holding += trade_units

            if epoch >= 10 and prior_port_val == port_val:
                break
            prior_port_val = port_val

  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # this method should use the existing policy and test it against new data  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def testPolicy(self, symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000):

        symbol = [symbol]
        df_prices = self.getPrices(symbol, sd, ed)
        df_states = self.getDiscrIndicators(symbol, df_prices)

        df_trades = pd.DataFrame(data=None, columns=df_prices.columns, index=df_prices.index).fillna(0)

        net_holding = 0  # Start with nothing
        for day in range(df_prices.shape[0]):

            action = self.qlearner.querysetstate(df_states.iloc[day].item())
            trade_units = self.getTradeAction(action, net_holding)

            df_trades.loc[df_prices.index[day], symbol] = trade_units
            net_holding += trade_units

        if self.verbose: print(type(df_trades)) # it better be a DataFrame!
        if self.verbose: print(df_trades)
        if self.verbose: print(df_prices)

        return df_trades

    def getPrices(self, symbol, sd, ed):

        dates = pd.date_range(sd, ed)
        df_prices_all = ut.get_data(symbol, dates)      # automatically adds SPY
        df_prices = df_prices_all[symbol].fillna(method='ffill').fillna(method='bfill')

        return df_prices

    def getDiscrIndicators(self, symbol, df_prices):

        # Build the 3 required indicators and discretize them.

        # Bollinger Band
        df_Bollinger = compute_bollingerBBval(df_prices)
        df_Bollinger = df_Bollinger['bb %']
        df_Bollinger = pd.qcut(df_Bollinger, self.disc_buckets, labels=False)

        # price/SMA
        df_SMA = compute_SMA(df_prices)
        df_SMA = df_SMA['price/sma']
        df_SMA = pd.qcut(df_SMA, self.disc_buckets, labels=False)

        # CCI
        df_CCI = compute_CCI(df_prices)
        df_CCI = df_CCI['CCI']
        df_CCI = pd.qcut(df_CCI, self.disc_buckets, labels=False)

        # Sum-up all the discretized indicators to create 1000 states
        df_states = 100 * df_Bollinger
        df_states += 10 * df_SMA
        df_states += df_CCI
        df_states  = df_states.fillna(0).astype(int)

        return df_states


    def getTradeAction(self, action, net_holding):

        if action == 1:
            if net_holding == 0:
                trade = 1000
            elif net_holding == -1000:
                trade = 2000
            elif net_holding == 1000:  # Hold
                trade = 0
        elif action == 2:
            if net_holding == 0:
                trade = -1000
            elif net_holding == 1000:
                trade = -2000
            elif net_holding == -1000:  # Hold
                trade = 0
        else:  # Hold - for any other scenarios
            trade = 0

        return trade

    def crtBenchmark(self, symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000):

        dates = pd.date_range(sd, ed)

        df_prices = get_data([symbol], dates)
        df_prices = df_prices.fillna(method='ffill').fillna(method='bfill')

        # BUY 1000 shares on first trading day, SELL those 1000 shares on last trading day
        date = [df_prices.index[0], df_prices.index[len(df_prices.index) - 1]]
        df_benchmark = pd.DataFrame(data=[1000, -1000], index=date, columns=[symbol])

        return df_benchmark

    def author(self):
        return 'sbiswas67'  # replace tb34 with your Georgia Tech username.

if __name__=="__main__":  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print("One does not simply think up a strategy")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
