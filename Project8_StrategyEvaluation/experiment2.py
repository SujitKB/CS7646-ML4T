"""MC2-P1: Optimal Strategy.
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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

Student Name: Sujit Kanti Biswas
GT User ID: sbiswas67
GT ID: 903549376
"""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import pandas as pd  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import datetime as dt  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import os  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
from util import get_data, plot_data
import matplotlib.pyplot as plt
import QLearner as ql
import ManualStrategy as ms
import StrategyLearner as sl
import indicators as ind
from marketsimcode import compute_portvals, assess_portfolio

def experiment2(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):

    '''
    Hypothesis regarding how changing the value of 'impact' should affect 'in-sample' trading behavior
    '''

    start_date = sd
    end_date = ed
    sv = 100000
    commission = 0

    # ################################# Strategy Learner ################################# #
    # ManualStrategy and StrategyLearner Report: Commission: $0, Impact: 0, 0.005, 0.05.

    commission = 0.0
    impact=0.0
    learner_1 = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)  # constructor

    # The in-sample/development period is January 1, 2008 to December 31 2009.
    learner_1.addEvidence(symbol=symbol, sd=start_date, ed=end_date, sv=100000)

    df_trades_sl_1 = learner_1.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=100000)
    portvals_sl_1  = compute_portvals(df_trades_sl_1, start_val=sv, commission=commission, impact=impact)
    portvals_sl_normed_1 = portvals_sl_1 / portvals_sl_1.ix[0,]

    print_stats(portvals_sl_1, 'StrategyLearner_Impact_0.0', sd=start_date, ed=end_date)

    ########################
    commission = 0.0
    impact=0.005
    learner_2 = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)  # constructor

    # The in-sample/development period is January 1, 2008 to December 31 2009.
    learner_2.addEvidence(symbol=symbol, sd=start_date, ed=end_date, sv=100000)

    df_trades_sl_2 = learner_2.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=100000)
    portvals_sl_2  = compute_portvals(df_trades_sl_2, start_val=sv, commission=commission, impact=impact)
    portvals_sl_normed_2 = portvals_sl_2 / portvals_sl_2.ix[0,]

    print_stats(portvals_sl_2, 'StrategyLearner_Impact_0.005', sd=start_date, ed=end_date)

    ########################
    commission = 0.0
    impact = 0.05
    learner_3 = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)  # constructor

    # The in-sample/development period is January 1, 2008 to December 31 2009.
    learner_3.addEvidence(symbol=symbol, sd=start_date, ed=end_date, sv=100000)

    df_trades_sl_3 = learner_3.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=100000)
    portvals_sl_3 = compute_portvals(df_trades_sl_3, start_val=sv, commission=commission, impact=impact)
    portvals_sl_normed_3 = portvals_sl_3 / portvals_sl_3.ix[0,]

    print_stats(portvals_sl_3, 'StrategyLearner_Impact_0.05', sd=start_date, ed=end_date)

    print("No. of trades for Impact 0: ", len(df_trades_sl_3[df_trades_sl_1[symbol] != 0]))
    print("No. of trades for Impact 0.005: ", len(df_trades_sl_3[df_trades_sl_2[symbol] != 0]))
    print("No. of trades for Impact 0.05: ", len(df_trades_sl_3[df_trades_sl_3[symbol] != 0]))

    ###### Plot ######

    plt.clf()
    plt.figure(figsize=(20, 7))
    plt.plot(portvals_sl_normed_1.index, portvals_sl_normed_1, label='Strategy Learner - Impact 0')
    plt.plot(portvals_sl_normed_2.index, portvals_sl_normed_2, label='Strategy Learner - Impact 0.005')
    plt.plot(portvals_sl_normed_3.index, portvals_sl_normed_3, label='Strategy Learner - Impact 0.05')
    plt.grid(True)
    plt.title(symbol + ' Comparison - In-Sample Trading Behaviour with varying Impact values')
    plt.axis('tight')
    plt.ylabel('Portfolio Value (Normalized)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.savefig('experiment2.png', bbox_inches='tight')

def print_stats(df_portval, title, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31)):

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = assess_portfolio(df_portval)

    print()
    print(f"Date Range: {sd} to {ed}")
    print(f"Cumulative Return of {title}: {cum_ret}")
    print(f"Standard Deviation of {title}: {std_daily_ret}")
    print(f"Average Daily Return of {title}: {avg_daily_ret}")
    print(f"Sharpe Ratio of {title}: {sharpe_ratio}")
    print()

def author():
        return 'sbiswas67'  # Change this to your user ID

def main():
    # this is a helper function you can use to test your code  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # note that during autograding his function will not be called.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Define input parameters

    experiment2()


if __name__ == "__main__":
    main()
