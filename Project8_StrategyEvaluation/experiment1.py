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

def experiment1(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):

    '''
    Compare your Manual Strategy with your Strategy Learner in-sample trading JPM. Create a chart that shows:

        Value of the ManualStrategy portfolio (normalized to 1.0 at the start)
        Value of the StrategyLearner portfolio (normalized to 1.0 at the start)
        Value of the Benchmark portfolio (normalized to 1.0 at the start)
    '''

    start_date = sd
    end_date = ed
    sv = 100000

    # ManualStrategy and StrategyLearner Report: Commission: $9.95, Impact: 0.005 (unless stated otherwise).
    commission = 9.95
    impact = 0.005

    # ################################# Strategy Learner ################################# #
    learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)  # constructor

    # The in-sample/development period is January 1, 2008 to December 31 2009.
    learner.addEvidence(symbol=symbol, sd=start_date, ed=end_date, sv=100000)

    df_trades_sl = learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=100000)
    portvals_sl  = compute_portvals(df_trades_sl, start_val=sv, commission=commission, impact=impact)
    portvals_sl_normed = portvals_sl / portvals_sl.ix[0,]

    print_stats(portvals_sl, 'Strategy Learner', sd=start_date, ed=end_date)

    # ################################# Manual Strategy ################################# #
    df_trades_ms = ms.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=100000)
    portvals_ms  = compute_portvals(df_trades_ms, start_val=sv, commission=commission, impact=impact)
    portvals_ms_normed = portvals_ms / portvals_ms.ix[0,]

    print_stats(portvals_ms, 'Manual Strategy', sd=start_date, ed=end_date)

    # #################################### Benchmark #################################### #
    df_trades_bench = crtBenchmark(symbol=symbol, sd=start_date, ed=end_date, sv=100000)
    portvals_bench  = compute_portvals(df_trades_bench, start_val=sv, commission=commission, impact=impact)
    portvals_bench_normed = portvals_bench / portvals_bench.ix[0,]

    print_stats(portvals_bench, 'Benchmark', sd=start_date, ed=end_date)

    ###### Plot ######

    plt.clf()
    plt.figure(figsize=(20, 7))
    plt.plot(portvals_sl_normed.index, portvals_sl_normed, label='Strategy Learner')
    plt.plot(portvals_ms_normed.index, portvals_ms_normed, label='Manual Strategy')
    plt.plot(portvals_bench_normed.index, portvals_bench_normed, label='Benchmark')
    plt.grid(True)
    plt.title(symbol + ' In-Sample Comparison - Strategy Learner v/s Manual Strategy v/s Benchmark')
    plt.axis('tight')
    plt.ylabel('Portfolio Value (Normalized)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.savefig('experiment1.png', bbox_inches='tight')


def crtBenchmark(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):

    dates = pd.date_range(sd, ed)

    df_prices = get_data([symbol], dates)
    df_prices = df_prices.fillna(method='ffill').fillna(method='bfill')

    # BUY 1000 shares on first trading day, SELL those 1000 shares on last trading day
    date = [df_prices.index[0], df_prices.index[len(df_prices.index) - 1]]
    df_benchmark = pd.DataFrame(data=[1000, -1000], index=date, columns=[symbol])

    return df_benchmark

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

    experiment1()


if __name__ == "__main__":
    main()
