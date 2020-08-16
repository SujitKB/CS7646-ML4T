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
from marketsimcode import compute_portvals, assess_portfolio

class TheoreticallyOptimalStrategy(object):

    def testPolicy(self,symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv=100000):

        dates = pd.date_range(sd, ed)
        units = []
        tradedates = []
        net_holding = 0

        df_prices = get_data([symbol], dates)
        df_prices = df_prices.fillna(method='ffill').fillna(method='bfill')
        num_dates = len(df_prices.index)


        # We shall loop till the second last day [num_dates - 1].
        # The trade on last day shall be SELL OUT everything
        daycount = 0
        while daycount < (num_dates - 2):
            daycount += 1
            if df_prices[symbol][daycount] < df_prices[symbol][daycount + 1]:
                if net_holding == 0:
                    units.append(1000)
                    tradedates.append(df_prices.index[daycount])
                    net_holding += 1000
                elif net_holding == -1000:
                    units.append(2000)
                    tradedates.append(df_prices.index[daycount])
                    net_holding += 2000
                elif net_holding == 1000:  # Hold
                    units.append(0)
                    tradedates.append(df_prices.index[daycount])
                    net_holding += 0
            elif df_prices[symbol][daycount] > df_prices[symbol][daycount + 1]:
                if net_holding == 0:
                    units.append(-1000)
                    tradedates.append(df_prices.index[daycount])
                    net_holding += -1000
                elif net_holding == 1000:
                    units.append(-2000)
                    tradedates.append(df_prices.index[daycount])
                    net_holding += -2000
                elif net_holding == -1000:  # Hold
                    units.append(0)
                    tradedates.append(df_prices.index[daycount])
                    net_holding += 0
            else:  # Hold - for any other scenarios
                units.append(0)
                tradedates.append(df_prices.index[daycount])
                net_holding += 0

        if net_holding > 0:
            units.append(-net_holding)
            tradedates.append(df_prices.index[num_dates - 1])

        df_trades = pd.DataFrame(data=units, index=tradedates, columns=[symbol])

        return df_trades

    def testBenchmark(self,symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv=100000):

        dates = pd.date_range(sd, ed)

        df_prices = get_data([symbol], dates)
        df_prices = df_prices.fillna(method='ffill').fillna(method='bfill')
        num_dates = len(df_prices.index)

        # BUY 1000 shares on first trading day, SELL those 1000 shares on last trading day
        #date = [df_prices.index[0], df_prices.index[len(df_prices.index) - 1]]
        df_benchmark = pd.DataFrame(data=[1000, -1000],
                                    index=[df_prices.index[0], df_prices.index[num_dates - 1]],
                                    columns=[symbol])

        return df_benchmark

    def author(self):
        return 'sbiswas67'  # Change this to your user ID

def main():
    # this is a helper function you can use to test your code  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # note that during autograding his function will not be called.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Define input parameters

    start_date = '2008-1-1'
    end_date = '2009-12-31'
    symbol = 'JPM'
    sv = 100000

    tos = TheoreticallyOptimalStrategy()

    ############### Theoritically Optimal Strategy ###############
    df_trades = tos.testPolicy(symbol, start_date, end_date, sv)  	        # Process trades	  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    portvals_tos = compute_portvals(df_trades, start_val = sv, commission=0.00, impact=0.00)        # Compute Portfolio Value

    # Get portfolio stats 
    cum_ret_tos, avg_daily_ret_tos, std_daily_ret_tos, sharpe_ratio_tos = assess_portfolio(portvals_tos)    

    ######################## Benchmark #########################
    df_benchmark = tos.testBenchmark(symbol, start_date, end_date, sv)          # Process trades
    portvals_bchmrk = compute_portvals(df_benchmark, start_val = sv, commission=0.00, impact=0.00)   # Compute Portfolio Value

    # Get portfolio stats 
    cum_ret_bchmrk, avg_daily_ret_bchmrk, std_daily_ret_bchmrk, sharpe_ratio_bchmrk = assess_portfolio(portvals_bchmrk)

    # Compare TOS against Benchmark  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Date Range: {start_date} to {end_date}")
    print()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Cumulative Return of TOS: {cum_ret_tos}")
    print(f"Cumulative Return of Benchmark : {cum_ret_bchmrk}")
    print()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Standard Deviation of TOS: {std_daily_ret_tos}")
    print(f"Standard Deviation of Benchmark : {std_daily_ret_bchmrk}")
    print()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Average Daily Return of TOS: {avg_daily_ret_tos}")
    print(f"Average Daily Return of Benchmark : {avg_daily_ret_bchmrk}")
    print()

    portvals_bchmrk_normed  = portvals_bchmrk / portvals_bchmrk.ix[0,]
    portvals_tos_normed     = portvals_tos / portvals_tos.ix[0,]
    portvals_bchmrk_normed  = portvals_bchmrk_normed.to_frame()
    portvals_tos_normed     = portvals_tos_normed.to_frame()


    re = portvals_bchmrk_normed.join(portvals_tos_normed, lsuffix='_benchmark', rsuffix='_TOS_portfolio')
    re.columns = ['Benchmark', 'Theoretically Optimal Portfolio']
    ax = re.plot(title="Relative Performance of TOS v/s Benchmark", fontsize=12,
                 color=["green", "red"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio Value")
    plt.savefig('TOSvsBench.png')


if __name__ == "__main__":
    main()
