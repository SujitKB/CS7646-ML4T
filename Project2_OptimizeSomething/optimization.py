"""MC1-P2: Optimize a portfolio.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
'''from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")'''

import pandas as pd  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import matplotlib.pyplot as plt  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import datetime as dt  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import scipy.optimize as spo
from util import get_data, plot_data

def assess_portfolio(allocs, normed_price, prices_SPY):

    start_val = 1000000
    risk_free_return = 0.
    sample_freq = 252

    ######## Get daily portfolio value ########
    normed_alloc = normed_price * allocs
    position_vals = normed_alloc * start_val
    portfolio_val = position_vals.sum(axis=1)

    ######## Get portfolio statistics (note: std_daily_ret = volatility) ########

    # Daily Returns
    daily_returns = portfolio_val.pct_change(1)
    daily_returns = daily_returns[1:]

    # cumulative returns
    cr = (portfolio_val[-1] / portfolio_val[0]) - 1

    # Average Daily Return
    adr = daily_returns.mean()

    # Standard Deviation of Daily Return
    sddr = daily_returns.std()

    # Sharpe Ratio: sqrt(k) * [mean(Daily Returns - Risk Free Returns)/std(Daily Returns)]
    # where Risk Free Returns = 0.0 per day
    sr = np.sqrt(sample_freq) * (daily_returns[:] - risk_free_return).mean() / daily_returns.std()

    return cr, adr, sddr, sr, portfolio_val

def negative_sharpe_ratio (allocs, normed_price, prices_SPY):

    return (-1.0) * assess_portfolio(allocs, normed_price, prices_SPY)[3]

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):


    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    'Fill (in place) missing values in data frame.'
    #prices.fillna(method='ffill', inplace=True)
    #prices.fillna(method='bfill', inplace=True)

    normed_price = prices.div(prices.iloc[0], axis='columns')

    start_val = 1000000     # start with a portfolio value of one million dollars
    sample_freq = 252       # Used in Sharpe Ratio

    # Define Initial portfolio allocation
    allocs = np.zeros(len(syms))
    allocs[:-1] = 1/len(syms)
    allocs[-1:] = 1 - (allocs[:-1].sum())   #Assign residual allocation to the last position (total sum = 1)

    constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})     # Total Allocation returned must be 1
    bounds = tuple((0, 1) for x in allocs)                          # Upper/lower bounds of allocation values

    optimized_alloc = spo.minimize(negative_sharpe_ratio, allocs, args=(normed_price,prices_SPY,), method='SLSQP', \
                                   bounds=bounds, constraints=constraints)

    allocs = optimized_alloc.x      # Update 'allocs' with the final optimized allocations

    cr, adr, sddr, sr, portfolio_val = assess_portfolio (allocs, normed_price, prices_SPY)


    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        # add code to plot here  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        df_temp = pd.concat([portfolio_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)

        # Normalize daily portfolio value and S&P500 by dividing with first row before doing comparison.
        df_temp = df_temp.div(df_temp.iloc[0], axis='columns')
        title = "Daily Portfolio Value and SPY"
        xlabel = "Date"
        ylabel = "Price"
        ax = df_temp.plot(title=title, fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.savefig('plot.png')

  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return allocs, cr, adr, sddr, sr  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def test_code():  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # This function WILL NOT be called by the auto grader  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Do not assume that any variables defined here are available to your function/code  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # It is only here to help you set up and test your code  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Define input parameters  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Note that ALL of these values will be set to different values by  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # the autograder!

    '''start_date = dt.datetime(2009,1,1)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    end_date = dt.datetime(2010,1,1)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM']'''

    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    symbols = ['IBM', 'X', 'GLD', 'JPM']

  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Assess the portfolio  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

    # Print statistics  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Start Date: {start_date}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"End Date: {end_date}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Symbols: {symbols}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Allocations:{allocations}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Sharpe Ratio: {sr}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Average Daily Return: {adr}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Cumulative Return: {cr}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
if __name__ == "__main__":  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # This code WILL NOT be called by the auto grader  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Do not assume that it will be called  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_code()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
