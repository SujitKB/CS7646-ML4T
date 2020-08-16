"""MC2-P1: Market simulator.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def compute_portvals(df_orders, start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # code should work correctly with either input  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # TODO: Your code here
    ##################### df_orders #########################
    symbol = list(df_orders.columns)
    dates = pd.date_range(df_orders.index[0], df_orders.index[-1])

    ##################### df_prices #########################
    # Import price for the relevant symbol
    df_prices = get_data(symbol, dates)
    df_prices.fillna(method='ffill')
    df_prices.fillna(method='bfill')
    df_prices = df_prices[symbol]          # remove SPY
    df_prices["CASH"] = 1.0                # add CASH price

    ##################### df_trades #########################
    # 1) Build dataframe for trades from df_price
    # 2) The new dataframe (df_trades) that represents changes in the number of shares each day by day for each assets.
    # 3) We shall use the same shape as df_prices, and is initially filled with zeros

    df_trades = pd.DataFrame(np.zeros((df_prices.shape)), df_prices.index, df_prices.columns)

    for index, order_row in df_orders.iterrows():
        shares = order_row.item()

        # Calculate total $ value of shares Bought or Sold in each order
        share_value = df_prices.loc[index, symbol] * shares
        share_value = share_value.item()

        # Transaction cost
        transaction_cost = commission + impact * share_value

        # Update the no. of shares and CASH based on the type of transaction done (BUY/SELL)
        # The same asset may be traded more than once on a particular day
        if share_value > 0:     # BUY
            #df_trades.loc[index, symbol] = df_trades.loc[index, symbol] + shares
            df_trades.loc[index, symbol] = shares
            df_trades.loc[index, "CASH"] = df_trades.loc[index, "CASH"] + share_value * (-1.0) - transaction_cost
        else:
            #df_trades.loc[index, symbol] = df_trades.loc[index, symbol] - shares
            df_trades.loc[index, symbol] = shares
            df_trades.loc[index, "CASH"] = df_trades.loc[index, "CASH"] + share_value* (-1.0) - transaction_cost

    ##################### df_holdings #########################
    # 1) Build dataframe for holdings from df_trades.
    # 2) The new dataframe (df_holdings) shall represents cumulative holdings of stock and CASH up to that date.
    # 3) We shall use the same shape as df_trade, and is initially filled with zeros.

    df_holdings = pd.DataFrame(np.zeros((df_trades.shape)), df_trades.index, df_trades.columns)

    # 4) We shall then initialize the first row (start trade date) of df_holdings with same share units as that of the
    #    first day traded share amount.
    df_holdings.iloc[0, :-1] = df_trades.iloc[0, :-1].copy()

    # 5) The CASH column of the first row shall be updated as (total shares Bought/SOLD + start_val).
    df_holdings.loc[df_holdings.index[0], 'CASH'] = df_trades.loc[df_holdings.index[0], 'CASH'] + start_val

    # 6) We shall then roll forward and add the previous day holding with the current day trade (i.e. adjusting the
    # previous holdings with today's change due to price or additional trades) to update the current day holding.

    for i in range(df_holdings.shape[0]-1):
        df_holdings.loc[df_holdings.index[i+1]] = df_trades.loc[df_holdings.index[i+1]] + df_holdings.loc[df_holdings.index[i]]

    ##################### df_assetvalue #######################
    # This is the cumulative (i.e. summed up to that day) holding value for each of the stocks and CASH.
    df_assetvalue = df_prices * df_holdings

    ##################### df_portvals #########################
    # This is the cumulative (i.e. summed up to that day) total portfolio value.
    df_portvals = df_assetvalue.sum(axis=1)

    return df_portvals

def assess_portfolio(portfolio_val, sv=1000000, risk_free_return=0.0, sample_freq=252.0):

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

    return cr, adr, sddr, sr

def author():
    return 'sbiswas67' #Change this to your user ID

  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
if __name__ == "__main__":  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    pass
