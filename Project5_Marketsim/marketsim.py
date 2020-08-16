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
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # this is the function the autograder will call to test your code  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # code should work correctly with either input  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # TODO: Your code here
    ##################### df_orders #########################
    # Import the required order file into a dataframe
    df_orders = pd.read_csv(orders_file, index_col='Date',
                          parse_dates=['Date'], na_values=['nan'])

    # Determine the relevant stock symbols and trade date range from the order file
    symbols = list(df_orders.Symbol.unique())
    dates = pd.date_range(df_orders.index[0], df_orders.index[-1])

    ##################### df_prices #########################
    # Import price for the relevant symbols
    df_prices = get_data(symbols, dates)
    df_prices.fillna(method='ffill')
    df_prices.fillna(method='bfill')
    df_prices = df_prices[symbols]          # remove SPY
    df_prices["CASH"] = 1.0                 # add CASH price

    ##################### df_trades #########################
    # 1) Build dataframe for trades from df_orders
    # 2) The new dataframe (df_trades) that represents changes in the number of shares each day by day for each assets.
    # 3) We shall use the same shape as df_prices, and is initially filled with zeros

    df_trades = pd.DataFrame(np.zeros((df_prices.shape)), df_prices.index, df_prices.columns)

    for index, order_row in df_orders.iterrows():

        # Calculate total $ value of shares Bought or Sold in each order
        share_value = df_prices.loc[index, order_row["Symbol"]] * order_row["Shares"]
        # Transaction cost
        transaction_cost = commission + impact * share_value

        # Update the no. of shares and CASH based on the type of transaction done (BUY/SELL)
        # The same asset may be traded more than once on a particular day
        if order_row["Order"] == "BUY":
            df_trades.loc[index, order_row["Symbol"]] = df_trades.loc[index, order_row["Symbol"]] + order_row["Shares"]
            df_trades.loc[index, "CASH"] = df_trades.loc[index, "CASH"] + share_value * (-1.0) - transaction_cost
        else:
            df_trades.loc[index, order_row["Symbol"]] = df_trades.loc[index, order_row["Symbol"]] - order_row["Shares"]
            df_trades.loc[index, "CASH"] = df_trades.loc[index, "CASH"] + share_value - transaction_cost

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

def author():
    return 'sbiswas67' #Change this to your user ID

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

def test_code():  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # this is a helper function you can use to test your code  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # note that during autograding his function will not be called.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Define input parameters

    import marketsim as ms
    print(author())
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    of = "./orders/orders2.csv"
    sv = 1000000
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Process orders  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    portvals = compute_portvals(orders_file = of, start_val = sv)

    if isinstance(portvals, pd.DataFrame):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        portvals = portvals[portvals.columns[0]] # just get the first column

    else:  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        "warning, code did not return a DataFrame"
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Get portfolio stats
    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = assess_portfolio(portvals)

    # Read in adjusted closing prices for SPY
    dates = pd.date_range(start_date, end_date)
    prices_SPX = get_data(['$SPX'], dates, addSPY=False)

    if isinstance(prices_SPX, pd.DataFrame):
        prices_SPX = prices_SPX[prices_SPX.columns[0]]      # just get the first column
    else:
        "warning, code did not return a prices_SPX DataFrame"

    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = assess_portfolio(prices_SPX)
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Compare portfolio against $SPX  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Date Range: {start_date} to {end_date}")
    print()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Sharpe Ratio of SPX : {sharpe_ratio_SPX}")
    print()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Cumulative Return of SPX : {cum_ret_SPX}")
    print()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Standard Deviation of SPX : {std_daily_ret_SPX}")
    print()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Average Daily Return of SPX : {avg_daily_ret_SPX}")
    print()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Final Portfolio Value: {portvals[-1]}")
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
if __name__ == "__main__":  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_code()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
