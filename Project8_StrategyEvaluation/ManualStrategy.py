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
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import QLearner as ql
from indicators import *
from marketsimcode import compute_portvals, assess_portfolio

# this method should use the existing policy and test it against new data
def testPolicy(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):

    symbol = [symbol]
    df_prices = getPrices(symbol, sd, ed)
    df_indicators = getIndicators(symbol, df_prices)

    df_trades = pd.DataFrame(data=None, columns=df_prices.columns, index=df_prices.index).fillna(0)

    net_holding = 0  # Start with nothing
    for day in range(df_prices.shape[0]):

        if (df_indicators['bb %'].iloc[day] < 0.18 and df_indicators['price/sma'].iloc[day] < 0.95) \
                                                 or df_indicators['CCI'].iloc[day] > 80.0:
            signal = 'BUY'
        elif (df_indicators['bb %'].iloc[day] > 0.75 and df_indicators['price/sma'].iloc[day] > 1.083) \
                                                    or df_indicators['CCI'].iloc[day] < -80.0:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        trade_units = getTradeAction(signal, net_holding)
        df_trades.loc[df_prices.index[day], symbol] = trade_units
        net_holding += trade_units

    return df_trades

def getPrices(symbol, sd, ed):

    dates = pd.date_range(sd, ed)
    df_prices_all = ut.get_data(symbol, dates)      # automatically adds SPY
    df_prices = df_prices_all[symbol].fillna(method='ffill').fillna(method='bfill')

    return df_prices

def getIndicators(symbol, df_prices):

    # Build the 3 required indicators.

    # Bollinger Band
    df_Bollinger = compute_bollingerBBval(df_prices)

    # price/SMA
    df_SMA = compute_SMA(df_prices)

    # CCI
    df_CCI = compute_CCI(df_prices)

    frames = [df_Bollinger['bb %'], df_SMA['price/sma'], df_CCI['CCI']]
    df_indicators = pd.concat(frames, axis=1)

    return df_indicators


def getTradeAction(signal, net_holding):

    if signal == 'BUY':
        if net_holding == 0:
            trade = 1000
        elif net_holding == -1000:
            trade = 2000
        elif net_holding == 1000:  # Hold
            trade = 0
    elif signal == 'SELL':
        if net_holding == 0:
            trade = -1000
        elif net_holding == 1000:
            trade = -2000
        elif net_holding == -1000:  # Hold
            trade = 0
    else:  # Hold - for any other scenarios
        trade = 0

    return trade

def crtBenchmark(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):

    dates = pd.date_range(sd, ed)

    df_prices = get_data([symbol], dates)
    df_prices = df_prices.fillna(method='ffill').fillna(method='bfill')

    # BUY 1000 shares on first trading day, SELL those 1000 shares on last trading day
    date = [df_prices.index[0], df_prices.index[len(df_prices.index) - 1]]
    df_benchmark = pd.DataFrame(data=[1000, -1000], index=date, columns=[symbol])

    return df_benchmark

def generate_test(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):

    start_date = sd
    end_date = ed
    commission = 9.95
    impact = 0.005

    # ################################# In-Sample ################################# #
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)

    df_trades_ms = testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=100000)
    portvals_ms = compute_portvals(df_trades_ms, start_val=sv, commission=commission, impact=impact)
    portvals_ms_normed = portvals_ms / portvals_ms.ix[0,]

    df_trades_bench = crtBenchmark(symbol=symbol, sd=start_date, ed=end_date, sv=100000)
    portvals_bench = compute_portvals(df_trades_bench, start_val=sv, commission=commission, impact=impact)
    portvals_bench_normed = portvals_bench / portvals_bench.ix[0,]

    # Display stats
    print_stats(portvals_ms, 'Manual Strategy In-Sample', sd=start_date, ed=end_date)
    print_stats(portvals_bench, 'Benchmark In-Sample', sd=start_date, ed=end_date)

    plt.clf()
    plt.figure(figsize=(20, 7))
    plt.plot(portvals_ms_normed.index, portvals_ms_normed, color='red', label='Manual Strategy')
    plt.plot(portvals_bench_normed.index, portvals_bench_normed, color='green', label='Benchmark')

    for day in range(df_trades_ms.shape[0]):
        if df_trades_ms.iloc[day].item() > 0:
            plt.axvline(x=df_trades_ms.index[day], color='blue', linestyle='-')
            plt.legend(loc='upper left')
        elif df_trades_ms.iloc[day].item() < 0:
            plt.axvline(x=df_trades_ms.index[day], color='black', linestyle='-')
            plt.legend(loc='upper left')

    plt.grid(True)
    plt.title(symbol + ' In-Sample Comparison - Manual Strategy v/s Benchmark')
    plt.axis('tight')
    plt.ylabel('Portfolio Value (Normalized)')
    plt.xlabel('Date')

    # Legends
    colors = ['blue', 'black', 'red', 'green']
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
    labels = ['BUY', 'SELL', 'Manual Strategy', 'Benchmark']
    plt.legend(lines, labels, loc='upper left')

    plt.savefig('MS-Insample.png', bbox_inches='tight')

    # ################################# Out-Sample ################################ #
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2011, 12, 31)
    df_trades_ms = testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=100000)
    portvals_ms = compute_portvals(df_trades_ms, start_val=sv, commission=commission, impact=impact)
    portvals_ms_normed = portvals_ms / portvals_ms.ix[0,]

    df_trades_bench = crtBenchmark(symbol=symbol, sd=start_date, ed=end_date, sv=100000)
    portvals_bench = compute_portvals(df_trades_bench, start_val=sv, commission=commission, impact=impact)
    portvals_bench_normed = portvals_bench / portvals_bench.ix[0,]

    # Display stats
    print_stats(portvals_ms, 'Manual Strategy Out-Sample', sd=start_date, ed=end_date)
    print_stats(portvals_bench, 'Benchmark Out-Sample', sd=start_date, ed=end_date)

    plt.clf()
    plt.figure(figsize=(20, 7))
    plt.plot(portvals_ms_normed.index, portvals_ms_normed, color='red', label='Manual Strategy')
    plt.plot(portvals_bench_normed.index, portvals_bench_normed, color='green', label='Benchmark')

    for day in range(df_trades_ms.shape[0]):
        if df_trades_ms.iloc[day].item() > 0:
            plt.axvline(x=df_trades_ms.index[day], color='blue', linestyle='-')
        elif df_trades_ms.iloc[day].item() < 0:
            plt.axvline(x=df_trades_ms.index[day], color='black', linestyle='-')

    plt.grid(True)
    plt.title(symbol + ' Out-Sample Comparison - Manual Strategy v/s Benchmark')
    plt.axis('tight')
    plt.ylabel('Portfolio Value (Normalized)')
    plt.xlabel('Date')

    # Legends
    colors = ['blue', 'black', 'red', 'green']
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
    labels = ['BUY', 'SELL', 'Manual Strategy', 'Benchmark']
    plt.legend(lines, labels, loc='upper left')

    plt.savefig('MS-Outsample.png', bbox_inches='tight')

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
    return 'sbiswas67'  # replace tb34 with your Georgia Tech username.

if __name__=="__main__":

    #df_trades = testPolicy()
    #df_benchtrade = crtBenchmark()
    generate_test()

    print("One does not simply think up a strategy")
