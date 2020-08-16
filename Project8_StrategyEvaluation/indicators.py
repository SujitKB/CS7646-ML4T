"""MC2-P6: INDICATOR EVALUATION.
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
import matplotlib.pyplot as plt
from util import get_data, plot_data

def normalize_data(df):
    """Normalize stock prices using the first row of the price dataframe"""
    return df/df.ix[0,]

################## Compute and Return Market Indictors ##################

# 1) Bollinger Band / BB %
def compute_bollingerBBval(df_prices, roll_window=20):

    df_bollinger = pd.DataFrame(index=df_prices.index)
    df_bollinger['prices'] = df_prices


    rolling_std = df_bollinger['prices'].rolling(window=roll_window).std()
    rolling_mean = df_bollinger['prices'].rolling(window=roll_window).mean()

    df_bollinger['sma'] = rolling_mean
    df_bollinger['upper band'] = rolling_mean + 2.0 * rolling_std
    df_bollinger['lower band'] = rolling_mean - 2.0 * rolling_std

    df_bollinger['bb value'] = (df_bollinger['prices'] - rolling_mean) / (rolling_std)
    df_bollinger['bb %'] = (df_bollinger['prices'] - df_bollinger['lower band']) / \
                          (df_bollinger['upper band'] - df_bollinger['lower band'])

    return df_bollinger


# 2) price/SMA
def compute_SMA(df_prices, roll_window=20):

    df_SMA = pd.DataFrame(index=df_prices.index)
    df_SMA['prices'] = df_prices
    df_SMA['sma'] = df_SMA['prices'].rolling(window=roll_window).mean()
    df_SMA['price/sma'] = df_SMA['prices'] / df_SMA['sma']

    return df_SMA


# 3) Moving Average Convergence Divergence (MACD)
def compute_MACD(df_prices, ema1_days=12, ema2_days=26, macd_signal_days=9):

    df_macd = pd.DataFrame(index=df_prices.index)

    df_macd['prices'] = df_prices
    df_macd['ema1'] = df_macd.prices.ewm(span=ema1_days, adjust=False).mean()
    df_macd['ema2'] = df_macd.prices.ewm(span=ema2_days, adjust=False).mean()


    df_macd['macd'] = df_macd['ema1'] - df_macd['ema2']
    df_macd['macd_exp'] = df_macd['macd'].ewm(span=macd_signal_days, adjust=False).mean()

    return df_macd


# 4) Stochastic Oscillator
def compute_StochasticOsc(df_prices, window_k=14, window_d=3):

    df_StochasticOsc = pd.DataFrame(index=df_prices.index)

    df_StochasticOsc['prices'] = df_prices

    # Create the "H14" column in the DataFrame
    df_StochasticOsc['H14'] = df_StochasticOsc['prices'].rolling(window=window_k).max() #.fillna(method='bfill')

    # Create the "L14" column in the DataFrame
    df_StochasticOsc['L14'] = df_StochasticOsc['prices'].rolling(window=window_k).min() #.fillna(method='bfill')

    # Create the "%K" column in the DataFrame
    # %K = 100(C – L14)/(H14 – L14)
    df_StochasticOsc['%K'] = 100 * (df_StochasticOsc['prices'] - df_StochasticOsc['L14']) \
                             / (df_StochasticOsc['H14'] - df_StochasticOsc['L14'])

    # Create the "%D" column in the DataFrame
    # %D = 3-period moving average of %K
    df_StochasticOsc['%D'] = df_StochasticOsc['%K'].rolling(window=window_d).mean()

    return df_StochasticOsc

# 5) Commodity Channel Index (CCI)
def compute_CCI(df_prices, roll_window=20):

    df_CCI = pd.DataFrame(index=df_prices.index)

    # Calculate Rolling Mean
    # SMA = df_prices.rolling(window=roll_window).mean()
    df_CCI['SMA'] = df_prices.rolling(window=roll_window).mean()
    df_CCI['prices'] = df_prices

    df_CCI['CCI'] = (df_CCI['prices'] - df_CCI['SMA']) / (.015 * df_CCI['prices'].std())

    return df_CCI

def author():
    return 'sbiswas67' #Change this to your user ID

def test_code():  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # this is a helper function you can use to test your code  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # note that during autograding his function will not be called.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Define input parameters
    start_date = '2008-1-1'
    end_date = '2009-12-31'
    symbol = 'JPM'

    dates = pd.date_range(start_date, end_date)

    df_prices = get_data([symbol], dates)
    df_prices = df_prices[symbol]
    df_prices = df_prices.fillna(method='ffill').fillna(method='bfill')
    df_prices = normalize_data(df_prices)

    # Bollinger Band
    df_Bollinger = compute_bollingerBBval(df_prices)

    plt.clf()
    plt.figure(figsize=(20, 7))
    plt.plot(df_Bollinger.index, df_Bollinger['prices'], label='Price (Normalized)')
    plt.plot(df_Bollinger.index, df_Bollinger['upper band'], label='Upper Band')
    plt.plot(df_Bollinger.index, df_Bollinger['lower band'], label='Lower Band')
    plt.plot(df_Bollinger.index, df_Bollinger['sma'], label='20d-SMA')
    plt.grid(True)
    plt.title(symbol + ' Bollinger Bands')
    plt.axis('tight')
    plt.xlabel('Date')
    plt.ylabel('Price (Normalized)')
    plt.legend(loc='lower left')
    plt.savefig('bollinger_band.png', bbox_inches='tight')

    # Bollinger Percentage
    plt.clf()
    plt.figure(figsize=(20, 7))
    plt.plot(df_Bollinger.index, df_Bollinger['prices'], label='Price (Normalized)')
    plt.plot(df_Bollinger.index, df_Bollinger['bb %'], label='BB Percentage')
    plt.grid(True)
    plt.title(symbol + ' Bollinger Band Percentage')
    plt.axis('tight')
    plt.xlabel('Date')
    plt.ylabel('Price (Normalized)')
    plt.legend(loc='upper left')
    plt.savefig('bollinger_percent.png', bbox_inches='tight')

    # price/SMA
    df_SMA = compute_SMA(df_prices)
    plt.clf()
    plt.figure(figsize=(20, 7))
    plt.plot(df_SMA.index, df_SMA['prices'], label='Price (Normalized)')
    plt.plot(df_SMA.index, df_SMA['sma'], label='20d-SMA')
    plt.plot(df_SMA.index, df_SMA['price/sma'], label='Price/20d-SMA')
    plt.grid(True)
    plt.title(symbol + ' Simple Moving Average (Price/SMA)')
    plt.axis('tight')
    plt.ylabel('Price (Normalized)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.savefig('sma.png', bbox_inches='tight')

    df_macd = compute_MACD(df_prices)
    plt.clf()
    plt.figure(figsize=(20, 7))
    plt.plot(df_macd.index, df_macd['prices'], label='Price (Normalized)')
    plt.plot(df_macd.index, df_macd['macd'], label='MACD')
    plt.plot(df_macd.index, df_macd['macd_exp'], label='Signal Line')
    plt.grid(True)
    plt.title(symbol + ' MACD Crossover Indicator')
    plt.axis('tight')
    plt.ylabel('Price (Normalized)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.savefig('macd.png', bbox_inches='tight')

    df_StochasticOsc = compute_StochasticOsc(df_prices)
    plt.clf()
    plt.figure(figsize=(20, 7))
    plt.plot(df_StochasticOsc.index, df_StochasticOsc['prices'], label='Price (Normalized)')
    plt.plot(df_StochasticOsc.index, df_StochasticOsc['%K'], label='%K')
    plt.plot(df_StochasticOsc.index, df_StochasticOsc['%D'], label='%D')
    plt.grid(True)
    plt.title(symbol + ' Stochastic Oscillator - Crossover %K and %D')
    plt.axis('tight')
    plt.ylabel('Price (Normalized)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.savefig('StochasticOsc.png', bbox_inches='tight')

    plt.clf()
    plt.figure(figsize=(20, 7))
    plt.plot(df_StochasticOsc.index, df_StochasticOsc['prices'], label='Price (Normalized)')
    plt.grid(True)
    plt.title(symbol + ' Stochastic Oscillator - Price only')
    plt.axis('tight')
    plt.ylabel('Price (Normalized)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.savefig('StochasticOsc_price.png', bbox_inches='tight')

    # CCI
    df_CCI = compute_CCI(df_prices)
    plt.clf()
    plt.figure(figsize=(20, 7))
    plt.plot(df_CCI.index, df_CCI['prices'], label='Price (Normalized)')
    plt.plot(df_CCI.index, df_CCI['CCI'], label='CCI')
    plt.grid(True)
    plt.title(symbol + ' Commodity Channel Index (CCI)')
    plt.axis('tight')
    plt.ylabel('Price (Normalized)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.savefig('CCI.png', bbox_inches='tight')




  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
if __name__ == "__main__":  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_code()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
