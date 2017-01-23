import pandas as pd
import numpy as np
import requests
import time
from nytimesarticle import articleAPI
from yahoo_finance import Share
import pickle

def create_cooccurrence_dict(date1='20000101', date2='20100101', n=100):
    """
    Takes: optional start and end dates in the format 'YYYYMMDD', and number of largest companies to consider (e.g. n=500 searches Fortune 500)

    Returns: dictionary whose keys are two ticker symbols (as strings) and values are number of times they co-occurred within the New York Times dataset between the start and end dates

    Note: default search is from 01/01/2000 to 01/01/2010
    """
    # Create NY Times API requestor instance
    api = articleAPI('183d226fb68241c8ae9afca9c6f9d24e')

    # Import reference data associating names with tickers, join into single dataframe stock_info
    nasdaq = pd.read_csv('data/NASDAQ_filtered_by_market_cap.csv')
    nyse = pd.read_csv('data/NYSE_filtered_by_market_cap.csv')
    stock_info = nasdaq.append(nyse)

    # Sort and subset to top n companies by market cap
    stock_info.sort('MarketCap', ascending=False)
    stock_info = stock_info[:n]

    # Create co-occurrences dictionary of dictionaries
    cooccur = {}
    for ticker in stock_info.Symbol:
        cooccur[ticker] = {}

    # For each pair of distinct ticker indices
    for i in xrange(len(stock_info.Symbol)):
        for j in xrange(i+1, len(stock_info.Symbol)):
            # Look up the company names and get the number of co-occurrences
            company1 = stock_info.Name.iloc[i]
            company2 = stock_info.Name.iloc[j]
            search_response = api.search(q="'"+company1+"' '"+company2+"'", begin_date = int(date1), end_date=int(date2))
            count=int(search_response['response']['meta']['hits'])

            # Update both values within the co-occurrences dictionary
            ticker1 = stock_info.Symbol.iloc[i]
            ticker2 = stock_info.Symbol.iloc[j]
            cooccur[ticker1][ticker2] = count
            cooccur[ticker2][ticker1] = count

            # Sleep for 5 seconds to stay under max allowed query rate
            time.sleep(5)

    # Pickle and save co-occurrence dictionary
    with open('data/cooccurrence_dict.pickle', 'wb') as handle:
        pickle.dump(cooccur, handle, protocol=pickle.HIGHEST_PROTOCOL)





def get_stock_data(ticker, start_date, end_date):
    """
    Takes: ticker symbol (str) and start & end date strings in the form 'YYYY-MM-DD'

    Returns: Pandas series of prices of type data_type at specified frequency between start and end dates (inclusive)
    """
    # Instantiate requestor instance and request historical data
    yahoo = Share(ticker)
    data = yahoo.get_historical(start_date, end_date)

    # Subset returned data (list of dictionaries) down to just closing prices
    series = pd.Series([x['Close'] for x in data]).astype(float)
    return series




def calculate_cov_and_corr(series1, series2):
    """
    Takes: Two series (Pandas series-like)

    Returns: covariance and correlation of series (floats)
    """

    # Create dataframe copies of our series to convenietly add columns
    series1_df = pd.DataFrame(series1, columns=['Close'])
    series2_df = pd.DataFrame(series2, columns=['Close'])

    # Create 'percent_change' column, the percent increase (/100) with each time period
    series1_df['percent_change'] = (series1_df.Close - series1_df.Close.shift(1))/series1_df.Close.shift(1)
    series2_df['percent_change'] = (series2_df.Close - series2_df.Close.shift(1))/series2_df.Close.shift(1)

    # Calculate mean percentage change for each series
    series1_mean = series1_df.percent_change[1:].mean()
    series2_mean = series2_df.percent_change[1:].mean()

    # Calculate covariance
    covariance = np.sum(series1_df.percent_change[1:]*series2_df.percent_change[1:])/len(series1_df - 2)

    # Calculate variances
    var1 = np.sum((series1_df.percent_change[1:]-series1_mean)**2)/(len(series1_df)-2)
    var2 = np.sum((series2_df.percent_change[1:]-series2_mean)**2)/(len(series2_df)-2)

    # Calculate correlation
    correlation = covariance/((var1**0.5)*(var2**0.5))

    return covariance, correlation





def create_corr_and_cov_dicts(start_date, end_date, n=100):
    # Import reference data associating names with tickers, join into single dataframe stock_info
    nasdaq = pd.read_csv('data/NASDAQ_filtered_by_market_cap.csv')
    nyse = pd.read_csv('data/NYSE_filtered_by_market_cap.csv')
    stock_info = nasdaq.append(nyse)

    # Sort and subset to top n companies by market cap
    stock_info.sort_values(by='MarketCap', ascending=False)
    stock_info = stock_info[:n]

    # Instantiate empty correlation and covariance dictionaries, with an empty sub-dictionary for each ticker symbol
    corr = {}
    cov = {}
    for ticker in stock_info.Symbol:
        corr[ticker] = {}
        cov[ticker] = {}
    for ticker1 in stock_info.Symbol:
        corr[ticker1] = {}
        cov[ticker1] = {}

    # For each pair (regardless of ordering) of distinct ticker indices
    for i in xrange(len(stock_info.Symbol)):
        for j in xrange(i+1, len(stock_info.Symbol)):
            # Get pricing data from Yahoo Finance API
            series1 = get_stock_data(stock_info.Symbol[i], start_date, end_date)
            series2 = get_stock_data(stock_info.Symbol[j], start_date, end_date)

            # Calculate covariance and correlation
            covariance, correlation = calculate_cov_and_corr(series1, series2)

            # Update both values within the covariance and correlation dictionaries
            ticker1 = stock_info.Symbol.iloc[i]
            ticker2 = stock_info.Symbol.iloc[j]
            cov[ticker1][ticker2] = covariance
            cov[ticker2][ticker1] = covariance
            corr[ticker1][ticker2] = correlation
            corr[ticker2][ticker1] = correlation

    # Pickle and save covariance and correlation dictionaries
    with open('data/covariance_dict.pickle', 'wb') as handle:
        pickle.dump(cov, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/correlation_dict.pickle', 'wb') as handle:
        pickle.dump(corr, handle, protocol=pickle.HIGHEST_PROTOCOL)





def munge_stock_metadata():
    """Run once to munge raw downloaded NASDAQ and NYSE data, will transform MarketCap column into float where possible and sort"""
    nasdaq = pd.read_csv('data/NASDAQ.csv')
    nyse = pd.read_csv('data/NYSE.csv')

    for row_index in xrange(len(nasdaq)):
        nasdaq.MarketCap.iloc[row_index] = nasdaq.MarketCap.iloc[row_index][1:]
    for row_index in xrange(len(nasdaq)):
        if nasdaq.MarketCap.iloc[row_index][-1] == 'B':
            nasdaq.MarketCap.iloc[row_index] = float(nasdaq.MarketCap.iloc[row_index][:-1])*1000000000
        elif nasdaq.MarketCap.iloc[row_index][-1] == 'M':
            nasdaq.MarketCap.iloc[row_index] = float(nasdaq.MarketCap.iloc[row_index][:-1])*1000000
    nasdaq = nasdaq[nasdaq.apply(lambda x: type(x['MarketCap']) == float, axis=1)]
    nasdaq = nasdaq.sort('MarketCap', ascending=False)
    nasdaq.to_csv('data/NASDAQ_filtered_by_market_cap.csv')

    for row_index in xrange(len(nyse)):
        nyse.MarketCap.iloc[row_index] = nyse.MarketCap.iloc[row_index][1:]
    for row_index in xrange(len(nyse)):
        if nyse.MarketCap.iloc[row_index][-1] == 'B':
            nyse.MarketCap.iloc[row_index] = float(nyse.MarketCap.iloc[row_index][:-1])*1000000000
        elif nyse.MarketCap.iloc[row_index][-1] == 'M':
            nyse.MarketCap.iloc[row_index] = float(nyse.MarketCap.iloc[row_index][:-1])*1000000
    nyse = nyse[nyse.apply(lambda x: type(x['MarketCap']) == float, axis=1)]
    nyse = nyse.sort('MarketCap', ascending=False)
    nyse.to_csv('data/NYSE_filtered_by_market_cap.csv')