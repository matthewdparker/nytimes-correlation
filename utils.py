import pandas as pd
import numpy as np
import time
import itertools
from nytimesarticle import articleAPI
from yahoo_finance import Share
import cPickle as pickle

"""
List of utility functions [in order]:

- create_cooccurrence_dict()
- get_stock_data()
- calculate_cov_and_corr()
- create_corr_and_cov_dicts()
- munge_stock_metadata()
- separate_stocks_by_field()
- create_cooccurrence_subdict()
- chunk_out()
- reconstitue_dictionary()
- clean_raw_stock_data()

Uses:
separate_stocks_by_field() separates a collection of input csv's containing company metadata into a different collection of csv's sorted by field of your choice (within / common to input data)

munge_stock_metadata() takes a list of input filepaths and changes MarketCap and IPOyear columns to floats where possible, then sorts by MarketCap in descending order. Also adds binary for companies which IPO'd in 2012-2013.

create_cooccurrence_dict() takes a csv containing a list of companies (e.g. one sorted by industry via separate_stocks_by_field) and queries the NY Times API to determine the number of pairwise co-occurrences, stores this info in a dictionary, and pickles it.

calculate_cov_and_corr() takes two series and returns covariance and correlation

create_corr_and_cov_dicts_dict() takes a csv containing a list of companies (e.g. one sorted by industry via separate_stocks_by_field) and queries the Yahoo Finance API for historical data within a specified timeframe, then uses calculate_cov_and_corr() to calculate pairwise covariances and correlations.

chunk_out() takes filepaths of csv's containing metadata corresponding to companies within a specific sector, and chunks that data out into multiple csv's where each contains a 1 in the ipo column for at most 1/20th the number of companies which IPO'd in 2012-2013. This data will then be used by an individual machine to create pairs of recent ipo's and all other companies to ping requests to NY Times and Yahoo Finance API's.

reconstitue_dictionary() takes a list of filepaths to various pickled dictionaries, consolidates them, and pickles & saves the result to a specified filepath
"""





def create_cooccurrence_dict(filepath, api_key):
    """
    Takes: filepath to metadata on companies to create co-occurrences for, optional start and end dates in the format 'YYYYMMDD', and number of largest companies to consider (e.g. n=500 searches Fortune 500)

    Returns: dictionary whose keys are two ticker symbols (as strings) and values are number of times they co-occurred within the New York Times dataset between the start and end dates

    Note: default search is from 01/01/2000 to 01/01/2010
    """

    date1='20000101'
    date2='20111231'
    iteration = 0

    # Create NY Times API requestor instance
    api = articleAPI(api_key)

    # Import company data and create list of pairs to ping API for
    raw_df = pd.read_csv(filepath)
    ipos = raw_df[raw_df.ipo == 1][['Symbol', 'Name']]
    pairs = []
    for i in xrange(len(ipos)):
        for j in xrange(len(raw_df)):
            if i != j:
                pairs.append([[ipos.Symbol.iloc[i], ipos.Name.iloc[i]],[raw_df.Symbol.iloc[j], raw_df.Name.iloc[j]]])

    # Create co-occurrences dictionary of dictionaries
    cooccur = {}
    with open(filepath[:-4]+'cooccur_dict.pkl', 'wb') as f:
        pickle.dump(cooccur, f, pickle.HIGHEST_PROTOCOL)

    # Create errors list
    errors = []
    with open(filepath[:-4]+'cooccur_errors.pkl', 'wb') as g:
        pickle.dump(errors, g, pickle.HIGHEST_PROTOCOL)


    # Unpack pairs
    for pair in pairs:
        ticker1 = pair[0][0]
        ticker2 = pair[1][0]
        company1 = pair[0][1]
        company2 = pair[1][1]

        # Reload pickled dictionary to re-add new count, then re-pickle
        with open(filepath[:-4]+'cooccur_dict.pkl', 'rb') as f:
            dictionary = pickle.load(f)


        # Query API and get co-occurence count
        try:
            search_response = api.search(q="'"+company1+"' '"+company2+"'", begin_date = int(date1), end_date=int(date2))
            count = search_response['response']['meta']['hits']

            # If the co-occurrence dictionary already contains key ticker1, update it, otherwise create the new key and update the value
            #[to avoid over-writing when we recombine]
            if ticker1 in cooccur:
                cooccur[ticker1][ticker2] = count
            else:
                cooccur[ticker1] = {ticker2 : count}


            # Print tracking points
            print pair, count
            print iteration

            # Pickle & save results
            with open(filepath[:-4]+'cooccur_dict.pkl', 'wb') as f:
                pickle.dump(cooccur, f, pickle.HIGHEST_PROTOCOL)

        except:
            with open(filepath[:-4]+'cooccur_errors.pkl', 'rb') as g:
                errors = pickle.load(g)
            errors.append([pair, iteration])
            with open(filepath[:-4]+'cooccur_errors.pkl', 'wb') as g:
                pickle.dump(errors, g, pickle.HIGHEST_PROTOCOL)

        # Update iteration count
        iteration += 1

        # Sleep for 5 seconds to stay under qeury rate limit
        time.sleep(5)







def get_stock_data(ticker, start_date, end_date):
    """
    Takes: ticker symbol (str) and start & end date strings in the form 'YYYY-MM-DD'

    Returns: Pandas series of prices of type data_type at specified frequency between start and end dates (inclusive)
    """
    print ticker
    # Instantiate requestor instance and request historical data
    yahoo = Share(ticker)
    data = yahoo.get_historical(start_date, end_date)

    return data








def calculate_cov_and_corr(series1, series2):
    """
    Takes: Two series (Pandas series-like)

    Returns: covariance and correlation of series (floats)
    """

    # Create dataframe copies of our series to convenietly add columns
    series1_df = pd.DataFrame(series1, columns=['Close'])
    series2_df = pd.DataFrame(series2, columns=['Close'])

    # Change column types to float
    series1_df.Close = series1_df.Close.astype(float)
    series2_df.Close = series2_df.Close.astype(float)

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







def pull_stock_data(filepath, start_date, end_date):
    """
    Takes: ticker symbol (str) and start & end date strings in the form 'YYYY-MM-DD'

    Returns: Nothing

    Action: creates correlation and covariance dictionaries for the companies contained in the csv specified at filepath, between start_date and end_date
    """
    # Import reference data associating names with tickers, join into single dataframe stock_info
    raw_df = pd.read_csv(filepath)

    # Instantiate empty correlation and covariance dictionaries, with an empty sub-dictionary for each ticker symbol
    corr = {}
    cov = {}
    errors = []
    # Instantiate pricing dict, which will hold all pricing data for later covariance & correlation calculations
    pricing = {}
    for i in xrange(len(raw_df.Symbol)):
        # Update pricing dict with historical pricing data from Yahoo Finance API for each ticker
        try:
            series = get_stock_data(raw_df.Symbol[i], start_date, end_date)
            pricing[raw_df.Symbol[i]] = series
        except:
            errors.append([raw_df.Symbol[i], raw_df.Name[i], raw_df.IPOyear[i]])

    print 'Finished collecting stock info'

    with open(filepath[:-4]+'_raw_stock_data.pkl', 'wb') as f:
        pickle.dump(pricing, f)




def munge_stock_metadata(list_of_filepaths):
    """Run once to munge raw stock metadata for multiple csv's at once containing lists of companies. Munges values from string formats to floats where appropriate; will transform MarketCap column into float where possible and sort, then transform IPOyear column to integer where possible"""

    for i in list_of_filepaths:
        df = pd.read_csv(i)

        for row_index in xrange(len(df)):
            df.MarketCap.iloc[row_index] = df.MarketCap.iloc[row_index][1:]
        for row_index in xrange(len(df)):
            if df.MarketCap.iloc[row_index][-1] == 'B':
                df.MarketCap.iloc[row_index] = float(df.MarketCap.iloc[row_index][:-1])*1000000000
            elif df.MarketCap.iloc[row_index][-1] == 'M':
                df.MarketCap.iloc[row_index] = float(df.MarketCap.iloc[row_index][:-1])*1000000
        df = df[df.apply(lambda x: type(x['MarketCap']) == float, axis=1)]
        df = df.sort_values(by='MarketCap', ascending=False)

        df.IPOyear = pd.to_numeric(df.IPOyear, errors='coerce')

        df = pd.read_csv(filepath)
        df['ipo'] = np.logical_and(df['IPOyear']>2011, df.IPOyear<2014)*1
        df.to_csv(filepath)

        df.to_csv(i[:-4]+"_munged.csv")






def separate_stocks_by_field(f, list_of_filepaths):
    stock_info = pd.DataFrame()
    for i in list_of_filepaths:
        stock_info.append(pd.read_csv(i))

    for value in stock_info[f].unique():
        stock_info[stock_info[f] == value].to_csv('data/'+str(value)+'.csv')




def chunk_out(filepath):
    """
    Takes: Filepath to .csv of list of company metadata within an industry

    Returns: Nothing

    Actions: makes copies of .csv, with an added row 'ipo' which is zero for all but 2 of the number of total ipo's [2013-2014], and those non-zero 2 are unique to each copy.
    """
    not_done = True
    n = 0
    while not_done:
        df = pd.read_csv(filepath)
        df['ipo'] = np.logical_and(df['IPOyear']>2011, df.IPOyear<2014)*1
        if n*2 > df.ipo.sum():
            not_done = False
        else:
            count = 0
            for row_index in df.index:
                count += df.ipo.iloc[row_index]
                if count < n*2:
                    df.loc[row_index, 'ipo'] = 0
                elif count >= (n+1)*2:
                    df.loc[row_index, 'ipo'] = 0


            df.to_csv(filepath[:-4]+'_chunked_'+str(n)+'.csv')
            n += 1







def reconstitute_dictionary(list_of_filepaths, save_filepath):
    """
    Takes: list of filepaths to pickled dictionaries

    Returns: Nothing

    Action: loads and consolidates dictionaries from list_of_filepaths, then pickles and saves to save_filepath
    """
    reconstituted_dict = {}
    # For each file
    for filepath in list_of_filepaths:
        # Load the pickled dictionary and update reconstituted_dict accordingly
        with open(filepath, 'rb') as f:
            dictionary = pickle.load(f)
            reconstituted_dict.update(dictionary)

    # Pickle & save resulting reconstituted 'cooccur' dictionary
    with open(save_filepath, 'wb') as f:
        pickle.dump(reconstituted_dict, f, pickle.HIGHEST_PROTOCOL)




def clean_raw_stock_data(pricing_filepath, metadata_filepath, save_filepath):
    """
    Takes: filepath to pickled dictionary full of historical pricing data from Yahoo Finance for multiple stocks, and filepath to Sector.csv containing stock metadata for the sector the pickled dictionary was generated for

    Returns: errors

    Action: creates, pickles, and saves dataframe full of cleaned pricing data
    """
    with open(pricing_filepath, 'rb') as f:
        d = pickle.load(f)

    # Instantiate empty prices df to fill with pricing data
    prices = pd.DataFrame().astype(float)
    errors = []
    # Read in sector metadata, convert tickers to list
    sector_tickers_list = pd.read_csv(metadata_filepath)['Symbol'].tolist()

    for ticker in sector_tickers_list:
        price_list = []

        # For each day in the pricing data, append the close price
        try:
            for day_dict in d[ticker]:
                price_list.append(day_dict['Close'])

        except:
            errors.append(ticker)
        # Turn the data into a pandas series to append to prices df
        price_series = pd.Series(price_list)

        # Append price_series to prices df with col name as ticker
        prices[ticker] = price_series

    with open(save_filepath, 'wb') as f:
        pickle.dump(prices, f)

    return errors


def create_cov_and_corr_dicts(pricing_filepath, metadata_filepath, cov_save_filepath, corr_save_filepath):
    """
    Takes: filepath to pickled dictionary of raw Yahoo Finance pricing data for a specific sector, filepath to metadata about companies within the sector, and filepaths to pickle & save the resulting dictionaries to.

    Returns: List containing tickers which produced errors

    Action: creates, saves, and pickles correlation and covariance dictionaries indexed by ticker symbols for the companies contained in the metadata csv which did not produce errors.
    """
    with open(pricing_filepath, 'rb') as f:
        pricing_df = pickle.load(f)

    pricing_df = pricing_df.astype(float)

    stock_info = pd.read_csv(metadata_filepath)
    cov = {}
    corr = {}
    errors = []

    ipo_info = stock_info[stock_info.ipo == 1]
    for row_index in ipo_info.index:

        cov[ipo_info.Symbol[row_index]] = {}
        corr[ipo_info.Symbol[row_index]] = {}

    # For each pair of stocks where the first has ipo'd recently
    for ipo_index in ipo_info.index:
        for stock_index in stock_info.index:
            try:
                # Calculate covariance and correlation
                covariance = pricing_df[ipo_info.Symbol[ipo_index]].cov( pricing_df[stock_info.Symbol[stock_index]])

                correlation = pricing_df[ipo_info.Symbol[ipo_index]].corr( pricing_df[stock_info.Symbol[stock_index]])

                # And update their values in the respective dictionaries
                cov[ipo_info.Symbol[ipo_index]][stock_info.Symbol[stock_index]] = covariance
                corr[ipo_info.Symbol[ipo_index]][stock_info.Symbol[stock_index]] = correlation

                print 'finished with {}, {}: cov = {}, corr = {}'.format(ipo_info.Symbol[ipo_index], stock_info.Symbol[stock_index], covariance, correlation)

            except:
                errors.append(stock_info.Symbol[stock_index])
                print '{} and {} produced an error'.format(ipo_info.Symbol[ipo_index], stock_info.Symbol[stock_index])

    # Pickle and save covariance and correlation dictionaries
    with open(cov_save_filepath, 'wb') as handle:
        pickle.dump(cov, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(corr_save_filepath, 'wb') as handle:
        pickle.dump(corr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if errors:
        return errors


def reshape_pickled_dict(filepath):
    with open(filepath, 'rb') as f:
        d = pickle.load(f)
    raw_df = pd.DataFrame(d).unstack()
    raw_df.to_csv(filepath[:-4]+'_temp.csv')
    df = pd.read_csv(filepath[:-4]+'_temp.csv', header=None)
    df.rename(columns={0:'ipo', 1:'compare', 2:'value'}, inplace=True)
    df.to_csv(filepath[:-4]+'.csv', header=True)
    # Note: when reading in the resulting .csv, set 'usecols=[1, 2, 3]', otherwise the index will be read in as a row


def match_cooccur_and_cov_cor_companies(filepath):
    """
    Takes: filepath to directory containing cooccur, corr, and cov csv's

    Returns: nothing

    Action: cleans cooccur, corr, and cov dicts so they all have the same ipo and compare company lists
    """
    cooccur = pd.read_csv(filepath+'/cooccur.csv', usecols=[1, 2, 3])
    corr = pd.read_csv(filepath+'/corr.csv', usecols=[1, 2, 3])
    cov = pd.read_csv(filepath+'/cov.csv', usecols=[1, 2, 3])

    # Create mask for cooccur and apply
    cooccur_mask = []
    for x in cooccur.index:
        cooccur_mask.append(cooccur.compare.iloc[x] in set(corr.compare.unique()))

    cooccur = cooccur[cooccur_mask]

    # Create mask for corr and cov (same mask) and apply
    corr_mask = []
    for x in corr.index:
        corr_mask.append(corr.ipo.iloc[x] in set(cooccur.ipo.unique()))

    corr = corr[corr_mask]
    cov = cov[corr_mask]

    # Save the results
    corr.to_csv(filepath+'/corr_clean.csv')
    cov.to_csv(filepath+'/cov_clean.csv')
    cooccur.to_csv(filepath+'/cooccur_clean.csv')
