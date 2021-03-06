eimport pandas as pd
import numpy as np
import time
from itertools import combinations_with_replacement
from nytimesarticle import articleAPI
from yahoo_finance import Share
import cPickle as pickle

'''
Data collection & munging utility functions defined:

-  map_companies_to_files_of_pairs
-  create_cooccurrence_dict
-  merge_dicts (helper function)
-  reconstitute_dictionary
-  map_cooccur_tuple_to_float (helper function)
-  create_and_clean_cooccur_from_dicts
-  get_stock_data (helper function)
-  pull_and_clean_stock_data
-  match_cooccurrence_and_stock_data
'''

def map_companies_to_files_of_pairs(filepath_to_csv, save_filepath_stem):
    """
    Takes: Filepath to csv containing metadata about which companies to create cooccurrence and correlation data for

    Returns: Nothing

    Other Actions: Creates csv's of length 750, where each row is a pair of companies and columns are company1, ticker1, company2, ticker2

    Note: lenth 750 chosen because a single NY Times API key has a limit of 1,000 requests per day; since we may request multiple pages of results for a single pair of companies, we need to limit the number of companies to well below this threshold. This function, together with reconstitute_dictionary, allows for parallelization of the data collection process.
    """
    company_metadata = pd.read_csv(filepath)
    list_of_pairs = []

    # Append all possible pairs to list_of_pairs
    for pair in combinations_with_replacement(range(len(company_metadata)), 2):
        list_of_pairs.append((company_metadata['Name'].ix[pair[0]],
                              company_metadata['Symbol'].ix[pair[0]],
                              company_metadata['Name'].ix[pair[1]],
                              company_metadata['Symbol'].ix[pair[1]]))
    list_of_pairs_df = pd.Dataframe(list_of_pairs,
                                    columns=['company1', 'ticker1',
                                             'company2', 'ticker2'])

    # Break list_of_pairs_df into chunks of length 750 and save to .csv
    i = 0
    while i < np.ceil(float(len(list_of_pairs))/750):
        list_of_pairs_df.ix[i*750:(i+1)*750].to_csv(save_filepath_stem+int(i)+'.csv')
        i += 1


def create_cooccurrence_dict(filepath_to_pairs_csv, api_key):
    """
    Takes: Filepath to csv containing pairs of companies to create cooccurrence counts for, with column names company1, company2, ticker1, ticker2

    Returns: Nothing

    Other Actions: Creates dictionary of cooccurrences and pickles & saves to location filepath_to_pairs_csv[:-4]+'_cooccur_dict.pkl'
    """
    date1 = 20100101
    date2 = 20150101
    iteration = 0

    # Create NY Times API requestor instance
    api = articleAPI(api_key)

    # Import list of pairs to ping API for
    pairs = pd.read_csv(filepath_to_pairs_csv, usecols=[1, 2, 3, 4])

    # Create co-occurrences dictionary of dictionaries
    cooccur = {}
    with open(filepath[:-4]+'_cooccur_dict.pkl', 'wb') as f:
        pickle.dump(cooccur, f, pickle.HIGHEST_PROTOCOL)

    # Create errors list
    errors = []
    with open(filepath[:-4]+'_cooccur_errors.pkl', 'wb') as g:
        pickle.dump(errors, g, pickle.HIGHEST_PROTOCOL)

    for row_index in pairs.index:
        # Start keeping track of how many publication dates we've collected for this article, and initialize a high value for article count
        current_count = 0
        article_count = 100000
        page_offset = 0
        dates = []
        hits = 0
        [company1, company2, ticker1, ticker2] = pairs.ix[row_index].values

        if current_count < article_count:
            # Reload pickled dictionary to re-add new count, then re-pickle
            with open(filepath[:-4]+'_cooccur_dict.pkl', 'rb') as f:
                dictionary = pickle.load(f)

            if company1 == company2:
                search_response = api.search(q=company1, begin_date = date1, end_date=date2, fl='pub_date', page=page_offset, sort='newest')

            else:
                # Query API and get co-occurence count and publication dates, convert these to integers (months before 2017/01), and both the count and list of dates to the dictionary
                search_response = api.search(q="'"+company1+"' '"+company2+"'", begin_date = date1, end_date=date2, fl='pub_date', page=page_offset)

            # Update article count with actual value
            article_count = int(search_response['response']['meta']['hits'])

            # Start adding publication dates to dates list, as # of years old as of 01/01/2017
            for date in search_response['response']['docs']:
                dates.append(float(2015*12-12*int(date['pub_date'][:4])-int(date['pub_date'][5:7]))/12)

            # Update number of articles we've already seen, and the page offset
            current_count += len(search_response['response']['docs'])
            page_offset += 1

            hits = search_response['response']['meta']['hits']

        # If the co-occurrence dictionary already contains key ticker1, update it, otherwise create the new key and update the value [to avoid over-writing when we recombine], and same for ticker2
        if ticker1 in cooccur:
            cooccur[ticker1][ticker2] = (hits, dates)
        else:
            cooccur[ticker1] = {ticker2 : (hits, dates)}

        if ticker2 in cooccur:
            cooccur[ticker2][ticker1] = (hits, dates)
        else:
            cooccur[ticker2] = {ticker1 : (hits, dates)}

        # Print tracking points to make sure everything is working correctly
        print pairs.iloc[row_index], row_index

        # Pickle & save results
        with open(filepath[:-4]+'_cooccur_dict.pkl', 'wb') as f:
            pickle.dump(cooccur, f, pickle.HIGHEST_PROTOCOL)

        # Sleep for 5 seconds to stay under qeury rate limit
        time.sleep(5)


def merge_dicts(a, b, path=None):
    """
    Helper function for reconstituting dictionaries when parallelizing data collection, merges dict2 into dict1
    """
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def reconstitute_dictionary(list_of_filepaths_to_dicts,
                            save_filepath=None,
                            return_=True):
    """
    Reconstitutes non-overlapping dictionaries pickled & saved to list of input filepaths, then option to pickle & save to save_filepath or return resulting dictionary
    """
    reconstituted_dict = {}
    for filepath in list_of_filepaths_to_dicts:
        # Load the pickled dictionary and update reconstituted_dict accordingly
        with open(filepath, 'rb') as f:
            dictionary = pickle.load(f)
        reconstituted_dict = merge_dicts(reconstituted_dict, dictionary)

    # If given a save_filepath, pickle & save resulting reconstituted 'cooccur' dictionary
    if save_filepath:
        with open(save_filepath, 'wb') as f:
            pickle.dump(reconstituted_dict, f, pickle.HIGHEST_PROTOCOL)

    # If return argument set to True, return dictionary
    if return_:
        return reconstituted_dict


def map_cooccur_tuple_to_float(tup, yearly_decay=0.15, downweight=False):
    """Helper function to create and clean cooccurrence dicts to dataframe"""
    if type(tup) == float:
        return tup
    else:
        if downweight:
            mean_age = np.mean(tup[1])
            k = tup[0]//10
            l = tup[0]%10
            count = 0
            for i in range(1,k+1):
                count += 10*((1-yearly_decay)**(i*mean_age))
            count += l*((1-yearly_decay)**((k+1)*mean_age))
            return count
        else:
            return tup[0]


def create_and_clean_cooccur_from_dicts(list_of_filepaths_to_dicts,
                                        save_filepath,
                                        downweight=False,
                                        yearly_decay=0.15):
    """
    Takes: List of filepaths to saved, non-overlapping dictionaries created by create_cooccurrence_dict, and filepath to save resulting cleaned, reconstituted cooccurrence counts dataframe to

    Returns: Nothing

    Other Actions: Saves pickled dataframe of full, cleaned cooccurrence counts to save_filepath
    """
    # Merge dictionaries and change type to dataframe
    if len(list_of_filepaths_to_dicts) > 1:
        cooccur_dict = reconstitute_dictionary(list_of_filepaths_to_dicts)
    else:
        with open(list_of_filepaths_to_dicts[0], 'rb') as f:
            cooccur_dict = pickle.load(f)

    cooccur = pd.DataFrame(cooccur_dict)

    # Map (count, [pub_1_age, pub_2_age, ...]) tuples to a single float
    cooccur = cooccur.applymap(map_cooccur_tuple_to_float)

    # Clean columns; discard any columns with <100 total co-occurrences
    for company in cooccur.index:
        if cooccur.company.sum() < 100:
            cooccur.drop(company, inplace=True)
            cooccur.drop(company, axis=1, inplace=True)

    with open(save_filepath, 'wb') as f:
        pickle.dump(cooccur, f)


def get_stock_data(ticker, start_date, end_date):
    """
    Takes: Ticker symbol (str) and start & end date strings in the form 'YYYY-MM-DD'

    Returns: List of dictionaries of daily pricing data between start and end dates. Sample fields for each day include Open, Close, Volume, etc.
    """
    print ticker
    # Instantiate requestor instance and request historical data
    yahoo = Share(ticker)
    data = yahoo.get_historical(start_date, end_date)
    return data



def pull_and_clean_stock_data(filepath, start_date, end_date):
    """
    Takes: CSV containing metadata about companies to pull pricing data for, and start & end date strings in the form 'YYYY-MM-DD'

    Returns: Nothing

    Action: Creates dictionary of raw pricing data, dataframe of cleaned pricing data, and dataframe of pairwise correlations for the companies contained in the csv specified at filepath, between start_date and end_date.
    """
    companies = pd.read_csv(filepath)
    raw_pricing_info = {}
    for row_index in companies.index:
        series = get_stock_data(companies.Symbol[row_index], start_date, end_date)
        raw_pricing_info[companies.Symbol[row_index]] = series
    print 'Finished collecting stock info'

    # Save & pickle raw stock info, just in case
    with open(filepath[:-4]+'_raw_stock_data_dict.pkl', 'wb') as f:
        pickle.dump(raw_pricing_info, f)

    # Turn raw pricing dict into cleaned Pandas DF
    pricing_df = pd.DataFrame(raw_pricing_info).applymap(lambda x : x['Close']).astype(float)

    # Save & pickle cleaned dataframe of closing prices
    with open(filepath[:-4]+'_cleaned_stock_df.pkl', 'wb') as f:
        pickle.dump(pricing_df, f)

    # Create, save, & pickle dataframe of pairwise correlations
    correlation_df = pd.DataFrame(index=companies, columns=companies)
    for (a, b) in combinations_with_replacement(companies):
        correlation_df.ix[a, b] = pricing_df[a].corr(pricing_df[b])
    with open(filepath[:-4]+'_correlation_df.pkl', 'wb') as f:
        pickle.dump(correlation_df, f)


def match_cooccurrence_and_stock_data(cooccur_filepath,
                                      correlation_filepath):
    """Makes sure cooccurrence and correlation matrices contain same columns & rows. Drops any which occur in one but not the other, returns agreeing versions"""
    with open(cooccur_filepath, 'rb') as f:
        cooccur = pickle.load(f)
    with open(correlation_filepath, 'rb') as f:
        correlation = pickle.load(f)

    for company in list(set(cooccur.index)-set(correlation.index)):
        cooccur.drop(company, inplace=True)
        cooccur.drop(company, axis=1, inplace=True)
    for company in list(set(correlation.index)-set(cooccur.index)):
        correlation.drop(company, inplace=True)
        correlation.drop(company, axis=1, inplace=True)

    return cooccur, correlation
