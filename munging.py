import pandas as pd
from utils import create_cooccurrence_dict, create_corr_and_cov_dicts, separate_stocks_by_field, munge_stock_metadata

"""Note: pairwise co-occurrence data is stored as dictionary of dictionaries. Each company is a key whose value is a dictionary associated with them, and which has ticker symbols as keys and number of total co-occurrences in NY Times articles as values.

Pairwise correlation and covariance data is stored similarly.
"""

if __name__ == "__main__":
    # Set parameters for munging
    nytimes_start_date = '20010101'
    nytimes_end_date = '20091231'
    stock_start_date = '2010-01-01'
    stock_end_date = '2015-01-01'
    n = 5

    # # Munge data and create sector-specific csv's of company metadata
    # munge_stock_metadata(['data/NASDAQ.csv', 'data/NYSE.csv'])
    # separate_stocks_by_field('Sector', ['data/NASDAQ_munged.csv', 'data/NYSE_munged.csv'])

    # # Read in csv containing all sector values, then for each sector value find associated
    # sectors = pd.read_csv('data/Sector/list_of_sectors.csv')
    # for sector in sectors.iloc[:,1]:
    for sector in ['Technology']:
        # Create and pickle co-occurrence dictionary
        create_cooccurrence_dict('data/Sector/'+sector+'.csv', nytimes_start_date, nytimes_end_date, n)

        # Create and pickle covariance and correlation dictionaries
        create_corr_and_cov_dicts('data/Sector/'+sector+'.csv', stock_start_date, stock_end_date, n)
