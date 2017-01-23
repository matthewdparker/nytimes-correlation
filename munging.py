import pandas as pd
from utils import create_cooccurrence_dict, create_corr_and_cov_dicts

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

    # Create and pickle co-occurrence dictionary
    create_cooccurrence_dict(nytimes_start_date, nytimes_end_date, n)

    # Create and pickle covariance and correlation dictionaries
    create_corr_and_cov_dicts(stock_start_date, stock_end_date, n)
