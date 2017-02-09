from munging_utils import map_companies_to_files_of_pairs, create_cooccurrence_dict, map_cooccur_tuple_to_float, create_and_clean_cooccur_from_dicts, get_stock_data, pull_and_clean_stock_data, create_correlation_df, match_cooccurrence_and_stock_data
from modeling_utils import cooccurrence_train_to_tfidf, format_X_for_model_as_rows, format_y_for_model
from sklearn.ensemble import RandomForestRegressor
import sys

"""
Sample script to:
    - Retrieve cooccurrence data from 2010-2015 from NY Times for top 25 NASDAQ companies by revenue, and downweight co-occurrences by 15% annually for each year older than 2015

    - Fit & transform cooccurrences with TF-IDF model, and pickle & save fitted TF-IDF transformer for future use on testing data

    - Fit, pickle, and save Random Forest regression model on TF-IDF transformed data for future use on testing data


Note: This script assumes a file 'top_40.csv' containing metadata on largest 40 NASDAQ-listed companies has already been created stored locally. This simply requires sorting and truncating version of .csv file available for download at: http://www.nasdaq.com/screening/company-list.aspx.


Command line execution: python sample_script.py ***API Key***
"""

if __name__ == '__main__':
    api_key = sys.argv[1]

    # Map company metadata to company pairs for cooccurrence retreival
    map_companies_to_files_of_pairs('top_40.csv', 'company_pairs.csv')

    # Create dictionary of cooccurrences for company pairs
    create_cooccurrence_dict('company_pairs0.csv', api_key)

    # Create & clean cooccurrence dataframe, downweighting cooccurrences by 15% for each year they are old
    create_and_clean_cooccur_from_dicts(
                                'company_pairs0_cooccur_dict.pkl',
                                'cooccurrence_df.pkl',
                                downweight=True)

    # Pull and daily stock closing price data for 2015-2017 for companies contained in top_40.csv
    pull_and_clean_stock_data('top_40.csv', '2015-01-01', '2017-01-01')

    # Create dataframe of stock correlations
    create_correlation_df('top_40.csv',
                          'top_40_cleaned_stock_df.pkl',
                          'correlation_df.pkl')

    # Make sure cooccurrence and correlation dataframes have the same values (i.e, no problems getting & cleaning data thus far)
    cooccur, correlation = match_cooccurrence_and_stock_data(
                                       'cooccur_df.pkl',
                                       'correlation_df.pkl'.)

    # Fit TF-IDF transformer and transform cooccurrence dataframe
    tfidf = cooccurrence_train_to_tfidf(cooccur,
                                        'tfidf_transformer.pkl')

    # Format TF-IDF and correlation matrices for modeling
    X_train = format_X_for_model_as_rows(tfidf)
    y_train = format_y_for_model(correlation)

    # Fit Random Forest regression model to data, pickle, and save
    model = RandomForestRegressor(n_estimators=2000, max_depth=5)
    model.fit(X, y)
    with open('random_forest_model_on_tfidf.pkl', 'wb') as f:
        pickle.dump(model, f)
