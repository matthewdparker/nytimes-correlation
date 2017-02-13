import pandas as pd
import numpy as np
import cPickle as pickle
from itertools import product, combinations
from sklearn.decomposition import NMF, TruncatedSVD, PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
from sklearn.feature_extraction.text import TfidfTransformer

'''
Functions defined:
-  cooccurrence_train_to_pmi
-  cooccurrence_test_to_pmi
-  cooccurrence_to_expected_value
-  cooccurrence_train_to_tfidf
-  cooccurrence_test_to_tfidf
-  cooccurrence_train_to_jaccard_similarity
-  cooccurrence_test_to_jaccard_similarity
-  fit_transform_dimensionality_reduction_and_similarity
-  format_X_for_modeling_as_rows
-  format_y_for_model
'''


# Various tools for tansforming representations of data

def cooccurrence_train_to_pmi(cooccur):
    """
    Takes: cooccurrence matrix

    Returns: matrix of pointwise mutual information values

    Note: PMI is calculated as p(row, column)/(p(row)*p(column))
    """
    pmi = pd.DataFrame(index = cooccur.index, columns=cooccur.columns.values)
    for row in pmi.index:
        for column in pmi.columns.values:
            tot = cooccur.sum().sum()
            rowsum = cooccur.loc[row].sum()
            colsum = cooccur.loc[column].sum()
            pmi.ix[row, column] = float(exp.loc[row, column]*tot)/(rowsum*colsum)
    return pmi


def cooccurrence_test_to_pmi(cooccur_train, cooccur_test):
    """
    Takes: train & test cooccurrence matrices

    Returns: test PMI matrix, with rows as new indices from test and columns as old indices from train

    Note: test must not have any columns not already in train
    """
    cooccur_full = cooccur_test.append(cooccur_train)
    transformed_full = cooccur_train_to_pmi(cooccur_full)
    return transformed_full.ix[len(cooccur_test.index):]


def cooccurrence_to_expected_value(cooccur, smooth_by=0.1):
    """
    Takes: cooccurrence matrix and additive smoothing factor

    Returns: matrix of pairwise smoothed proportional deviations from expected values

    Notes: expected values are calculated as:
                p(row)*p(column)*(total number of cooccurrences)
    and returned values are calculated as:
                (observed value - expected value + smoothing factor) /
                    (expected value + smoothing factor)
    Can be used on testing data alone, training + testing with no new columns, or square matrix with indices in test indices union train indices
    """
    exp = pd.DataFrame(index = cooccur.index, columns=cooccur.columns.values)
    for row in exp.index:
        for column in exp.columns.values:
            tot = cooccur.sum().sum()
            rowsum = cooccur.loc[row].sum()
            colsum = cooccur.loc[column].sum()
            exp.loc[row, column] = float(rowsum*colsum)/tot
    prop_diff = pd.DataFrame(index = cooccur.index, columns=cooccur.columns.values)
    prop_diff = (cooccur - exp + smooth_by) / (exp + smooth_by)
    return prop_diff


def cooccurrence_train_to_tfidf(cooccur, save_filepath):
    """
    Takes: pairwise cooccurrence matrix and filepath to save the fitted TF-IDF transformer to

    Returns: matrix of TF-IDF frequencies, treating the collection of tags associated with each row index as one document, and frequency of occurrence of column names as word counts

    Other Actions: pickles & saves TF-IDF transformer for future use with testing data
    """
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(cooccur)
    with open(save_filepath, 'wb') as f:
        pickle.dump(transformer, f)
    tfidf = pd.DataFrame(tfidf.todense(), index=cooccur.index, columns=cooccur.columns.values)
    return tfidf


def cooccurrence_test_to_tfidf(cooccur, filepath_to_transformer):
    """
    Takes: pairwise cooccurrence matrix and filepath to the fitted TF-IDF transformer

    Returns: matrix of TF-IDF frequencies, treating the collection of tags associated with each row index as one document, and frequency of occurrence of column names as word counts

    Note: will not accept unseen columns which did not appear in training data for TF-IDF
    """
    with open(filepath_to_transformer, 'rb') as f:
        transformer = pickle.load(f)
    tfidf = transformer.transform(cooccur)
    return tfidf.applymap(lambda x : x[0][0])


def cooccurrence_train_to_jaccard_similarity(cooccur_train):
    """
    Takes: pairwise cooccurrence matrix

    Returns: matrix of pairwise similarities between rows using Jaccard distance, treating rows as vectors corresponding to the row index
    """
    jacsim = pd.DataFrame(index=cooccur_train.index, columns=cooccur_train.columns.values)
    for row in jacsim.index:
        for column in jacsim.columns.values:
            jacsim.loc[column, row] = 1-jaccard[cooccur_train[row], cooccur_train[column]]
    return jacsim.applymap(lambda x : x[0][0])


def cooccurrence_test_to_jaccard_similarity(cooccur_test,
                                            cooccur_train):
    """
    Takes: test & train pairwise cooccurrence matrices

    Returns: pairwise Jaccard similarities between pairs of rows of testing data and rows of training data. Returned similarities have index in the testing set and columns in the training set

    Note: testing matrix must have exactly the same columns as training matrix
    """
    cooccur_full = cooccur_test.append(cooccur_train)
    transformed_full = cooccur_train_to_jaccard_similarity(cooccur_full)
    return transformed_full.iloc[len(cooccur_test.index):]


def fit_transform_dimensionality_reduction_and_similarity(cooccur,
                                                          save_filepath,
                                                          distance_metric,
                                                          model_name,
                                                          **kwargs):
    """
    Takes: cooccurrence matrix, filepath to save fitted dimensionality reduction model to, distance metric (values in [-1, 1]), and model name & keyword arguments for dimensionality reduction model

    Returns: matrix of pairwise similarities under dimensionality reduction model

    Other Actions: pickles & saves fitted dimensionality reduction model to save_filepath for future use on training data
    """
    transformer = model_name(kwargs)
    reduced_cooccur = transformer.fit_transform(cooccur)
    # Pickle & save transformer for future use
    with open(save_filepath, 'wb') as f:
        pickle.dump(model, save_filepath)
    # Create dataframe to hold distances between
    transformed_cooccur = pd.DataFrame(index = cooccur.index, columns=cooccur.columns.values)
    for row in transformed_cooccur.index:
        for column in transformed_cooccur.columns.values:
            transformed_cooccur.loc[row, column] = distance_metric(reduced_cooccur.loc[row], reduced_coocur.loc[column])
    return 1-transformed_cooccur




# Various tools for creating, fitting, evaluating, and testing models

def format_X_for_modeling_as_rows(pairwise_df):
    """
    Takes: matrix of pairwise values (e.g. cooccurrence matrix)

    Returns: concatenated rows corresponding to stacked pairs of companies, one from index and one from columns, for modeling

    Notes: First a matrix is created, for which index and columns are companies, and values are the row values associated with company from index, concatenated with row values associated with company from column. Values are therefore 2*len(pairwise_df.columns) long. Function then calls and returns stack() on this new dataframe, so that resulting row is an input vector mapping directly to a target at similar index in correlation.stack().
    """
    pairwise_lower = pairwise_df.copy()
    pairs = [x for x in product(pairwise_df.index, pairwise_df.columns.values)]
    for (a, b) in pairs:
            unstacked[row][column] = pairwise_df.loc[row].append(pairwise_df.loc[column])
    stacked = pd.DataFrame([unstacked.stack()[i].values for i in range(len(pairwise_df.index)*len(pairwise_df.columns))])
    return stacked


def format_y_for_model(pairwise_df):
    if set(pairwise_df.index) == set(pairwise_df.columns.values):
        pairwise_lower = pairwise_df.copy()
        pairs = [x for x in product(pairwise_df.index, pairwise_df.columns.values)]
        for (a, b) in pairs:
            pairwise_lower[b][a] = pairwise_df.ix[b, a]
        return pairwise_lower.stack().ravel()
    else:
        return pairwise_df.stack().ravel()
