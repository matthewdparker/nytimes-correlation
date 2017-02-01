import pandas as pd
import numpy as np


def build_cooccur_obs_and_corr_matrices(filepath_to_sector):
    """
    Takes: filepath to directory for a specific sector containing cleaned co-occurrence and correlation data, saved as cooccur.csv and corr.csv

    Returns: cleand co-occurrence and correlation dataframes with columns 'ipo', 'compare', and 'value' corresponding to datapoints where ipo != compare, and missing values are imputed as 0.
    """
    corr = pd.read_csv(filepath_to_sector+'/corr_clean.csv', usecols=[1, 2, 3])
    cooccur = pd.read_csv(filepath_to_sector+'/cooccur_clean.csv', usecols=[1, 2, 3])

    # Force value column from string (as read in by read_csv) to float
    corr['value'].astype(float, inplace=True)
    cooccur['value'].astype(float, inplace=True)


    # Fill missing values with 0's
    cooccur.fillna(0, inplace=True)
    corr.fillna(0, inplace=True)


    # Drop datapoints where ipo and compare are the same
    to_drop = []
    for row_index in cooccur.index:
        if cooccur.iloc[row_index].ipo == cooccur.iloc[row_index].compare:
            to_drop.append(row_index)
    cooccur.drop(to_drop, inplace=True)

    to_drop2 = []
    for row_index in corr.index:
        if corr.iloc[row_index].ipo == corr.iloc[row_index].compare:
            to_drop2.append(row_index)
    corr.drop(to_drop2, inplace=True)

    return cooccur, corr




def build_cooccur_exp_matrix(cooccur):
    """
    Takes: observed cooccurrence matrix (columns: ipo, compare, value)

    Returns: expected cooccurrence matrix (columns: ipo, compare, value)
    """

    # Create dictionaries corresponding to expected row and column values
    col_avgs = {}
    row_avgs = {}
    total = cooccur.value.sum()
    for col in cooccur.ipo.unique():
        col_avgs[col] = cooccur[cooccur.ipo == col].value.sum()/total
    for row in cooccur.compare.unique():
        row_avgs[row] = cooccur[cooccur.compare == row].value.sum()/total

    # Initialize expected value df as a copy of cooccurrences, then over-write the value column with expected values rounded to nearest integer
    expected_cooccur = cooccur
    for row_index in expected_cooccur.index:
        col = expected_cooccur.ipo[row_index]
        row = expected_cooccur.compare[row_index]
        expected_cooccur.value[row_index] = col_avgs[col]*row_avgs[row]*total

    return expected_cooccur



def build_X_y(filepath_to_sector):
    """
    Takes: filepath to directory for a specific sector containing cleaned co-occurrence and correlation data, saved as cooccur.csv and corr.csv

    Returns: matrix of expected minus observed cooccurrences, normalized by expected cooccurrences, with additive smoothing.
    """
    # Build co-occurrence and expected co-occurrence df's
    cooccur, y = build_cooccur_obs_and_corr_matrices(filepath_to_sector)
    expected_cooccur = build_cooccur_exp_matrix(cooccur)

    # Instantiate X as a copy of cooccur, and drop observations for which both cooccurrences and expected co-occurrences are 0, since these provide no insight into correlation
    X = cooccur
    for row_index in cooccur.index:
        if expected_cooccur.value[row_index] == 0:
            X = X.drop(row_index)

    # Update X to be the proportion by which the observed co-occurrence value is greater than the rounded expected co-occurrence value

    for row_index in X.index:
        X.value[row_index] = ((cooccur.value[row_index] - expected_cooccur.value[row_index]+0.1)/(expected_cooccur.value[row_index]+0.1))

    for row_index in y.index:
        if row_index not in X.index:
            y = y.drop(row_index)

    return X, y
