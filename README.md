# Predicting Stock Correlations with the NY Times Dataset


### Project Summary

This data science project is aimed at identifying latent signal within the New York Times dataset for future stock correlations. The motivating use case is the portfolio construction problem encountered when deciding whether to buy stock in a company which recently IPO'd; one needs to understand the correlation with the other stocks in the portfolio to accurately assess the impact to the risk profile, but no pricing data is available for back-testing correlations so these quantities must be estimated in other ways.

The project examines the number of co-occurrences of pairs of companies and derives an estimate for the future (24-month) correlation based on these counts, the hypothesis being that companies which co-occur disproportionately are somehow similar.



### Data Sources & Dependencies

The project utilizes the NY Times Article Search API to determine co-occurrences counts of pairs of companies within articles, and the Yahoo Finance API for historical pricing data. The project examined NY Times articles from 2010 - 2014 (inclusive), and stock correlations 2015 - 2016 (inclusive). The raw metadata for the companies used (e.g. stock ticker symbols, company names, annual revenue) can be found at http://www.nasdaq.com/screening/company-list.aspx.

Packages Used:

    - nytimesarticle
    - yahoo_finance  
    - sklearn
    - pandas
    - numpy
    - scipy
    - cPickle



### A Guide to Using this Repo

**Metadata** which enable the data collection & munging process are found in the metadata folder. Metadata about companies listed on the NASDAQ and NYSE are found in the respective files, and can be subsetted to just the companies of interest for modeling purposes.

**Data Collection & Munging** tools are found in the file munging_utils.py. It includes functions which automate the collecting and formatting of both the NY Times and the stock correlation data.

**Modeling and Predicting** tools are found in the file modeling_utils.py. It includes functions for applying various transforms to the array of cooccurrences, including TF-IDF, Pointwise Mutual Information, and various dimensionality reduction techniques such as NMF and SVD. 



### Thanks To:

__Delia Rusu__ - her excellent PyData talks on Estimating Stock Correlations Using Wikipedia provided inspiration and guidance for this project.

__Kyle Polich__ - his podcast Data Skeptic brought Delia's project (and countless other interesting topics) to my attention in the first place.
