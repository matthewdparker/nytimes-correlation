# Predicting Stock Correlations with the NY Times Dataset

[Work in progress]

### Abstract

This data science project is aimed at identifying latent signal within the New York Times dataset for future stock correlations. The motivating use case is the portfolio construction problem encountered when deciding whether to buy stock in a company which recently IPO'd; one needs to understand the correlation with the other stocks in the portfolio to accurately assess the impact to the risk profile, but no pricing data is available for back-testing correlations so these quantities must be estimated in other ways.

The project examines the number of co-occurrences of pairs of companies and derives an estimate for the future (24-month) correlation based on these counts, the hypothesis (borne out in the data) being that companies which co-occur disproportionately are somehow similar.


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


### Thanks To:

_Delia Rusu_ - her excellent PyData talks on Estimating Stock Correlations Using Wikipedia provided inspiration and guidance for this project.

_Kyle Polich_ - his podcast Data Skeptic brought Delia's project (and countless other interesting topics) to my attention in the first place.
