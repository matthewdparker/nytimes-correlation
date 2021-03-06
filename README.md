# Predicting Stock Correlations with the NY Times Dataset

*Examining Unconventional, Unstructured Data for Financial Signal*

## Project Summary

This data science project is aimed at identifying latent signal within the New York Times dataset for future stock correlations. The motivating use case is the portfolio construction problem encountered when deciding whether to buy a stock without an extensive pricing history for back-testing, for example one which was just recently issued. One needs to understand the correlation with the other stocks in the portfolio to accurately assess the impact to the risk profile, but no pricing data is available for back-testing correlations so these quantities must be estimated in other ways.

The project examines the number of co-occurrences of pairs of companies and derives an estimate for the future (24-month) correlation between daily closing prices based on these counts, the hypothesis being that companies which co-occur disproportionately are somehow similar.

The project is intended to serve only as a proof-of-concept for the latent predictive power for financial variables within common, unstructured datasets; any production use would require significant optimization and likely commercial access to NY Times API for data collection purposes, due to API access limits.


## Results

Multiple different data manipulation techniques were combined and compared, with the highest *R*<sup>2</sup> value of 0.653 achieved by applying a Pointwise Mutual Information transformation to the matrix of co-occurrences, and comparing cosine similarity of resulting row vectors.

Due to time constraints, I was only able to roll out the model on the top 100 companies by market capitalization across the combined NYSE and NASDAQ. Of these, 17 companies did not fit the data parameters of the model - for example, Facebook only IPO'd in 2012 - for a total dataset of 83 distinct companies, and 3,404 unique pairwise correlations.


![](https://github.com/matthewdparker/nytimes-correlation/blob/master/images/R_Squared_values.png)


Clearly we may conclude there is indeed predictive power for stock correlations within the NY Times dataset, though it's noisy enough that any production model would almost certainly require additional data sources and techniques. 


## Additional Considerations / Next Steps

One primary area of interest for me is to see what the impact might be of adding additional news sources which have minimal overlap with, or different primary missions from, the NY Times; for example, regional business journals (San Francisco Business Times) or industry-specific journals or blogs (TechCrunch). In a similar vein I'm also interested in the effect and mechanics of adding additional regions to the scope of the model, both on the data source and target metrics side; for example, to add the LSE, one would also need to add primarily London- or UK-focused news sources. 

All of the above amount to creating an ensemble model, and one important question that would require further investigation is how to put different data sources on a common footing. For example, San Francisco-based business journals might have more latent signal for correlations between Salesforce and Oracle than a manufacturing industry periodical, but it would likely also mention them together more frequently in articles; the question is, what does the general relationship between frequency and signal look like, and what is the best method for accounting for it in an ensemble model?

Another area of interest which was not investigated fully due to time constraints is the optimization of down-weighting the contributed score of older articles; it's intuitively clear that an article which mentions Amazon.com from 2001 is not doing so in the same business context as one which mentions it from 2014. Mastery, or even optimization, of this time-dependency of signal would allow the model to reach further back into history to identify useful co-occurrences and deliver a more informed final prediction.

The project can also be taken in a more NLP-heavy direction, if the code were modified to return the full article text as well as co-occurrence counts. This would enable sentiment analysis to try to predict the strength and direction of the signal implicit in the article, and introduce more directionality into the model. While I imagine this might be the most time-consuming of the potential modifications, I believe it may have the most potential for performance improvement as well.

Finally, there is the question of benchmarking performance, and quantifying "how good is good" for a model. I would consider some sort of naive prediction based on SIC-code similarity to be a good baseline; again, given time constraints this was not initially within the scope of the project but is a key element in assessing relative improvement.



## Data Sources & Dependencies

The project utilizes the NY Times Article Search API to determine co-occurrences counts of pairs of companies within articles, and the Yahoo Finance API for historical pricing data. The project examined NY Times articles from 2010 - 2014 (inclusive), and stock correlations 2015 - 2016 (inclusive). The raw metadata for the companies used (e.g. stock ticker symbols, company names, annual revenue) can be found at http://www.nasdaq.com/screening/company-list.aspx.

Packages Used:

- nytimesarticle
- yahoo_finance  
- Scikit-Learn
- Pandas
- Numpy
- Scipy
- cPickle


## A Guide to Using this Repo

**Metadata** which enable the data collection & munging process are found in the metadata folder. Metadata about companies listed on the NASDAQ and NYSE are found in the respective files, and can be subsetted to just the companies of interest for modeling purposes.

**Data Collection & Munging** tools are found in the file munging_utils.py. It includes functions which automate the collecting and formatting of both the NY Times and the stock correlation data.

**Modeling and Predicting** tools are found in the file modeling_utils.py. It includes functions for applying various transforms to the array of cooccurrences, including TF-IDF, Pointwise Mutual Information, and various dimensionality reduction techniques such as NMF and SVD.


## Thanks To:

__Delia Rusu__ - her excellent PyData talks on Estimating Stock Correlations Using Wikipedia provided inspiration and guidance for this project.

__Kyle Polich__ - his podcast Data Skeptic brought Delia's project (and countless other interesting topics) to my attention in the first place.
