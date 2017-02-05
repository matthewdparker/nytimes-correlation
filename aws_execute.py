from utils2 import create_cooccurrence_dict
import sys

"""
Script retrieves cooccurrence counts within NY Times articles for pairs of companies contained in local .csv file found in local pairsn.csv file, where n is first input after filename.

Sample command line execution: python aws_execute.py 4 ***API Key***
"""

if __name__ == '__main__':
    n = sys.argv[1]
    api_key = sys.argv[2]
    create_cooccurrence_dict('/home/ec2-user/pairs'+n+'.csv', api_key)
