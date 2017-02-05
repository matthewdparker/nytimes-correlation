from utils2 import create_cooccurrence_dict
import sys

if __name__ == '__main__':
    n = sys.argv[1]
    api_key = sys.argv[2]
    create_cooccurrence_dict('/home/ec2-user/pairs'+n+'.csv', api_key)
