import numpy
import scipy
from math import *
import csv

# from util import *
# from collections import Counter
#
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# # from sklearn.cross_validation import cross_val_score
# # from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split
# from sklearn import metrics

def average_volatility(raw, start=1, end=0):
    # Return: estimated volatility using all available data
    # parameter:
        # raw - matrix of history price
    r = len(raw)
    count = 0
    # print("Number of records = " + str(r))
    sum = 0
    for i in range (r):
        if i <= start or (end != 0 and i > end):
            continue
        p1 = float(raw[i-1][4])
        p2 = float(raw[i][4])
        temp1 = log(p2/p1)
        # print(temp1)
        sum = sum + temp1
        count = count + 1

    avg = sum/count
    print("Sum is " + str(sum))
    print("Avg is " + str(avg))
    return avg

# Calculate estimated variance - volatility
def get_volatility(raw, start=1, end=0):
    r = len(raw)
    sum = 0
    count = 0
    avg = average_volatility(raw, start=start, end=end)
    for i in range(r):
        if i <= start or (end != 0 and i > end):
            continue
        p1 = float(raw[i-1][4])
        p2 = float(raw[i][4])
        temp1 = log(p2/p1)
        sum = sum + temp1 * temp1
        count = count + 1

    # Use variance formula which \sum (X-E[X])^2 = E[X^2] - E[X]^2
    E_X2 = sum/count
    var = E_X2 - avg * avg
    print("Estimated volatility is " + str(var))
    return var


def main():
    # data = load_data("AAPL_032018.csv", header=1, predict_col=0)
    # print(data.X)
    # print(data.y)

    # Use csv library to import data

    # Load historical option data on 2018/3/2
    with open("../data/AAPL_032018.csv") as data:
        data_reader = csv.reader(data)
        raw_data = list(data_reader)
    # for row in data

    # Load historical price
    with open("../data/AAPL_HP.csv") as data:
        data_reader = csv.reader(data)
        raw_hp = list(data_reader)

    # # Print out headers in raw_data
    # for row in raw_data:
    #     print(row)
    #     break
    #
    # # Print out headers inn raw_hp
    # for row in raw_hp:
    #     print(row)
    #     break

    # Test printing
    for i in range (2):
        print(raw_data[i])


    for i in range (2):
        print(raw_hp[i])


    # Test function
    # average_volatility(raw_hp)
    get_volatility(raw_hp)

if __name__ == "__main__":
    main()
