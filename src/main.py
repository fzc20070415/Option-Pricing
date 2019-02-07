import numpy
import scipy.stats
from math import *
import csv
from datetime import datetime
import matplotlib.pyplot as plt


### GLOBAL VARIABLES ###
# We set risk-free rate at 3.01%. Data obtained from treasury.gov, using Treasury 20-yr CMT
rate = 0.0301

# Calculate average R_t in the given data
def average_volatility(raw, start=1, end=0):
    # parameter:
        # raw - matrix of history price
    r = len(raw)
    count = 0
    # print("Number of records = " + str(r))
    sum = 0
    for i in range (r):
        if i <= start:
            continue
        if end != 0 and i >= end:
            break
        p1 = float(raw[i-1][4])
        p2 = float(raw[i][4])
        temp1 = log(p2/p1)
        # print(temp1)
        sum = sum + temp1
        count = count + 1

    avg = sum/count
    # print("Sum is " + str(sum))
    # print("Avg is " + str(avg))
    return avg

# Calculate estimated variance - volatility
def get_volatility(raw, start=1, end=0):
    # parameter:
        # raw - matrix of history price (including header)
    r = len(raw)
    sum = 0
    count = 0
    avg = average_volatility(raw, start=start, end=end)
    for i in range(r):
        if i <= start:
            continue
        if end != 0 and i >= end:
            break
        p1 = float(raw[i-1][4])
        p2 = float(raw[i][4])
        temp1 = log(p2/p1)
        sum = sum + temp1 * temp1
        count = count + 1

    # Use variance formula which \sum (X-E[X])^2 = E[X^2] - E[X]^2
    E_X2 = sum/count
    var = E_X2 - avg * avg
    # We times volatility of one business day with #business days
    var = 252 * var
    print("Estimated volatility is " + str(var))
    return var

# Calculate estimated call option price
def estimate_call_put(raw, vol, start=1, end=0):
    # parameters:
        # raw - raw_data (including header)
        # vol - estimated volatility
        # start - starting row
        # end - ending row
    r = len(raw)
    # Calculate Standard Deviation
    sd = sqrt(vol)
    # print("Standard Deviation is " + str(sd))
    # Obtain maturity of option

    output = []

    for i in range(r):
        # Get T-t
        if i < start:
            continue
        if i >= end and end != 0:
            break
        raw_trade = raw[i][3]
        raw_expire = raw[i][10]
        trade_date = datetime.strptime(raw_trade, '%m/%d/%Y')
        trade_days = int(datetime.strftime(trade_date, '%j'))
        trade_year = int(datetime.strftime(trade_date, '%Y'))
        expire_date = datetime.strptime(raw_expire, '%m/%d/%Y')
        expire_days = int(datetime.strftime(expire_date, '%j'))
        expire_year = int(datetime.strftime(expire_date, '%Y'))
        expire_days = expire_days + 365 * (expire_year - trade_year)
        day_difference = expire_days - trade_days
        year_difference = day_difference/365
        # print("Maturity is " + str(year_difference))

        # Get Initial Stock price
        S = float(raw[i][26])
        # print("Stock price is " + str(S))

        # Get Strike price
        K = float(raw[i][11])
        # print("Strike price is " + str(K))

        d1 = (1/(sd * sqrt(year_difference))) * (log(S/K) + (rate + (vol/2)) * year_difference)
        # Use d2 = d1 - sqrt(vol) * sqrt(maturity)
        d2 = d1 - ( sd * sqrt(year_difference) )
        # print(d1, d2)

        # We ignore divident here
        N1 = scipy.stats.norm.cdf(d1)
        N2 = scipy.stats.norm.cdf(d2)
        call = (N1 * S) - (N2 * K * exp(rate * year_difference))
        N_1 = 1 - N1
        N_2 = 1 - N2
        put = (N_2 * K * exp(rate * year_difference)) - (S * N_1)
        output.append([call, put])
        # print((call, put))
        # if i == 10:
        #     break
    return output

# Evaluate how accurate my estimation is
def check_estimation(est, raw, start=1, end=0):
    # parameters:
        # est - estimated call & put prices
        # raw - raw_data (including header)
        # start - starting row
        # end - ending row
    r = len(raw)
    diff_table = []
    perc_table = []
    est_l = []
    act_l = []
    for i in range(r):
        if i < start:
            continue
        if i >= end and end != 0:
            break
        act_i = float(raw[i][5])
        act_l.append(act_i)
        if raw[i][12] == 'C':
            est_i = float(est[i-start][0])
            abs_diff = abs(act_i-est_i)
            diff_table.append(abs_diff)
            perc_table.append(abs_diff/act_i)
            est_l.append(est_i)
        elif raw[i][12] == 'P':
            est_i = float(est[i-start][1])
            abs_diff = abs(est_i - act_i)
            diff_table.append(abs_diff)
            perc_table.append(abs_diff/act_i)
            est_l.append(est_i)
        else:
            print("Neither C or P detected at i = " + str(i))

    # Print average absolute difference and percentage difference
    num = 0
    if (end == 0):
        num = r - start
    else:
        num = end - start
    avg_abs_diff = sum(diff_table)/num
    avg_perc_diff = sum(perc_table)/num
    print("Average absolute error is " + str(avg_abs_diff))
    print("Average percentage error is " + str(avg_perc_diff))

    # Plot histograms
    # TODO

    # Plot Dots vs Linear
    # plt.scatter(est_l, act_l)
    # plt.plot([1],[1])
    # plt.xlabel('estimated data')
    # plt.ylabel('actual data')
    # plt.show()

    return avg_abs_diff, avg_perc_diff


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

    # # Test printing
    # for i in range (2):
    #     print(raw_data[i])
    # for i in range (2):
    #     print(raw_hp[i])


    # # Test function (All data)
    # volatility = get_volatility(raw_hp)
    # call_put_table = estimate_call_put(raw_data, volatility)
    # # Evaluation
    # check_estimation(call_put_table, raw_data)

    # Use different volatility to estimate call & put
    len_hp = len(raw_hp)

    report = [['First Period', 'last Period', 'Risk Free Rate', 'Estimated Volatility', 'Avg Abs Err', 'Avg Perc Err']]
    if 1:
        for r in range(0, 500, 5):
            global rate
            print(r)
            rate = rate - r*0.00001
            for i in range(1, len_hp, 20):
                for j in range(0, len_hp, 20):
                    k = len_hp - j - 1
                    if k <= i:
                        break
                    if k <2045:
                        break
                    # print(i, k)
                    # Estimate volatility using data from i to k
                    print("Using rate = " + str(rate))
                    print("Using data from " + raw_hp[i][0] + " to " + raw_hp[k][0])
                    volatility = get_volatility(raw_hp, start=i, end=k)
                    call_put_table = estimate_call_put(raw_data, volatility)

                    avg_abs_err, avg_perc_err = check_estimation(call_put_table, raw_data)
                    report.append([raw_hp[i][0], raw_hp[k][0], rate, volatility, avg_abs_err, avg_perc_err])


    # start=800, end=2200 gives best estimation for volatility for original risk free rate



    # write into csv
    with open("../report/report.csv", 'w') as book:
        wr = csv.writer(book, dialect='excel')
        for row in report:
            wr.writerow(row,)



    print("Task done. Exiting main function.")

if __name__ == "__main__":
    main()
