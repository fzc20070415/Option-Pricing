import numpy
import scipy.stats
from math import *
import csv
from datetime import datetime
import matplotlib.pyplot as plt


### GLOBAL VARIABLES ###
# We set risk-free rate at 3.01%. Data obtained from treasury.gov, using Treasury 20-yr CMT
RATE = 0.0256
DAY = 365
BEST = 1

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
    var = 260 * var
    print("Estimated volatility is " + str(var))
    return var

def get_maturity(raw_trade, raw_expire):
    trade_date = datetime.strptime(raw_trade, '%m/%d/%Y')
    trade_days = float(datetime.strftime(trade_date, '%j')) * DAY / 365
    trade_year = float(datetime.strftime(trade_date, '%Y'))
    expire_date = datetime.strptime(raw_expire, '%m/%d/%Y')
    expire_days = float(datetime.strftime(expire_date, '%j')) * DAY / 365
    expire_year = float(datetime.strftime(expire_date, '%Y'))
    expire_days = expire_days + (expire_year - trade_year) * DAY
    day_difference = expire_days - trade_days
    h = day_difference / DAY        # Length of maturity
    return h

# Calculate d1 and d2
def get_d1_d2(sd, h, s, k):
    d1 = (1/(sd * sqrt(h))) * ((log(s/k)) + (RATE + ((sd * sd)/2)) * h)
    # Use d2 = d1 - sqrt(vol) * sqrt(maturity)
    d2 = d1 - ( sd * sqrt(h) )
    # print(d1, d2)
    return d1, d2

# Estimate option price using BS formula
def bs_call_put(h, s, k, sd):
    d1, d2 = get_d1_d2(sd, h, s, k)
    # We ignore divident here
    N1 = scipy.stats.norm.cdf(d1)
    N2 = scipy.stats.norm.cdf(d2)
    call = (N1 * s) - (N2 * k * exp(- RATE * h))
    N_1 = 1 - N1
    N_2 = 1 - N2
    put = (N_2 * k * exp(- RATE * h)) - (s * N_1)
    return call, put

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
        year_difference = get_maturity(raw_trade, raw_expire)
        # print("Maturity is " + str(year_difference))

        # Get Initial Stock price
        S = float(raw[i][26])
        # print("Stock price is " + str(S))

        # Get Strike price
        K = float(raw[i][11])
        # print("Strike price is " + str(K))

        # Get call, put option premium
        call, put = bs_call_put(year_difference, S, K, sd)
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
    global BEST
    if BEST > avg_perc_diff:
        BEST = avg_perc_diff
    print("Best result: " + str(BEST))
    # Plot histograms
    # TODO

    # Plot Dots vs Linear
    # plt.scatter(est_l, act_l)
    # plt.plot([1],[1])
    # plt.xlabel('estimated data')
    # plt.ylabel('actual data')
    # plt.show()

    return avg_abs_diff, avg_perc_diff

# Returns a table of implied volatility
def create_iv_csv(raw):
    output = [[0]]
    r = len(raw)
    for i in range(1, r):
        i = 109 #TEST
        # print("i = " + str(i))
        # Option type
        opt_t = raw[i][12]
        oc = 1

        # Set up constants
        # Option price
        opt_p = float(raw[i][5])
        # Strike price
        k = float(raw[i][11])
        # Initial stock price
        s = float(raw[i][26])
        # Period
        raw_trade = raw[i][3]
        raw_expire = raw[i][10]
        h = get_maturity(raw_trade, raw_expire)

        # # Set up BS to obtain N(d1) and N(d2)
        # # Fact1: N(d1) and N(d2) takes value from 0 to 1
        # # Fact2: N(d1) > N(d2)
        # # Thus, I will initialize N(d1) as 1 and lessen it accordingly to see what's the pattern
        # Nd1 = 0.92
        if (opt_t == 'P'):
            # continue
            oc = 0
        #     Nd1 = 0.08
        # # Then, I will set a decreasing interval to find the volatility we are looking for
        # gap = 0.1
        # Nd1 = 0.9974349652396559#TEST
        # # gap = 0.001#TEST
        # # print("i_sd, d1-i_d1, d2-i_d2")
        # count = 0
        # # Call
        # while oc:
        #     print("Nd1 is " + str(Nd1))
        #     # print(h, s, k)
        #     if Nd1 < 0:
        #         print("\n\nerror\n\n")
        #         exit(1)#TEST
        #         Nd1 = Nd1 + gap * 2
        #         gap = gap * 0.1
        #         count = count + 1
        #         print("Gap is now " + str(gap) + "Nd1 is now " + str(Nd1))
        #     if Nd1 >= 1:
        #         Nd1 = 0.9999999999999
        #         gap = 0.001
        #         count = 3
        #
        #     # Compute N(d2)
        #     Nd2 = (Nd1 * s - opt_p) / (k * exp(- RATE * h))
        #     # Derive d1 and d2 using inverse Norm
        #     d1 = scipy.stats.norm.ppf(Nd1)
        #     d2 = scipy.stats.norm.ppf(Nd2)
        #     print(d1, d2)
        #     # Obtain implied standard deviation sinze d1 - d2 = sd * sqrt(maturity period)
        #     i_sd = (d1 - d2) / sqrt(h)
        #     # Check feasibility of d1 and d2 by recalculating d1 d2 using isd
        #     i_d1, i_d2 = get_d1_d2(i_sd, h, s, k)
        #     # Analyse result, adjust gap if necessary (i.e. when the absolute difference become larger)
        #     print(i_sd, d1 - i_d1, d2 - i_d2)
        #     if d1 - i_d1 < 0 or isnan(i_d1):
        #         print("\n\ngap changed\n\n")
        #         Nd1 = Nd1 + gap
        #         gap = gap * 0.1
        #         count = count + 1
        #         print("Gap is now " + str(gap))
        #     if count >= 16:
        #         output.append([i_sd])
        #         print(i_sd)
        #         exit(1)#TEST
        #         break
        #     Nd1 = Nd1 - gap
        #     exit(1)#TEST
        # # Put
        # while not oc:
        #     # print("This is a put option")
        #     if Nd1 > 1:
        #         # print("\n\nerror\n\n")
        #         Nd1 = Nd1 - gap * 2
        #         gap = gap * 0.1
        #         count = count + 1
        #         # break
        #     if Nd1 <= 0:
        #         Nd1 = 0.000001
        #         gap = 0.001
        #         count = 3
        #     # else:
        #     # Compute N(d2)
        #     # print("Nd1 is " + str(Nd1))
        #     Nd2 = (Nd1 * s + opt_p) / (k * exp(- RATE * h))
        #     # Derive d1 and d2 using inverse Norm
        #     d1 = - scipy.stats.norm.ppf(Nd1)
        #     d2 = - scipy.stats.norm.ppf(Nd2)
        #     # print(d1,d2)
        #     # Obtain implied standard deviation sinze d1 - d2 = sd * sqrt(maturity period)
        #     i_sd = (d1 - d2) / sqrt(h)
        #     # Check feasibility of d1 and d2 by recalculating d1 d2 using isd
        #     i_d1, i_d2 = get_d1_d2(i_sd, h, s, k)
        #     # Analyse result, adjust gap if necessary (i.e. when the absolute difference become larger)
        #     # print(i_sd, d1 - i_d1, d2 - i_d2)
        #     if d1 - i_d1 < 0 or isnan(i_d1):
        #         # print("\n\ngap enlarged, " + str(count) + "\n\n")
        #         Nd1 = Nd1 - gap
        #         gap = gap * 0.1
        #         count = count + 1
        #     if count == 16:
        #         output.append([i_sd])
        #         # exit(1)#TEST
        #         print(i_sd)
        #         break
        #     Nd1 = Nd1 + gap


        # Estimate volatility using brute force
        i_sd = 0.01
        gap = 0.1
        count = 1
        gap = 0.01#test
        while 1:
            i_c, i_p = bs_call_put(h, s, k, i_sd)
            print(i_c, i_p, opt_p, opt_t, i_sd)
            # if oc:
            #     if i_c > opt_p:
            #         i_sd = i_sd - gap
            #         gap = gap * 0.1
            #         count = count + 1
            # else:
            #     if i_p > opt_p:
            #         i_sd = i_sd - gap
            #         gap = gap * 0.1
            #         count = count + 1
            if i_sd > 1:
                exit(1)
            if count == 16:
                print(i_sd)
                exit(1)#TEST
                output.append([i_sd])
                break
            i_sd = i_sd + gap
    return output

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

    # Load Prepared Implied Volatility csv {ivlist[i][0]}
    if 1:
        with open("../data/IV.csv") as data:
            data_reader = csv.reader(data)
            iv_list = list(data_reader)
            print(iv_list)
    else:
        print("IV csv skipped")

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

    report = [['First Period', 'last Period', 'Risk Free Rate', 'Estimated Volatility', 'Avg Abs Err', 'Avg Perc Err']]

    ### Task 1: HV ###
    # Test with different volatility and different risk free rate to obtain the least error.
    if 0:
        len_hp = len(raw_hp)
        global RATE
        rate = RATE
        for r in range(0, 500, 50):
            print(r)
            RATE = rate - r*0.00001
            for i in range(1, len_hp, 20):
                for j in range(0, len_hp, 100):
                    k = len_hp - j - 1
                    k = 2045   # fix to the last day
                    if k <= i:
                        break
                    if k <2045:
                        break
                    # print(i, k)
                    # Estimate volatility using data from i to k
                    print("Using rate = " + str(RATE))
                    print("Using data from " + raw_hp[i][0] + " to " + raw_hp[k][0])
                    volatility = get_volatility(raw_hp, start=i, end=k)
                    call_put_table = estimate_call_put(raw_data, volatility)

                    avg_abs_err, avg_perc_err = check_estimation(call_put_table, raw_data)
                    report.append([raw_hp[i][0], raw_hp[k][0], RATE, volatility, avg_abs_err, avg_perc_err])
                    break   # Temporary Test
    else:
        print("HV skipped")

    # start=800, end=2200 gives best estimation for volatility for original risk free rate


    ### Task 2: IV ###

    ### Structure:
        # Use Option price to calculate implied volatility
        # Interpolate a "Smile Curve" using implied volatility and strike price (X: K; Y: sigma)
            # Parabola: y=a(x-b)^2+c
            # How many points to use? Overfitting?
            # ML method?

    # Preparation before computation
    # Generate data for implied volatility
    if 0:
        report = create_iv_csv(raw_data)
        print("IV csv created")
    else:
        print("IV Preparation skipped")

    # Estimate option premium from IV
    if 1:

        pass
    else:
        print("IV estimation skipped")





    # write into csv
    if 1:
        with open("../report/report.csv", 'w') as book:
            wr = csv.writer(book, delimiter=',', lineterminator='\n')
            for row in report:
                wr.writerow(row, )
                #####NOTICE: How to write digits into crv?????
    else:
        print("Not writing into csv")


    print("Task done. Exiting main function.")

if __name__ == "__main__":
    main()
