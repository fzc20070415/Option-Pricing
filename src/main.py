import numpy as np
import scipy.stats
from math import *
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit
import statistics
import time
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge

### GLOBAL VARIABLES ###
# We set risk-free rate at 3.01%. Data obtained from treasury.gov, using Treasury 20-yr CMT
RATE = 0.0256
DAY = 365
BEST = 1
DATE = '0-0-0'
DATE = '3/20/2018'   ### DEBUG ###
PT_NUM = 10

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
    # print("GET_MATURITY CALLED", raw_trade, raw_expire)
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
        # i = 109 #TEST
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

        # Estimate volatility using brute force
        i_sd = 0.01
        gap = 0.1
        count = 1
        # gap = 0.01#test
        while 1:
            i_c, i_p = bs_call_put(h, s, k, i_sd)
            # print(i_c, i_p, opt_p, opt_t, i_sd)
            if oc:
                if i_c > opt_p:
                    i_sd = i_sd - gap
                    gap = gap * 0.1
                    count = count + 1
            else:
                if i_p > opt_p:
                    i_sd = i_sd - gap
                    gap = gap * 0.1
                    count = count + 1
            # if i_sd > 1:#TEST
            #     exit(1)#TEST
            if count == 16:
                # print(i_sd)
                # exit(1)#TEST
                output.append([i_sd])
                break
            i_sd = i_sd + gap
    return output

# Add hour indicator for report
def pre_process_time(report, raw):
    ### Parameters:
        # report - iv table
        # raw - raw_data

    # Trade time = raw[i][27]
    output = [[0,0,0,0]]
    s = len(raw)
    for i in range(1, s):
        hhmmss = raw[i][27]
        # print(hhmmss)
        (hr, min, sec) = hhmmss.split(':')
        # print(hr, min, sec)
        output.append([float(report[i][0]), int(hr), int(min), float(sec)])
        # print(output[i])
    return output

def convert_input(time='-1', k='-1', p='-1'):
    # Validate input
    try:
        k = float(k)
        # print("k is ", k)
    except:
        # print ("Error (Strike Price):", sys.exc_info()[0])
        k = -1
    try:
        tm = datetime.strptime(time,'%H:%M')
        hr = int(datetime.strftime(tm, '%H'))
        min = int(datetime.strftime(tm, '%M'))
        # print(hr,min)
    except:
        # print ("Error (Time):", sys.exc_info()[0])
        hr = -1
        min = -1
    try:
        p = get_maturity(DATE, p)
    except:
        # print ("Error (Maturity):", sys.exc_info()[0])
        p = -1
    return hr, min, k, p

def build_database(raw, iv, hh, mm, interval=0):
    # Handle overflow
    while mm < 0:
        mm = 60 + mm
        hh = hh - 1
    if hh<=9 and mm<30:
        return [[-1,-1,-1,-1,-1,-1,-1,-1]]

    mm2 = mm + interval

    # Extract useful data based on input to build a temporary database
    database = []
    # Format of database:
        # 0. Option Trade Price
        # 1. Option Maturity/Period (Expiration Date - Trade Date) (in term of year)
        # 2. Strike Price
        # 3. Call or Put
        # 4. Price of Underlying Asset
        # 5. Implied Volatility (Standard Deviation)
        # 6. Minute
        # 7. Second
    size = len(iv)
    while 1:
        for i in range(1, size):
            if int(iv[i][1]) >= hh:
                if int(iv[i][1]) == hh:
                    if int(iv[i][2]) >= mm:
                        if int(iv[i][2]) <= mm2 and float(iv[i][0]) > 0.05:
                            # print("Time", hh, mm)
                            database.append([float(raw[i][5]),                      # Option Trade Price
                                             get_maturity(raw[i][3], raw[i][10]),   # Maturity
                                             float(raw[i][11]),                     # Strike Price
                                             raw[i][12],                            # Put or Call
                                             float(raw[i][26]),                     # Stock Price
                                             float(iv[i][0]),                       # Implied Volatility (sd)
                                             float(iv[i][2]),                       # Minute
                                             float(iv[i][3])                        # Second
                                            ])
                        else:
                            break
                else:
                    break
        if interval != 0 and mm2 >= 60:
            mm2 = mm2 - 60
            mm = 0
            hh = hh + 1
        else:
            break

    # print(database, len(database))
    return database

def func(x, a, b, c):
    return a * (x - b) * (x - b) + c

def inc_seq(min, max):
    x = []
    for i in range(min,max):
        x.append(i)
        x.append(i+0.5)
    return x

def draw_smile_curve(database, current_price, kk, plot=1):
    size = len(database)
    if size == 0:
        print("Empty input database, Smile curve can't be drawn.")
        return -1, -1, -1
    x = np.zeros(size)
    y = np.zeros(size)
    a = [0, 0.2]
    b = []
    for i in a:
        b.append(database[0][4])
    for i in range(size):
        y[i] = pow(database[i][5], 2)
        # y[i] = database[i][5]
        x[i] = database[i][2]



    # Show distribution of data points
    # print(x)
    # plt.scatter(x, y, s=10)
    # plt.plot(b, a)
    # plt.xlabel('Strike Price (K)')
    # plt.ylabel('Implied Volatility (σ^2)')
    # time.sleep(20)

    try:
        # Use relevant data points to fit a curve
        # x_std = inc_seq(int(current_price * 0.5), int(current_price * 1.5))
        x_std = inc_seq(80, 240)
        # popt = estimated [a, b, c]
        popt1, pcov1 = curve_fit(func, x, y, bounds=([0, current_price - 0.02, 0], [np.inf, current_price + 0.02, np.inf]))
        # popt2, pcov2 = curve_fit(func, x,`` y, bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]))
        popt2, pcov2 = curve_fit(func, x, y, bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]))
    except:
        print("Error (draw_smile_curve):", sys.exc_info()[0])
        return -1, -1, -1
    # print(x_std)
    if plot:
        plt.plot(x_std, func(x_std, *popt1), 'r--', label='fit: a=%5.10f, b=%5.3f, c=%5.3f' % tuple(popt1))
        plt.plot(x_std, func(x_std, *popt2), 'g--', label='fit: a=%5.10f, b=%5.3f, c=%5.3f' % tuple(popt2))
        plt.scatter(x, y, s=10)

        plt.legend()
        plt.grid()
        plt.show()
        # plt.figure()    ### DEBUG ####
        pass

    # return popt1
    return popt2

def single_iv_est_quad(raw, iv, hh, mm, kk, pp, v, plot=1):
    # Construct database
    database1 = build_database(raw, iv, hh, mm)
    # print(database1)
    temp_minute = mm
    # Choose data which has similar underlying asset spot price and similar maturity
    database_all = []
    database_5 = []
    current_price = database1[0][4]
    current_maturity = pp
    database2 = []
    database_temp1 = []
    # database_temp2 = []


    while len(database2) < 50 and mm - temp_minute < 10:
        threshold = 0
        # prv_threshold = 0
        temp_minute = temp_minute - 1
        # while len(database_all) <= 50 && :
        database_temp1 = build_database(raw, iv, hh, temp_minute)
        database2 = database2 + database_temp1
    # Not Enough Data
    if len(database2) < PT_NUM:
        if v == 1:
            return -1, -1
        elif v == 2:
            return -1, -1, -1, -1

    # Set current price as the medean price of the obtained data
    if 1:
        underlying_asset_price = []
        for row in database2:
            underlying_asset_price.append(row[4])
        current_price = statistics.median(underlying_asset_price)

    # for row in database2:
    #     if row[4] == current_price and row[5] > 0:
    #         database_all.append(row)
    #         # print(row)
    #
    # while threshold < 1:
    #     prv_threshold = threshold
    #     threshold = 1
    #     for row in database2:
    #         temp = abs(row[4] - current_price)
    #         if temp < threshold and temp > prv_threshold:
    #             threshold = temp
    #     for row in database2:
    #         temp = abs(row[4] - current_price)
    #         if temp <= threshold and temp > prv_threshold and row[5] > 0:
    #             database_all.append(row)

    for row in database2:
        if row[5] > 0:
            database_all.append(row)
            # print(row)

    # print("Checkpoint1")
    # #################### TEMP ############################
    # database3 = []
    # for row in database_all:
    #     if row[2] < 200 and row[2] > 150:
    #         database3.append(row)
    #
    # database_all = database3.copy()
    # #################### TEMP ############################

    # print(database_all)
    threshold = 0
    for row in database_all:
        if row[1] == current_maturity:
            database_5.append(row)
            # print(row)
        # Choose up to 5 points
        if len(database_5) >= PT_NUM:
            break
    # print("Checkpoint2", len(database_5))
    while len(database_5) < PT_NUM and threshold <= 0.5:
        prv_threshold = threshold
        threshold = 1
        for row in database_all:
            # print(row)
            temp = abs(row[1] - current_maturity)
            if temp < threshold and temp > prv_threshold:
                threshold = temp
        for row in database_all:
            temp = abs(row[1] - current_maturity)
            if temp <= threshold and temp > prv_threshold:
                database_5.append(row)

    # Not Enough Data
    if len(database_5) < PT_NUM:
        if v == 1:
            return -1, -1
        elif v == 2:
            return -1, -1, -1, -1

    #### DEBUG ###
    # for row in database_5:
    #     print(row)

    # Draw Smile Curve (X - K , Y - iv = sd^2)
    # size = len(database2)
    # draw_smile_curve(database2[size-10:])
    # draw_smile_curve(database2[size-20:])
    # draw_smile_curve(database2[size-30:])
    # print("Size of data", len(database_5))
    # print(database_5)
    popt_5 = draw_smile_curve(database_5, current_price, kk, plot=plot)
    if popt_5[0] == -1 and v == 1:
        return -1, -1
    popt_all = draw_smile_curve(database_all, current_price, kk, plot=plot)
    if (popt_5[0] == -1 or popt_all[0] == -1) and v == 2:
        return -1, -1, -1, -1

    # Use the estimated curve to predict the volatility
    est_vol_5 = func(kk, *popt_5)
    # print(est_vol_5)
    est_sd_5 = sqrt(est_vol_5)
    est_call_5, est_put_5 = bs_call_put(current_maturity, current_price, kk, est_sd_5)

    print("5 sample test:   The estimated implied volatility = ", est_vol_5)
    print("5 sample test:   The estimated standard deviation = ", est_sd_5)
    print("5 sample test:   The estimated Call option price = ", est_call_5)
    print("5 sample test:   The estimated Put option price  = ", est_put_5)

    if v == 1:
        return est_call_5, est_put_5

    est_vol_all = func(kk, *popt_all)
    est_sd_all = sqrt(est_vol_all)
    est_call_all, est_put_all = bs_call_put(current_maturity, current_price, kk, est_sd_all)

    # print("All sample test: The estimated Call option price =: ", est_call_5)
    # print("All sample test: The estimated Put option price  =: ", est_put_5)

    # if v == 2:
    return est_call_5, est_put_5, est_call_all, est_put_all

# User input time, strike price and maturity
def input_all(THRESHOLD=30):
    # Read input
    # time = input("Input time for evaluation in 12-h format [hh:mm]: ")
    # if time == "exit":
    #     return 0
    # k = input("Input the strike price: ")
    # if k == "exit":
    #     return 0
    # p = input("Input the expiration date in MM/DD/YYYY: ")
    # if p == "exit":
    #     return 0

    ### DEBUG ###
    time = "15:30"
    k = "175"
    p = "10/13/2018"
    ### DEBUG ###

    # Convert input and validate
    hh, mm, kk, pp = convert_input(time, k, p)
    if hh == -1 and kk == -1:
        print("Invalid time and K!")
        return -1
    elif hh == -1:
        print("Invalid time!")
        return -1
    elif kk < 0:
        print("Invalid K! Input should be one real number.")
        return -1
    elif pp < 0:
        print("Invalid maturity!")
        return -1
    else:
        # Date format verified.
        if hh < 9:
            hh = hh + 12
        if hh >= 16 or (hh == 9 and mm < 30):
            print("Invalid Trading Hour requested")
            return -1
        elif hh == 9 and mm <= THRESHOLD:
            print("We cannot estimate option price at 9:30 due to lack of prior data.")
            return -1
        print("Estimating option price at K =", kk, "at", hh, ":", mm, "--")
    return kk, hh, mm, pp

def iv_estimation_quad(raw, iv):
    kk, hh, mm, pp = input_all()

    est_call_5, est_put_5, est_call_all, est_put_all = single_iv_est_quad(raw, iv, hh, mm, kk, pp, v=2)
    if est_call_5 == -1 and est_put_5 == -1:
        print("Not Enough Data to predict option price.")
        return -1

    print("5 sample test:   The estimated Call option price =: ", est_call_5)
    print("5 sample test:   The estimated Put option price  =: ", est_put_5)
    print("All sample test: The estimated Call option price =: ", est_call_5)
    print("All sample test: The estimated Put option price  =: ", est_put_5)

def mass_iv_assessment_quad(raw, iv, plot=0, specific=-1):
    # min_abs_err = 100
    # max_abs_err = 0
    # min_per_err = 100
    # max_per_err = 0
    size = len(raw)
    diff_table = []
    perc_table = []
    diff_call_table = []
    diff_put_table = []
    large_diff_table = []
    # print(len(iv), len(raw))
    for i in range(1, size):  ############ DEBUG ##########
        if specific != -1:
            i = specific
            plot = 1
        # print("Plot is ", plot)
        print(i, iv[i])
        print(raw[i])
        if iv[i][1] == '9' and iv[i][2] == '30':
            # print("9:30 detected")
            continue
        # Ignore those with volatility < 0 (Abnormal data)
        if float(iv[i][0]) < 0:
            continue

        hh, mm, kk, pp = convert_input(raw[i][25], raw[i][11], raw[i][10])
        hh = int(iv[i][1])
        mm = int(iv[i][2])
        # print(hh,mm,kk,pp)
        option_type = raw[i][12]
        # print(hh, mm, kk, pp)
        est_call_5, est_put_5 = single_iv_est_quad(raw, iv, hh, mm, kk, pp, v=1, plot=plot)
        if est_call_5 == -1 and est_put_5 == -1:
            print("Not enought data to predict option price.")
            continue
        actual_price = float(raw[i][5])
        print("Actual Option:", option_type, actual_price)
        diff = 0
        if option_type == 'C':
            diff = abs(est_call_5 - actual_price)
            diff_call = est_call_5 - actual_price
            diff_call_table.append(diff)
        elif option_type == 'P':
            diff = abs(est_put_5 - actual_price)
            diff_put = est_put_5 - actual_price
            diff_put_table.append(diff)
        else:
            print("Option Type Error: ", option_type)
            continue
        diff_table.append(diff)
        perc_table.append(diff/actual_price)
        if diff/actual_price > 0.15:
            if option_type == 'C':
                large_diff_table.append([i, diff/actual_price, option_type, est_call_5, actual_price])
            if option_type == 'P':
                large_diff_table.append([i, diff/actual_price, option_type, est_put_5, actual_price])

        print("Perc diff is ", diff/actual_price)
        #####DEBUG####
        # if i == 5000:
        #     print("DEBUG: break at i=5000")
        #     break
        #####DEBUG####
        if specific != -1:
            break

    size = len(diff_table)
    avg_abs_diff = sum(diff_table)/size
    avg_perc_diff = sum(perc_table)/size
    avg_diff_call = 0
    avg_diff_put = 0
    if len(diff_call_table) != 0:
        avg_diff_call = sum(diff_call_table)/len(diff_call_table)
    if len(diff_put_table) != 0:
        avg_diff_put = sum(diff_put_table)/len(diff_put_table)
    print("Large Difference Table")
    print(large_diff_table)
    print("Average absolute error is " + str(avg_abs_diff))
    print("Average percentage error is " + str(avg_perc_diff))
    print("Average (est_call_5 - actual_price) = ", avg_diff_call)
    print("Average (est_put_5 - actual_price) = ", avg_diff_put)

############ SVR ####################
# How to Tune parameters of SVR:
#  - C: How harsh error will be penalized (larger C gives bigger penalty)
#  - epsilon: Within this bound, error is not counted
#  - gamma: How curvy your curve will be (smaller gamma gives straighter lines))


def single_iv_estimation_svr(raw, iv, hh, mm, kk, pp, plot=1, method="kr"):
    # Extract database - clean data
    # Limit the range of data
    temp = mm + 4
    database_f = []
    while len(database_f) < 20:
        temp = temp - 5
        database_5 = build_database(raw, iv, hh, temp, 5)
        if database_5[0][0] == -1:
            break
        # Fix Underlying Asset Price
        uap_list = []
        MAX = 0
        MIN = 10000000
        for row in database_5:
            uap_list.append(row[4])
        current_price = statistics.median(uap_list)
        # Filter data based on maturity
        current_maturity = pp
        for row in database_5:
            abs_diff = abs(row[1] - current_maturity)/current_maturity
            if abs_diff < 0.5 and row[5] >0:
                database_f.append(row)
    print("database_f size: ", len(database_f))
    if len(database_f) < 20:
        return -1, -1

    # draw_smile_curve(database_f, current_price, kk, 1)
    # Fit a smile curve based on prepared database_f using SVR
    if method=="svr":
        svr = GridSearchCV(SVR(kernel='rbf'), cv=3,
                       param_grid={
                                    "C": [1e3, 1e4, 1e5],
                                   'epsilon':[0.001, 0.002, 0.003],
                                   "gamma": [1e-6, 1e-5, 1e-4]
                                   })
    # svr = GridSearchCV(SVR(kernel='poly', degree=2, gamma=1e5, epsilon=0.01, C=1e5), cv=3,
    #                    param_grid={
    #                                #  "C": [1e3, 1e4, 1e5],
    #                                # 'epsilon':[0.001, 0.002, 0.003],
    #                                # "gamma": [1e-6, 1e-5, 1e-4, 1e-2]
    #                                })

    # ##### DEBUG #####
    # svr = GridSearchCV(KernelRidge(kernel='poly', degree=2, gamma=1e-3, alpha=1e0), cv=3,
    #                   param_grid={
    #                               # "alpha": [1e0, 0.1, 1e-2, 1e1],
    #                               #"gamma": np.logspace(-2, 2, 5)
    #                               })

    if method=="kr":
        kr = GridSearchCV(KernelRidge(kernel='rbf'), cv=3,
                      param_grid={
                                  "alpha": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                                  "gamma": [1e-5, 1e-4, 1e-2, 1e-1, 1e1, 1e2]
                                  })
    # Form X - List of strike price & y - List of iv
    X = []
    y = []
    for row in database_f:
        X.append([row[2]])
        y.append(row[5]*row[5])
        if row[2] > MAX:
            MAX = row[2]
        if row[2] < MIN:
            MIN = row[2]
    # X.sort()
    # y.sort()

    # print(X)
    # print(y)

    if method=="svr":
        svr.fit(X, y)
    elif method=="kr":
        kr.fit(X, y)
    # print("Estimation Model: SVR")
    # print("Best parameters:", svr.best_params_)
    # print("Best score:", svr.best_score_)
    # print("Estimation Model: KR")
    # print("Best parameters:", kr.best_params_)
    # print("Best score:", kr.best_score_)
    # print(X)
    # Plot
    if plot:
        X_plot = np.linspace(80, 340, 260)[:, None]
        # X_plot = np.linspace(MIN, MAX, 160)[:, None]
        if method=="svr":
            y_svr = svr.predict(X_plot)
            plt.plot(X_plot, y_svr, c='r', label='fit: SVR')
        elif method=="kr":
            y_kr = kr.predict(X_plot)
            plt.plot(X_plot, y_kr, c='grey', label='fir: KR')
        plt.scatter(X, y, s=5, label='Actual Data')
        plt.xlabel('Strike Price (K)')
        plt.ylabel('Implied Volatility (σ^2)')
        plt.legend()
        plt.show()

    # print(X)
    # print(y_svr)

    # Output
    # print(svr.predict([[kk]]))
    if method=="svr":
        est_vol = svr.predict([[kk]])
    elif method=="kr":
        est_vol = kr.predict([[kk]])
    if est_vol[0] < 0:
        return -2, -2
    est_sd = sqrt(est_vol[0])

    est_call, est_put = bs_call_put(current_maturity, current_price, kk, est_sd)
    print("The estimated implied volatility = ", est_vol)
    print("The estimated standard deviation = ", est_sd)
    print("The estimated Call option price = ", est_call)
    print("The estimated Put option price  = ", est_put)
    return est_call, est_put


def iv_estimation_svr(raw, iv):
    kk, hh, mm, pp = input_all(32)
    est_call, est_put = single_iv_estimation_svr(raw, iv, hh, mm, kk, pp)
    print("All sample test: The estimated Call option price =: ", est_call)
    print("All sample test: The estimated Put option price  =: ", est_put)
    return 1

def mass_iv_assessment_svr(raw, iv, plot=0, specific=-1, method="kr"):
    size = len(raw)
    diff_table = []
    perc_table = []
    diff_call_table = []
    diff_put_table = []
    large_diff_table = []
    abnormal_table = []
    # print(len(iv), len(raw))
    for i in range(1, size):  ############ DEBUG ##########
        if specific != -1:
            i = specific
            plot = 1
        # print("Plot is ", plot)
        print(i, iv[i])
        print(raw[i])
        if iv[i][1] == '9' and iv[i][2] == '30':
            # print("9:30 detected")
            continue
        # Ignore those with volatility < 0 (Abnormal data)
        if float(iv[i][0]) < 0:
            continue

        hh, mm, kk, pp = convert_input(raw[i][25], raw[i][11], raw[i][10])
        hh = int(iv[i][1])
        mm = int(iv[i][2])
        # print(hh,mm,kk,pp)
        option_type = raw[i][12]
        # print(hh, mm, kk, pp)
        est_call, est_put = single_iv_estimation_svr(raw, iv, hh, mm, kk, pp, plot=plot, method=method)
        if est_call == -1 and est_put == -1:
            print("Not enought data to predict option price.")
            abnormal_table.append(i)
            continue
        elif est_call == -2 and est_put == -2:
            print("Negative iv detected")
            abnormal_table.append(i)
            continue

        actual_price = float(raw[i][5])
        print("Actual Option:", option_type, actual_price)
        diff = 0
        if option_type == 'C':
            diff = abs(est_call - actual_price)
            diff_call = est_call - actual_price
            diff_call_table.append(diff)
        elif option_type == 'P':
            diff = abs(est_put - actual_price)
            diff_put = est_put - actual_price
            diff_put_table.append(diff)
        else:
            print("Option Type Error: ", option_type)
            continue
        diff_table.append(diff)
        perc_table.append(diff/actual_price)
        if diff/actual_price > 0.15:
            if option_type == 'C':
                large_diff_table.append([i, diff/actual_price, option_type, est_call, actual_price])
            if option_type == 'P':
                large_diff_table.append([i, diff/actual_price, option_type, est_put, actual_price])

        print("Perc diff is ", diff/actual_price)
        #####DEBUG####
        # if i == 5000:
        #     print("DEBUG: break at i=5000")
        #     break
        #####DEBUG####
        if specific != -1:
            break

    size = len(diff_table)
    avg_abs_diff = sum(diff_table)/size
    avg_perc_diff = sum(perc_table)/size
    avg_diff_call = 0
    avg_diff_put = 0
    if len(diff_call_table) != 0:
        avg_diff_call = sum(diff_call_table)/len(diff_call_table)
    if len(diff_put_table) != 0:
        avg_diff_put = sum(diff_put_table)/len(diff_put_table)
    print("Large Difference Table")
    print(large_diff_table)
    print("Average absolute error is " + str(avg_abs_diff))
    print("Average percentage error is " + str(avg_perc_diff))
    print("Average (est_call_5 - actual_price) = ", avg_diff_call)
    print("Average (est_put_5 - actual_price) = ", avg_diff_put)

def main():
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
        print("IV csv loaded")
        with open("../data/IV.csv") as data:
            data_reader = csv.reader(data)
            iv_list = list(data_reader)
            # print(iv_list[1][0], iv_list[1][1], iv_list[1][2], iv_list[1][3])
    else:
        print("IV csv skipped")

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
                    if k < 2045:
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
        # raw_report = create_iv_csv(raw_data)
        raw_report = iv_list
        report = pre_process_time(raw_report, raw_data)
        print("IV csv created")

    else:
        print("IV Preparation skipped")

    # Formate of iv_table:
        # iv_table[i][iv, hr, min, sec]
        # iv  = ivtable[i][0]
        # hr  = ivtable[i][1]
        # min = ivtable[i][2]
        # sec = ivtable[i][3]

    global DATE
    ####################### QUADRATIC #################
    # Estimate individual option premium from IV (Quadratic)
    if 0:
        DATE = input("Input today's date in MM/DD/YYYY: ")
        h,m,k,p = convert_input(p=DATE)
        if p != 0:
            # print(p)
            print(DATE, "is not a valid date. Exiting")
            exit(1)
        while 1:
            print(" --- Start estimating option premium from implied volatility --- ")
            if (iv_estimation_quad(raw_data, iv_list) == 0):
                break
    else:
        print("Individual IV estimation (Quadratic) skipped")

    # Assess the accuracy of current IV method (Quadratic)
    if 0:
        DATE = raw_data[1][3]
        print("Date is ", DATE)
        h,m,k,p = convert_input(p=DATE)
        if p != 0:
            # print(p)
            print(DATE, "is not a valid date. Exiting")
            exit(1)
        mass_iv_assessment_quad(raw_data, iv_list)
    else:
        print("Mass IV assessment (Quadratic) skipped")

    # Check one specific record in database (Quadratic)
    if 0:
        DATE = raw_data[1][3]
        x = input("Enter the row number for assessment: ")
        mass_iv_assessment_quad(raw_data, iv_list, plot=1, specific=int(x))
    else:
        print("Specific IV assessment (Quadratic) skipped")
    ################ QUADRATIC ####################

    ################ SVR ####################
    # Estimate individual option premium from IV (SVR)
    if 0:
        # DATE = input("Input today's date in MM/DD/YYYY: ")
        h,m,k,p = convert_input(p=DATE)
        if p != 0:
            # print(p)
            print(DATE, "is not a valid date. Exiting")
            exit(1)
        while 1:
            print(" --- Start estimating option premium from implied volatility --- ")
            if (iv_estimation_svr(raw_data, iv_list) == 0):
                break
            break  ### DEBUG ###
    else:
        print("Individual IV estimation (SVR) skipped")

    # Assess the accuracy of current IV method (SVR)
    if 1:
        DATE = raw_data[1][3]
        print("Date is ", DATE)
        h,m,k,p = convert_input(p=DATE)
        if p != 0:
            # print(p)
            print(DATE, "is not a valid date. Exiting")
            exit(1)
        mass_iv_assessment_svr(raw_data, iv_list, plot=0, method="kr")
    else:
        print("Mass IV assessment (SVR) skipped")

    # Check one specific record in database (SVR)
    if 0:
        DATE = raw_data[1][3]
        x = input("Enter the row number for assessment: ")
        mass_iv_assessment_svr(raw_data, iv_list, plot=1, specific=int(x))
    else:
        print("Specific IV assessment (SVR) skipped")

    # write into csv
    if 0:
        with open("../report/report.csv", 'w') as book:
            wr = csv.writer(book, delimiter=',', lineterminator='\n')
            for row in report:
                wr.writerow(row, )
    else:
        print("Not writing into csv")


    print("Task done. Exiting main function...")

if __name__ == "__main__":
    main()
