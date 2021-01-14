import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime

# setting the period of data
# x/y/z = year/month/day
x = 2015
y = 1
z = 1

def settingperiod(x, y, z):
    start = datetime.datetime(x, y, z)
    end = datetime.date.today()
    return start, end

start, end = settingperiod(x, y, z)

sp_components = open("sp500components.txt")
lines = sp_components.readlines()
stock_list = []
for line in lines:
    if "NyseSymbol" in line:
        line_ny = line[14: -3]
        stock_list.append(line_ny)
    if "NasdaqSymbol" in line:
        line_nas = line[16: -3]
        stock_list.append(line_nas)

# sort by the Alphabet
stock_list.sort()

# get the daily data from start to end
for i in range(len(stock_list)):
    filename = stock_list[i]
    try:
        file = web.DataReader(filename, "yahoo", start, end)
        file.to_csv(path_or_buf='/Users/ruming/Documents/GitHub/Algorithms-Trading/Data/'+ filename + '.csv')
    except:
        pass

