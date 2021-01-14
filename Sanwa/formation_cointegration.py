# This module uses cointegration method to form pairs

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def cointegration(filename):
    cleanclose = pd.read_csv(filename + '.csv', index_col=0)
    cleanclose.dropna(axis=1, inplace=True)

    # Get the correlation matrix first
    corr_matrix = cleanclose.corr()

    stocklist = cleanclose.columns
    length = len(stocklist)

    pairs_corr = {}
    for i in range(length):
        stock_i = cleanclose[stocklist[i]]
        y = stock_i
        if adfuller(stock_i)[0] < adfuller(stock_i)[4]['5%']:
            i_0diff_station = 1
        else:
            i_0diff_station = 0
            i_lag1 = stock_i.diff(1).dropna()
            if adfuller(i_lag1)[0] < adfuller(i_lag1)[4]['5%']:
                i_1diff_station = 1
            else:
                i_1diff_station = 0
        for j in range(i+1, length):
            stock_j = cleanclose[stocklist[j]]
            x = stock_j
            x = sm.add_constant(x)
            if adfuller(stock_j)[0] < adfuller(stock_j)[4]['5%']:
                j_0diff_station = 1
            else:
                j_0diff_station = 0
                j_lag1 = stock_i.diff(1).dropna()
                if adfuller(j_lag1)[0] < adfuller(j_lag1)[4]['5%']:
                    j_1diff_station = 1
                else:
                    j_1diff_station = 0

            # run the regression of two stocks i and j
            model = sm.OLS(y, x).fit()
            spread_formation = stocklist[i] + '-' + str(model.params[1]) + stocklist[j]

            if adfuller(model.resid)[0] < adfuller(model.resid)[4]['5%']:
                spread_0diff_station = 1
            else:
                spread_0diff_station = 0

            pairs_corr[stocklist[i] + " - " + stocklist[j]] = \
                [corr_matrix.iloc[i, j], i_0diff_station, j_0diff_station, i_1diff_station, j_1diff_station, spread_formation, spread_0diff_station]

    pairs_df = pd.DataFrame(pairs_corr, index=["Correlation", "i ~ I(0)", "j ~ I(0)", "i ~ I(1)", "j ~ I(1)", "spread_formation", "I(0)"]).T
    pairs_df["Absolute Value"] = abs(pairs_df["Correlation"])
    pairs_df.sort_values(by=["Absolute Value"], ascending=False, inplace=True)

    pairs_df.to_csv("cointegration_pairs.csv")

cointegration("clean_close_data")