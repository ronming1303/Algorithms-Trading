# This file is using GGR method to form pairs

import pandas as pd
import numpy as np


# Input adj close data, output SSD matrix and sort pairs
def GGR(filename):
    # clean data first
    cleanclose = pd.read_csv(filename+'.csv', index_col=0)

    # how many days data we have, m
    m = len(cleanclose.index)

    # how man stocks data we have, n
    n = len(cleanclose.columns)

    stocklist = cleanclose.columns

    # use ssd_ij to save all possible SSD
    ssd_matrix = pd.DataFrame(np.zeros([n, n]), index=stocklist, columns=stocklist)
    for i in range(n):
        stock_i = cleanclose[stocklist[i]]
        for j in range(n):
            if j != i:
                stock_j = cleanclose[stocklist[j]]
                stock_ij = stock_i - stock_j
                sample_variance = np.var(stock_ij)
                ssd_ij = sample_variance + (1/m * np.sum(stock_ij))**2
                ssd_matrix.iloc[i, j] = ssd_ij

    print(ssd_matrix)

    # sort pairs, the lower ssd the better
    all_pairs = {}
    for i in range(n):
        stock_i_name = stocklist[i]
        for j in range(i+1, n):
            stock_j_name = stocklist[j]
            stock_ij = cleanclose[stock_i_name] - cleanclose[stock_j_name]
            a = np.var(stock_ij)
            b = (1/m * np.sum(stock_ij))**2
            all_pairs[stock_i_name + " - " + stock_j_name] = [ssd_matrix.iloc[i, j], a, b]

    # We also output a, b. We want SSD small, a big, b small
    all_pairs_df = pd.DataFrame(all_pairs, index=["SSD", 'a', 'b']).T
    all_pairs_df.sort_values(by=["SSD"], inplace=True)
    all_pairs_df.to_csv("GGR_pairs.csv")

