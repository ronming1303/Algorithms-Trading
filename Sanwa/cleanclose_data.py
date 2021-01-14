# This module put all data in one sheet

import numpy as np
import pandas as pd
import us_ticker

stocklist = us_ticker.stock_list
length = len(stocklist)


dateindex = pd.read_csv('/Users/ruming/Documents/GitHub/Algorithms-Trading/Data/'+stocklist[0]+'.csv', index_col=0).index
normalized_closed = pd.DataFrame(index=dateindex, columns=stocklist)

for i in range(length):
    try:
        normalized_closed.iloc[:, i] = pd.read_csv('/Users/ruming/Documents/GitHub/Algorithms-Trading/Data/'+stocklist[i]+'.csv', index_col=0)["Adj Close"]
    except:
        pass
# normalized_closed.dropna(inplace=True)

normalized_closed.to_csv("clean_close_data.csv")
print(normalized_closed)

