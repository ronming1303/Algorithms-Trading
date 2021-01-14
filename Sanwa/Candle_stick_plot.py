import pandas
import matplotlib
import mplfinance as mpf
import matplotlib.pyplot as plt

# This module can plot candle stick plot
def stockPricePlot(ticker):
    # step 1. Load data
    history = pandas.read_csv("../Data/"+ticker+'.csv', parse_dates=True, index_col=0)[:100]

    # step 2. Data Manipulation
    close = history['Close']
    close = close.reset_index()

    ohlc = history[['Open', 'High', 'Low', 'Close']]
    # ohlc.columns=[['Open', 'High', 'Low', 'Close']]
    print(ohlc)
    # # ohlc = ohlc.reset_index()
    # # ohlc['timestamp'] = ohlc['timestamp'].map(matplotlib.dates .date2num)
    # # step 3. Plot Figure. Subplot1:scatter plot, subplot2:candle stick plot
    # # scatter plot
    subplot1 = plt.subplot2grid((2,1), (0,0), rowspan=1, colspan=1)
    subplot1.scatter(x=close['Date'], y=close['Close'], c='b')
    plt.title(ticker)

    #candle stick plot
    subplot2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)
    mpf.plot(ax=subplot2, data=ohlc, type='candle')

    plt.show()

stockPricePlot('A')
print('it works')