import requests
import pandas
import io
import datetime
import os

def dataframeFromUrl(url):
    dataString = requests.get(url).content
    parsedResult = pandas.read_csv(io.StringIO(dataString.decode('utf-8')), index_col=0)
    return parsedResult

def stockPriceIntraday(ticker, folder):
    # step 1. Get data online
    url = 'https://www.alphavantage.co/' \
          'query?function=TIME_SERIES_INTRADAY&symbol={name}&interval=1min&outputsize=full' \
          '&apikey=G6DBME0FO4J0SSVS&datatype=csv'.format(name=ticker)
    intraday = dataframeFromUrl(url)

    # step 2. Append if history exists
    file = folder + '/' + ticker + '.csv'
    if os.path.exists(file):
        history = pandas.read_csv(file, index_col=0)
        print(history.shape)
        print(intraday.shape)
        intraday.append(history)
        print(intraday.shape)
### 问题？？？ 拼接的时候数据可能重复？

    # step 3. Inverse based on index
    intraday.sort_index(inplace=True)

    # setp 4. Save
    intraday.to_csv(file)
    print("Intraday for ", ticker, " got!")


# get minute data
stockPriceIntraday('NVDA', folder='/Users/ruming/Documents/GitHub/Algorithms-Trading/Data')


