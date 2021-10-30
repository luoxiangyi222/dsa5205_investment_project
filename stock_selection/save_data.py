import pandas as pd
tickers = []
with open('./data/contrarian_stocks.txt') as file:
    for line in file:
        tickers.append(line.rstrip())

c = pd.read_excel('./data/c.xlsx')[['tickers', 'P/E', 'P/B']]
c[['tickers', 'discarded']] = c['tickers'].str.split('.', 1, expand=True)

c = c[c['discarded'] == 'O'][['tickers', 'P/E', 'P/B']]
c = c[c['tickers'].isin(tickers)]
c = c.set_index('tickers')
print(c)
c.to_csv('./data/contrarian_stock_data.csv', index=True)
tickers = []
with open('./data/momentum_stocks.txt') as file:
    for line in file:
        tickers.append(line.rstrip())

m = pd.read_excel('./data/m.xlsx')[['tickers', 'P/E', 'P/B']]
m[['tickers', 'discarded']] = m['tickers'].str.split('.', 1, expand=True)

m = m[m['discarded'] == 'O'][['tickers', 'P/E', 'P/B']]
m = m[m['tickers'].isin(tickers)]
m = m.set_index('tickers')
print(m)
m.to_csv('./data/momentum_stock_data.csv', index=True)
