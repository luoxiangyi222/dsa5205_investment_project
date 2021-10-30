import pandas as pd


c = pd.read_excel('./data/c.xlsx')[['tickers', 'P/E', 'P/B']]
c[['tickers', 'discarded']] = c['tickers'].str.split('.', 1, expand=True)
c = c.set_index('tickers')
c = c[c['discarded'] == 'O'][['P/E', 'P/B']]
print(c)
c.to_csv('./data/contrarian_stock_data.csv', index=True)


m = pd.read_excel('./data/m.xlsx')[['tickers', 'P/E', 'P/B']]
m[['tickers', 'discarded']] = m['tickers'].str.split('.', 1, expand=True)
m = m.set_index('tickers')
m = m[m['discarded'] == 'O'][['P/E', 'P/B']]
print(m)
m.to_csv('./data/momentum_stock_data.csv', index=True)
