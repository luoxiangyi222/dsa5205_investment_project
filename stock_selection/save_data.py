import pandas as pd

with open('./data/contrarian_stock_data.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = [line.split()[0] for line in stripped if line]
print(len(lines))
c = pd.read_excel('./data/contrarian_stocks.xlsx')[['P/E', 'P/B']]
c['tickers'] = lines
c = c.set_index('tickers')
print(c)
c.to_csv('./data/contrarian_stock_data.csv', index=True)


with open('./data/momentum_stock_data.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = [line.split()[0] for line in stripped if line]
print(len(lines))
m = pd.read_excel('./data/momentum_stocks.xlsx')[['P/E', 'P/B']]
m['tickers'] = lines
m = m.set_index('tickers')
print(m)
m.to_csv('./data/momentum_stock_data.csv', index=True)
