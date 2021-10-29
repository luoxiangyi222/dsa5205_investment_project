# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from yahoo_fin import stock_info as si

### load data
# load portfolio
momentum_tickers = []
with open('./data/momentum_portfolio.txt') as file:
    for line in file:
        momentum_tickers.append(line.rstrip())

contrarian_tickers = []
with open('./data/contrarian_portfolio.txt') as file:
    for line in file:
        contrarian_tickers.append(line.rstrip())
#
# price_data = {ticker: si.get_data(ticker, start_date="2020-01-01", end_date="2021-07-31") for ticker in contrarian_tickers}
# combined = reduce(lambda x, y: x.append(y), price_data.values())
combined = pd.read_csv('./data/stock_pool_data.csv', index_col= 0)
combined['time'] = pd.Index(pd.to_datetime(combined.index))
combined = combined.set_index('time')

# calculate stock returns
data_raw = combined[['ticker', 'adjclose']]

data = data_raw.pivot_table(index=data_raw.index, columns='ticker', values=['adjclose'])
# flatten columns multi-index, `date` will become the dataframe index
data.columns = [col[1] for col in data.columns.values]


data = data[contrarian_tickers]
print(data)
returns_portfolio = data.pct_change()
w = [0.1]*10
portfolio_return = returns_portfolio.dot(w)
print(portfolio_return)


# Volatility is given by the annual standard deviation.
covariance = returns_portfolio.cov()*21
portfolio_variance = np.transpose(w)@covariance@w
volatility = np.sqrt(portfolio_variance)



p_ret = [] # Define an empty array for portfolio returns
p_vol = [] # Define an empty array for portfolio volatility
p_weights = [] # Define an empty array for asset weights

num_assets = 10
num_portfolios = 100000

individual_rets = data.resample('M').last().pct_change().mean()


for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, individual_rets) # Returns are the product of individual expected returns of asset and its
                                      # weights
    p_ret.append(returns)
    var = covariance.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
    sd = np.sqrt(var) # Daily standard deviation
    mth_sd = sd*np.sqrt(21) # Annual standard deviation = volatility
    p_vol.append(mth_sd)

df = {'Returns':p_ret, 'Volatility':p_vol}

for counter, symbol in enumerate(data.columns.tolist()):
    #print(counter, symbol)
    df[symbol+' weight'] = [w[counter] for w in p_weights]

portfolios = pd.DataFrame(df)
print(portfolios.head()) # Dataframe of the 10000 portfolios created

# Plot efficient frontier
# portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])

min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
# idxmin() gives us the minimum value in the column specified.
print(min_vol_port)
# Finding the optimal portfolio
rf = 0.016 # risk factor
print(((portfolios['Returns']-rf)/portfolios['Volatility']))
optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
print(optimal_risky_port)

# Plotting optimal portfolio
plt.subplots(figsize=(10, 10))
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)
plt.xlabel('Risk / Volatility')
plt.ylabel('Expected Returns')
plt.show()