
import requests
from yahoo_fin import stock_info as si

# API documentation: https://polygon.io/docs/getting-started
POLYGON_API_KEY = 'xKi2aeZYxFcpbe8xJXxwHlV7cj50AU6X'

url = f'https://api.polygon.io/v1/meta/symbols/AAPL/company?apiKey={POLYGON_API_KEY}'
r = requests.get(url)
data = r.json()

print(data['listdate'])

print((si.tickers_nasdaq(include_company_data=True)))



# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol=IBM&apikey={POLYGON_API_KEY}'
r = requests.get(url)
data = r.json()

print(data)

print(si.get_company_info('aapl'))