# %% Imports
import numpy as np
import pandas as pd
from datetime import datetime

# %% Path definition
path_price_raw = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/00_data/stocknet-dataset-master/price/raw/AAPL.csv'
path_price_pre = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/00_data/stocknet-dataset-master/price/preprocessed/AAPL.txt'

# %% Data loading
data_raw = pd.read_csv(path_price_raw)
data_pre = pd.read_csv(path_price_pre, delimiter='\t',
                       header=None,
                       names = ['Date',
                                'Movement_percent',
                                'Open',
                                'High',
                                'Low',
                                'Close',
                                'Volume'])    

#%%
data_pre['Date'] = [datetime.strptime(x, '%Y-%m-%d') for x in data_pre.Date]
data_pre = data_pre.sort_values(by='Date',
                                ascending=True,
                                ignore_index=True)

#%% Compute adjusted return
adj_close = list(data_raw['Adj Close'])
adj_close_shifted = [99999] + adj_close[:-1]
adj_return = [x/y-1 for x, y in zip(adj_close, adj_close_shifted)][1:]

#%% 
