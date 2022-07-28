# %% Imports
import json
import glob
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# %% Path definition
path_pre = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/00_data/stocknet-dataset-master/tweet/preprocessed/AAPL/2014-01-01'

# %% Data loading
fr = open(path_raw)
data_raw = json.load(fr)


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
kuku = glob.glob('C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/00_data/stocknet-dataset-master/tweet/preprocessed/**/*')

tweets_file = []
for path in tqdm(kuku):
    fr = open(path, 'r')
    tweets = []
    for line in fr.readlines():
        text = json.loads(line)['text']
        tweets.append(text)
    tweets_file.append(tweets)
    
lens = [len(x) for x in tweets_file]

kuku = ' '.join(tweets_file[555])

lens_2 = [len(x) for x in tweets_file[555]]

lista = []
for x in tweets_file[555]

kuku = []

tweets_pruned = [x[0:20] for x in tweets_file]
lens_pruned = [len(x) for x in tweets_pruned]

#%%
lista1 = []
for tweet_list in tweets_pruned:
    lista2 = []
    for tweet in tweet_list:
        lista2 += tweet
    lista1.append(lista2)
lens4 = [len(x) for x in lista1]
