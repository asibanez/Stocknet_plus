'''
Combines price and tweet files into a single file without any
text preprocesing.
Output always include price and may include tweet
'''

# %% Imports
import os
import json
import pandas as pd
from tqdm import tqdm
from glob import glob


# %% Function definitions
#  Generate ticker list
def get_tickers_f(prices_folder, tweets_folder):
    '''
    Extracts the list of tickers for which there are prices and twwets
    '''

    prices = glob(prices_folder + '*')
    prices = [os.path.basename(x) for x in prices]
    prices = [x.strip('.txt') for x in prices]
    prices = set(prices)

    tweets = glob(tweets_folder + '*')
    tweets = [os.path.basename(x) for x in tweets]
    tweets = [x.strip('.txt') for x in tweets]
    tweets = set(tweets)

    ticker_list = sorted(list(prices.intersection(tweets)))

    return ticker_list


# %% Extract tweets for a ticker
def get_tweets_f(ticker, folder):
    '''
    Extracts tweets from multiple file corresponding to a single ticker
    and combines results into lists

    Inputs:
        ticker: The relevant ticker
        folder: Folder including the tweet files for the relevan ticker

    Outpus:
        dates_all: List with tweet dates
        tickers_all: List with ticker (just the relevant ticker)
        tweets_all: List with tweet texts
    '''

    folder = os.path.join(folder, ticker)
    ticker_file_list = glob(folder + '/*')
    tickers_all = []
    dates_all = []
    tweets_all = []
    for file in ticker_file_list:
        date = os.path.basename(file)
        dates_all.append(date)
        tickers_all.append(ticker)
        tweets_1_day = []
        with open(file, 'r') as fr:
            for line in fr.readlines():
                text = json.loads(line)['text']
                text = (' ').join(text)
                tweets_1_day.append(text)
        tweets_all.append(tweets_1_day)

    return dates_all, tickers_all, tweets_all


# %% Extract prices for a ticker
def get_prices_f(ticker, folder):
    '''
    Extracts prices for a specific ticker from a single file and
    combines results in lists

    Inputs:
        ticker: The relevant ticker
        folder: Folder including all price files

    Output:
        output: Dataframe including dates, tickers and price movements
                for the relevant ticker
    '''

    path = os.path.join(folder, ticker + '.txt')
    data = pd.read_csv(path, delimiter='\t',
                       header=None,
                       names=['Date',
                              'Movement_percent',
                              'Open',
                              'High',
                              'Low',
                              'Close',
                              'Volume'])

    ticker = [ticker] * len(data)
    data['Ticker'] = ticker
    output = data[['Date',
                   'Ticker',
                   'Movement_percent']]

    return output


# %% Path definition
tweets_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/01_data/00_stocknet-dataset-master/tweet/preprocessed/'
prices_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/01_data/00_stocknet-dataset-master/price/preprocessed/'
output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/00_data_imbd/01_preprocessed/02_att_mask_1_day_paper_split'

# tweets_folder = '/data/users/sibanez/04_Stocknet_plus/00_data/00_stocknet-dataset-master/tweet/preprocessed/'
# prices_folder = '/data/users/sibanez/04_Stocknet_plus/00_data/00_stocknet-dataset-master/price/preprocessed/'
# output_folder = '/data/users/sibanez/04_Stocknet_plus/00_data/01_preprocessed/'

output_filename = '01_restructured.pkl'

# %% Generate ticker list
ticker_list = get_tickers_f(prices_folder, tweets_folder)

# %% Generate tweet dataframe
Date = []
Ticker = []
Text = []
for ticker in tqdm(ticker_list, desc='Generating tweet dataframe'):
    tweet_date, tweet_ticker, tweet_text = get_tweets_f(ticker, tweets_folder)
    Date += tweet_date
    Ticker += tweet_ticker
    Text += tweet_text

tweet_df = pd.DataFrame({'Date': Date,
                         'Ticker': Ticker,
                         'Text': Text})

# %% Generate price dataframe
price_df = pd.DataFrame()
for ticker in tqdm(ticker_list, desc='Generating price dataframe'):
    price_df_aux = get_prices_f(ticker, prices_folder)
    price_df = pd.concat([price_df, price_df_aux], axis=0)

# %% Remove extra dates from price dataframe
slicer = (price_df.Date >= '2014-01-01') & (price_df.Date < '2016-01-01')
price_df = price_df[slicer]

# %% Remove small price movements from price dataframe
slicer = ~((price_df.Movement_percent > -0.005) &
           (price_df.Movement_percent <= 0.0055))

price_df = price_df[slicer]

# %% Merge tweets and prices on price
output_df = pd.merge(left=price_df,
                     right=tweet_df,
                     how='left',
                     on=['Date', 'Ticker'])
output_df.Text = output_df.Text.fillna('NA')

# %% Double check dataframe sizes
print(f'\nShape tweets dataframe = {tweet_df.shape}')
print(f'Shape prices dataframe = {price_df.shape}')
print(f'Shape output dataframe = {output_df.shape}')

# %% Save results
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
    print(f'Created folder: {output_folder}')
output_path = os.path.join(output_folder, output_filename)
output_df.to_pickle(output_path)
print(f'Output saved to: {output_path}')
