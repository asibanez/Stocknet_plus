'''
Preprocesses tweet texts by:
    - Combining tweet lists for a specific ticker and date by:
        - Pruninig the number of tweets in the list to a max value
Output is a dataframe
'''

# %% Imports
import os
import pandas as pd
import matplotlib.pyplot as plt

# %% Path definition
data_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/00_data_imbd/01_preprocessed/02_att_mask_1_day_paper_split'
# data_folder = '/data/users/sibanez/04_Stocknet_plus/00_data/01_preprocessed'

input_filename = '01_restructured.pkl'
output_filename = '02_preprocessed.pkl'

# %% Global initialization
max_tweets_day = 50

# %% Load datasets
data_df = pd.read_pickle(os.path.join(data_folder, input_filename))

# %% EDA
tweets = [x if x != 'NA' else [] for x in list(data_df.Text)]
len_tweets = [len(x) for x in tweets]
_ = plt.hist(len_tweets, range=(1, 200),
             bins=100,  label='tweets per day')
_ = plt.legend()
plt.show()

# %% Prune to max tweets per day
tweets_pruned = [x[0:max_tweets_day] for x in tweets]
len_tweets_pruned = [len(x) for x in tweets_pruned]
_ = plt.hist(len_tweets_pruned, range=(1, 200),
             bins=100, label='tweets per day pruned')
_ = plt.legend()
plt.show()

# %% Flatten lists and replace in dataframe
tweets_flat = [(' ').join(x) for x in tweets_pruned]
data_df['Text'] = tweets_flat

# %% Rename columns and sort ascending
data_df.columns = ['Date', 'Ticker', 'Return_n', 'Text_n']
data_df = data_df.sort_values(by=['Ticker', 'Date'])

# %% Save results
output_path = os.path.join(data_folder, output_filename)
data_df.to_pickle(output_path)
print(f'Output saved to: {output_path}')
