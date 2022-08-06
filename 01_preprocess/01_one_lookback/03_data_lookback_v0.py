'''
- Generates a dataframe including all dates with prices for all stocks
- Adds columns with tweets from one lookback day
- Removes entries with no text
'''

# %% Imports
import os
import pandas as pd

# %% Path definition
data_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/01_data/02_preprocessed_one_lookback'
# data_folder = '/data/users/sibanez/04_Stocknet_plus/00_data/01_preprocessed'

input_filename = '02_preprocessed.pkl'
output_filename = '03_preprocessed_lookback.pkl'

# %% Global initialization
remove_emtpy_entries = True

# %% Load datasets
data_df = pd.read_pickle(os.path.join(data_folder, input_filename))

# %% Get list of tickers and preprocess dataset
data_df = data_df.rename(columns={'Text_n': 'Text_n_m_0'})

# %% Generate lookback days
shifted_text = [9999] + list(data_df['Text_n_m_0'][:-1])
data_df['Text_n_m_1'] = shifted_text
data_df = data_df[1:]

# %% Drop Text_n_m_0
data_df = data_df.drop(columns='Text_n_m_0')
print(f'\nShape full dataframe = {data_df.shape}')

# %% Remove empty entries
if remove_emtpy_entries:
    mask = data_df['Text_n_m_1'] != ""
    data_df = data_df[mask]
print(f'\nShape dataframe after removing empty entries = {data_df.shape}')

# %% Save results
output_path = os.path.join(data_folder, output_filename)
data_df.to_pickle(output_path)
print(f'Output saved to: {output_path}')
