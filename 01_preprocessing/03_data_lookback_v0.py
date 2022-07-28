'''
- Generates a dataframe including all dates with prices for all stocks
- Adds columns with tweets from n lookback days
- Removes entries with no text
'''

# %% Imports
import os
import pandas as pd
from tqdm import tqdm

# %% Function definitions
def generate_lookback_day_f(data_df, base_date):
    base_col_name = 'Text_n_m_' + str(base_date)
    new_col_name = 'Text_n_m_' + str(base_date+1)
    shifted_text = [9999] + list(data_df[base_col_name][:-1])
    data_df[new_col_name] = shifted_text
    data_df = data_df[1:]
    return data_df

# %% Path definition
data_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/00_data/01_preprocessed'
input_filename = '02_preprocessed.pkl'
output_filename = '03_preprocessed_lookback.pkl'

# %% Global initialization
remove_emtpy_entries = True
lookback_days = 5

# %% Load datasets
data_df = pd.read_pickle(os.path.join(data_folder, input_filename))

# %% Get list of tickers and preprocess dataset
data_df = data_df.rename(columns={'Text_n': 'Text_n_m_0'})
tickers = sorted(list(set(data_df.Ticker)))

# %% Generate lookback days
data_df_out = pd.DataFrame()
for ticker in tqdm(tickers, desc='Generating looback data'):
    df_aux = data_df[data_df.Ticker == ticker].copy()
    for day_back in range(0, lookback_days):
        df_aux = generate_lookback_day_f(df_aux, day_back)

    data_df_out = pd.concat([data_df_out, df_aux], axis=0)

# %% Drop Text_n_m_0
data_df_out = data_df_out.drop(columns='Text_n_m_0')

# %% Remove empty entries
if remove_emtpy_entries:
    text_columns = [x for x in data_df_out.columns if 'Text' in x]
    mask = [True] * len(data_df_out)
    for column in text_columns:
        mask_aux = (data_df_out[column] != "")
        mask = mask & mask_aux
    
    data_df_out = data_df_out[mask]

# %% Save results
print(f'\nShape saved dataframe = {data_df_out.shape}')
data_df_out.to_pickle(os.path.join(data_folder, output_filename))

#%%
