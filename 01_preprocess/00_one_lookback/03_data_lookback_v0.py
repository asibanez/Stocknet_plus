'''
- Generates a dataframe including all dates with prices for all stocks
- Option to convert to 1 lookback day
- Removes entries with no text
'''

# %% Imports
import os
import pandas as pd

# %% Path definition
data_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/01_data/01_preprocessed/02_att_0_day_split_paper'
# data_folder = '/data/users/sibanez/04_Stocknet_plus/00_data/01_preprocessed'

input_filename = '02_preprocessed.pkl'
output_filename = '03_preprocessed_lookback.pkl'

# %% Global initialization
remove_emtpy_entries = True
do_1_day_lookback = False

# %% Load datasets
data_df = pd.read_pickle(os.path.join(data_folder, input_filename))

# %% Generate lookback days
if do_1_day_lookback:
    # Get list of tickers and preprocess dataset
    shifted_text = [9999] + list(data_df['Text_n'][:-1])
    data_df['Text'] = shifted_text
    data_df = data_df[1:]

    # Drop old column
    data_df = data_df.drop(columns='Text_n')

print(f'\nShape full dataframe = {data_df.shape}')

# %% Remove empty entries
if remove_emtpy_entries:
    mask = data_df['Text_n'] != ""
    data_df = data_df[mask]
print(f'\nShape dataframe after removing empty entries = {data_df.shape}')

# %% Save results
output_path = os.path.join(data_folder, output_filename)
data_df.to_pickle(output_path)
print(f'Output saved to: {output_path}')
