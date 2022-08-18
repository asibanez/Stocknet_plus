'''
Generates a dataframe including all dates with prices for all stocks
and adds columns with tweets from n lookback days
'''

# %% Imports
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


# %% Function definitions
def BERT_encode_f(bert_tokenizer, text, seq_len):
    BERT_encoding = bert_tokenizer(text,
                                   return_tensors='pt',
                                   padding='max_length',
                                   truncation=True,
                                   max_length=seq_len)

    return BERT_encoding


# %% Path definition
data_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/01_data/01_preprocessed/02_att_0_day_split_paper'
# data_folder = '/data/users/sibanez/04_Stocknet_plus/00_data/01_preprocessed'

input_filename = '03_preprocessed_lookback.pkl'
output_filename = '04_BERT_encoded.pkl'

# %% Global initialization
model_name = 'ProsusAI/finbert'
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
seq_len = 256

# %% Load datasets
data_df = pd.read_pickle(os.path.join(data_folder, input_filename))

# %% BERT encoding
bert_token_ids = []
bert_att_masks = []
for text in tqdm(data_df['Text_n'], desc='Encoding BERT'):
    BERT_encoding = BERT_encode_f(bert_tokenizer, text, seq_len)
    bert_token_ids.append(BERT_encoding['input_ids'])
    bert_att_masks.append(BERT_encoding['attention_mask'])

data_df['bert_token_ids'] = bert_token_ids
data_df['bert_att_mask'] = bert_att_masks

# %% Save results
output_path = os.path.join(data_folder, output_filename)
data_df.to_pickle(output_path)
print(f'Output saved to: {output_path}')
