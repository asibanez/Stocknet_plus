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
                                   return_tensors = 'pt',
                                   padding = 'max_length',
                                   truncation = True,
                                   max_length = seq_len)

    return BERT_encoding

# %% Path definition
data_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/00_data/01_preprocessed'
input_filename = '03_preprocessed_lookback.pkl'
output_filename = '04_BERT_encoded.pkl'

# %% Global initialization
model_name = 'ProsusAI/finbert'
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
seq_len = 256

# %% Load datasets
data_df = pd.read_pickle(os.path.join(data_folder, input_filename))

# %% BERT encoding
columns = [x for x in data_df.columns if 'Text' in x ]
for column in tqdm(columns, desc = 'Encoding text'):
    BERT_encoding = []
    for text in tqdm(data_df[column]):
        BERT_encoding.append(BERT_encode_f(bert_tokenizer, text, seq_len)['input_ids'])
    data_df[column] = BERT_encoding

# %% Save results
data_df.to_pickle(os.path.join(data_folder, output_filename))





















#%% Compute empty headline
"""
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
bert_encoding = bert_tokenizer('hello how are you',
                          return_tensors = 'pt',
                          padding = 'max_length',
                          truncation = True,
                          max_length = seq_len)

empty_token_ids = (empty['input_ids'].squeeze(0).type(torch.LongTensor))
empty_token_types = (empty['token_type_ids'].squeeze(0).type(torch.LongTensor))
empty_att_masks = (empty['attention_mask'].squeeze(0).type(torch.LongTensor))
"""